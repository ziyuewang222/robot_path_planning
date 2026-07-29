import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

from config import DEFAULT_MAP_SIZE, MAP_GENERATOR, SENSOR_RANGE, TEST_SEED, resolve_test_checkpoint_path, resolve_test_fusion_weights_path
from core.models import SemanticUNet
from env.grid_env import GridWorldEnv
from parallel_worker import data_collection_worker
from result_manager import create_run_dirs, save_model_artifacts

class SimpleReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, data_tuple):
        self.buffer.append(data_tuple)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def _build_semantic_label(ground_truth: np.ndarray, rooms) -> np.ndarray:
    sem = np.zeros_like(ground_truth, dtype=np.int64)
    sem[ground_truth == 0] = 1
    for room in rooms:
        x, y, w, h = room["bbox"]
        patch_gt = ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h]
        patch_sem = sem[x + 1:x + 1 + w, y + 1:y + 1 + h]
        patch_sem[patch_gt == 0] = 2
        sem[x + 1:x + 1 + w, y + 1:y + 1 + h] = patch_sem
    return sem


def train_main():
    mp.set_start_method('spawn', force=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== 启动多核并行在线监督学习引擎 | 设备: {device} ===")
    
    num_workers = 12         
    map_size = DEFAULT_MAP_SIZE
    episodes_per_worker = 150  
    batch_size = 32
    buffer_capacity = 3000
    
    model = SemanticUNet(in_channels=1, num_classes=3).to(device)
    print(f"默认 checkpoint 解析为: {resolve_test_checkpoint_path()}")
    print(f"默认融合权重解析为: {resolve_test_fusion_weights_path()}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    memory = SimpleReplayBuffer(capacity=buffer_capacity)

    val_env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    val_explored_map, _ = val_env.reset()
    
    # 【修复重点】：这里直接将 -1, 0, 1 转为 float32 张量
    val_input = torch.from_numpy(val_explored_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    val_target = torch.from_numpy(_build_semantic_label(val_env.ground_truth, val_env.rooms)).unsqueeze(0).to(device)

    def evaluate_fixed_validation(model_eval):
        model_eval.eval()
        with torch.no_grad():
            logits = model_eval(val_input)
            val_loss = criterion(logits, val_target).item()
            pred = torch.argmax(logits, dim=1)
            pixel_acc = (pred == val_target).float().mean().item()
            pred_cpu = pred.squeeze(0).cpu().numpy()
            target_cpu = val_target.squeeze(0).cpu().numpy()
            class_acc = []
            for cls in range(3):
                mask = target_cpu == cls
                if np.any(mask):
                    class_acc.append(float(np.mean(pred_cpu[mask] == cls)))
                else:
                    class_acc.append(float('nan'))
        model_eval.train()
        return val_loss, pixel_acc, class_acc
    
    run_paths = create_run_dirs()
    data_queue = mp.Queue(maxsize=500)
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=data_collection_worker, 
            args=(i, data_queue, episodes_per_worker, map_size)
        )
        p.start()
        processes.append(p)
        
    print(f"已启动 {num_workers} 个数据生成子进程。等待数据注入...")
    
    active_workers = num_workers
    update_count = 0
    total_loss = 0.0
    maps_collected = 0
    episodes_finished = 0
    worker_finished = 0
    worker_episode_counts = {i: 0 for i in range(num_workers)}
    worker_map_counts = {i: 0 for i in range(num_workers)}
    last_data_print_update = 0
    
    pbar = tqdm(desc="Training Steps")
    
    try:
        while active_workers > 0:
            while True:
                try:
                    item = data_queue.get_nowait()
                except Exception:
                    break

                if item is None:
                    active_workers -= 1
                    continue

                if isinstance(item, dict) and item.get("type") == "progress":
                    worker_id = int(item.get("worker_id", -1))
                    if worker_id >= 0:
                        worker_episode_counts[worker_id] = int(item.get("episodes_finished", worker_episode_counts.get(worker_id, 0)))
                        worker_map_counts[worker_id] = int(item.get("maps_collected", worker_map_counts.get(worker_id, 0)))
                    episodes_finished = sum(worker_episode_counts.values())
                    maps_collected = sum(worker_map_counts.values())
                    worker_finished = sum(1 for value in worker_episode_counts.values() if value >= episodes_per_worker)
                    if item.get("worker_finished"):
                        worker_finished = min(num_workers, worker_finished + 1)
                    continue

                memory.push(item)
                maps_collected += 1
            
            if update_count > 0 and update_count - last_data_print_update >= 20:
                episodes_finished = sum(worker_episode_counts.values())
                worker_finished = sum(1 for value in worker_episode_counts.values() if value >= episodes_per_worker)
                print(
                    f"[Data] maps={maps_collected}/{num_workers * episodes_per_worker} | "
                    f"episodes_finished={episodes_finished}/{num_workers * episodes_per_worker} | "
                    f"worker_finished={worker_finished}/{num_workers} | buffer={len(memory)}"
                )
                last_data_print_update = update_count
                    
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                
                inputs = np.array([item[0] for item in batch]).astype(np.float32)
                targets = np.array([item[1] for item in batch]).astype(np.longlong)
                
                inputs_tensor = torch.from_numpy(inputs).unsqueeze(1).to(device)
                targets_tensor = torch.from_numpy(targets).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs_tensor)
                loss = criterion(outputs, targets_tensor)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    pixel_acc = (preds == targets_tensor).float().mean().item()

                total_loss += loss.item()
                update_count += 1
                pbar.update(1)

                if update_count % 10 == 0:
                    avg_loss = total_loss / max(1, update_count)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg': f"{avg_loss:.4f}", 'acc': f"{pixel_acc:.4f}", 'buf': len(memory)})
                    print(f"[Train] update={update_count} | loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | pixel_acc={pixel_acc:.4f} | buffer={len(memory)}")

                if update_count % 20 == 0:
                    val_loss, val_pixel_acc, class_acc = evaluate_fixed_validation(model)
                    class_acc_str = ", ".join(
                        f"c{idx}={acc:.4f}" if not np.isnan(acc) else f"c{idx}=nan"
                        for idx, acc in enumerate(class_acc)
                    )
                    print(
                        f"[Val] update={update_count} | val_loss={val_loss:.4f} | "
                        f"val_pixel_acc={val_pixel_acc:.4f} | {class_acc_str}"
                    )
                    
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在关闭进程...")
        
    finally:
        pbar.close()
        for p in processes:
            p.terminate()
            p.join()
            
        class _StandaloneSaver:
            def save_all_weights(self, checkpoint_path, fusion_path):
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), fusion_path)

        save_model_artifacts(_StandaloneSaver(), run_paths, is_best=True)
        save_path = run_paths.checkpoints_dir / "latest_checkpoint.pth"
        print(f"--> 训练结束，总更新次数: {update_count}。权重已保存至: {save_path}")

if __name__ == "__main__":
    train_main()