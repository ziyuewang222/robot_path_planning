import collections
import queue
import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

from config import DEFAULT_MAP_SIZE, SENSOR_RANGE, TEST_SEED
from core.models import SemanticUNet
from env.grid_env import GridWorldEnv
from parallel_worker import LightweightBFSPlanner, extract_perfect_labels
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


def _masked_ce_loss(logits, targets, inputs):
    """仅在已知区域（输入 != -1）上计算交叉熵，并按样本的已知格数归一化。

    - 排除迷雾像素：模型观测不到迷雾（输入恒为 -1），迷雾上的梯度只会把数据集
      先验灌进网络、并覆盖已知区的真实分类梯度。
    - 按样本归一化：每个样本的 loss 除它自己的已知格数后再做 batch 平均，
      避免早期小视野样本（已知格数很少）在 batch 平均时被稀释掉。
    """
    if targets.dtype != torch.long:
        targets = targets.long()

    known = inputs.squeeze(1) != -1.0  # (B, H, W)
    if not bool(known.any()):
        return logits.sum() * 0.0

    loss_map = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
    masked = loss_map * known
    per_sample_known = known.sum(dim=(1, 2)).clamp(min=1).to(loss_map.dtype)
    per_sample_loss = masked.sum(dim=(1, 2)) / per_sample_known
    return per_sample_loss.mean()


def hybrid_data_collection_worker(
    worker_id, data_queue, max_episodes, map_size, generator_name
):
    np.random.seed(np.random.randint(0, 100000) + worker_id)

    env = GridWorldEnv(
        map_size=map_size,
        sensor_range=SENSOR_RANGE,
        map_generator=generator_name,
    )
    planner = LightweightBFSPlanner()

    maps_collected = 0
    episodes_finished = 0

    for _ in range(max_episodes):
        explored, pos = env.reset()
        perfect_labels = extract_perfect_labels(
            env, generator_name=generator_name
        )

        done = False
        step_count = 0

        while not done:
            target = planner.find_nearest_frontier(explored, pos)
            if target is None:
                break

            explored, pos, _, _, done, _info = env.step_macro(target)
            step_count += 1

            if step_count % 3 == 0:
                try:
                    data_queue.put(
                        (
                            explored.copy(),
                            perfect_labels.copy(),
                            generator_name,
                        ),
                        timeout=10,
                    )
                    maps_collected += 1
                except queue.Full:
                    pass

        episodes_finished += 1
        try:
            data_queue.put(
                {
                    "type": "progress",
                    "worker_id": worker_id,
                    "episodes_finished": episodes_finished,
                    "worker_finished": False,
                },
                timeout=10,
            )
        except queue.Full:
            pass

    try:
        data_queue.put(
            {
                "type": "progress",
                "worker_id": worker_id,
                "episodes_finished": episodes_finished,
                "worker_finished": True,
            },
            timeout=10,
        )
    except queue.Full:
        pass

    data_queue.put(None)


def _build_validation_sample(map_size, generator_name, seed):
    env = GridWorldEnv(
        map_size=map_size,
        sensor_range=SENSOR_RANGE,
        seed=seed,
        map_generator=generator_name,
    )
    explored, _ = env.reset()
    labels = extract_perfect_labels(env, generator_name=generator_name)
    input_tensor = torch.from_numpy(explored.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(labels).unsqueeze(0).long()
    return input_tensor, target_tensor


def _evaluate_fixed_validation(model, validation_sample, device):
    val_input, val_target = validation_sample
    val_input = val_input.to(device)
    val_target = val_target.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(val_input)
        val_loss = _masked_ce_loss(logits, val_target, val_input).item()
        pred = torch.argmax(logits, dim=1)

        # 与训练一致：准确率只在已知区域（输入 != -1）上统计
        known = val_input.squeeze(1) != -1.0  # (1, H, W)
        known_mask = known[0]  # (H, W)
        if bool(known_mask.any()):
            pixel_acc = (pred[0, known_mask] == val_target[0, known_mask]).float().mean().item()
        else:
            pixel_acc = float("nan")

        pred_cpu = pred[0].cpu().numpy()
        target_cpu = val_target[0].cpu().numpy()
        known_cpu = known_mask.cpu().numpy()
        class_acc = []
        for cls in range(3):
            mask = (target_cpu == cls) & known_cpu
            class_acc.append(
                float(np.mean(pred_cpu[mask] == cls))
                if np.any(mask)
                else float("nan")
            )

    model.train()
    return val_loss, pixel_acc, class_acc


def train_main():
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 启动多核并行在线监督混合学习引擎 | 设备: {device} ===")
    print(
        "Loss: masked-known-area CE（无类权重） | 按样本已知格数归一化 | "
        "每完成一张地图打印一次进度"
    )

    num_workers = 12
    map_size = DEFAULT_MAP_SIZE
    episodes_per_worker = 150
    batch_size = 32
    buffer_capacity = 3000

    model = SemanticUNet(in_channels=1, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    memory = SimpleReplayBuffer(capacity=buffer_capacity)

    validation_samples = {
        "Easy": _build_validation_sample(
            map_size, "easy_big_room", TEST_SEED
        ),
        "Fish": _build_validation_sample(
            map_size, "fish_bone", TEST_SEED + 1
        ),
    }

    run_paths = create_run_dirs()
    data_queue = mp.Queue(maxsize=500)
    processes = []

    for worker_id in range(num_workers):
        generator_name = (
            "easy_big_room" if worker_id % 2 == 0 else "fish_bone"
        )
        process = mp.Process(
            target=hybrid_data_collection_worker,
            args=(
                worker_id,
                data_queue,
                episodes_per_worker,
                map_size,
                generator_name,
            ),
        )
        process.start()
        processes.append(process)

    print(f"已启动 {num_workers} 个混合数据生成子进程。等待数据注入...\n")

    active_workers = num_workers
    update_count = 0
    worker_episode_counts = {worker_id: 0 for worker_id in range(num_workers)}
    map_type_counts = {"easy_big_room": 0, "fish_bone": 0}

    # 用于累积两次地图完成之间的训练损失与监控统计（均为 masked 已知区口径）
    last_episodes_finished = 0
    loss_sum = 0.0
    known_frac_sum = 0.0
    class_frac_sum = np.zeros(3)
    updates_since_last_map = 0

    try:
        while active_workers > 0:
            map_completed_event = False

            # 清空当前队列中的所有可用数据
            while True:
                try:
                    item = data_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    active_workers -= 1
                    continue

                if isinstance(item, dict) and item.get("type") == "progress":
                    worker_id = int(item.get("worker_id", -1))
                    if worker_id >= 0:
                        worker_episode_counts[worker_id] = int(
                            item.get(
                                "episodes_finished",
                                worker_episode_counts.get(worker_id, 0),
                            )
                        )
                    
                    # 检查是否有新的地图探索完成
                    current_episodes_finished = sum(worker_episode_counts.values())
                    if current_episodes_finished > last_episodes_finished:
                        map_completed_event = True
                    continue

                if len(item) == 3:
                    explored_map, labels, map_type = item
                else:
                    explored_map, labels = item
                    map_type = "unknown"
                memory.push((explored_map, labels, map_type))
                map_type_counts[map_type] = map_type_counts.get(map_type, 0) + 1

            # 只要池子里数据够，就执行网络更新
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                inputs = np.asarray([sample[0] for sample in batch], dtype=np.float32)
                targets = np.asarray([sample[1] for sample in batch], dtype=np.int64)

                inputs_tensor = torch.from_numpy(inputs).unsqueeze(1).to(device)
                targets_tensor = torch.from_numpy(targets).to(device)

                optimizer.zero_grad()
                outputs = model(inputs_tensor)
                loss = _masked_ce_loss(outputs, targets_tensor, inputs_tensor)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                update_count += 1
                updates_since_last_map += 1

                # 训练监控统计（复用同一次前向，仅用于打印，不参与梯度）
                with torch.no_grad():
                    known = inputs_tensor.squeeze(1) != -1.0
                    known_frac = known.float().mean().item()
                    k_targets = targets_tensor[known]
                    if k_targets.numel() > 0:
                        counts = torch.bincount(k_targets, minlength=3).float()
                        class_frac = counts / counts.sum()
                    else:
                        class_frac = torch.zeros(3)
                    known_frac_sum += known_frac
                    class_frac_sum += class_frac.cpu().numpy()

            # 触发打印逻辑：仅当有地图完成时执行（保持一张地图打印一次）
            if map_completed_event:
                # 计算验证集表现
                val_str_list = []
                for map_type, validation_sample in validation_samples.items():
                    val_loss, val_pixel_acc, class_acc = _evaluate_fixed_validation(
                        model, validation_sample, device
                    )
                    class_acc_text = ", ".join(
                        f"c{idx}={acc:.3f}" if not np.isnan(acc) else f"c{idx}=NaN"
                        for idx, acc in enumerate(class_acc)
                    )
                    val_str_list.append(
                        f"{map_type}[Loss:{val_loss:.3f}, Acc:{val_pixel_acc:.3f}, {class_acc_text}]"
                    )
                val_str = " | ".join(val_str_list)

                # 统计收集到的样本类型分布
                map_type_text = ", ".join(f"{k}={v}" for k, v in map_type_counts.items())

                n_updates = updates_since_last_map
                if n_updates > 0:
                    avg_loss = loss_sum / n_updates
                    avg_known = known_frac_sum / n_updates
                    avg_class = class_frac_sum / n_updates
                    cls_text = ", ".join(
                        f"{idx}:{avg_class[idx] * 100:.1f}%" for idx in range(3)
                    )
                    train_line = (
                        f"Loss: {avg_loss:.4f} | known={avg_known * 100:.1f}% | "
                        f"cls(known): [{cls_text}]"
                    )
                else:
                    train_line = "no train updates since last map"

                # 使用原生 print，避免任何附带的进度条输出
                print(
                    f"\n[Map Completed] Total: {last_episodes_finished}/{num_workers * episodes_per_worker} | "
                    f"Updates: {update_count}\n"
                    f"  ├─ Train: {train_line}\n"
                    f"  ├─ Val: {val_str}\n"
                    f"  └─ Buffer: {len(memory)} | Samples: [{map_type_text}]"
                )

                # 重置累积变量，等待下一张地图完成
                loss_sum = 0.0
                known_frac_sum = 0.0
                class_frac_sum = np.zeros(3)
                updates_since_last_map = 0
                last_episodes_finished = sum(worker_episode_counts.values())

    except KeyboardInterrupt:
        print("\n检测到用户中断，正在关闭进程...")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()

        class StandaloneSaver:
            def save_all_weights(self, checkpoint_path, fusion_path):
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), fusion_path)

        save_model_artifacts(StandaloneSaver(), run_paths, is_best=True)
        save_path = run_paths.checkpoints_dir / "latest_checkpoint.pth"
        print(f"--> 混合训练结束，总更新次数: {update_count}。权重已保存至: {save_path}")


if __name__ == "__main__":
    train_main()