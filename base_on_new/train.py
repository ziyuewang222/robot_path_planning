import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from config import (
    TRAIN_MAP_SIZE,
    max_macro_steps_for,
    SENSOR_RANGE,
    SIGMA,
    MAP_GENERATOR,
    CHECKPOINTS_DIR,
    CHECKPOINTS_ROOT,
    FUSION_WEIGHTS_DIR,
    FUSION_WEIGHTS_ROOT,
    LOGS_DIR,
    METRICS_DIR,
    CONFIG_SNAPSHOTS_DIR,
    CURRENT_RUN_DIR,
    LATEST_CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    LATEST_FUSION_WEIGHTS_PATH,
    BEST_FUSION_WEIGHTS_PATH,
)
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline
from models import SpatialFCN, SpatialActionDecision


class SMDPReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_coord, reward, next_state, next_mask, done, tau_steps):
        self.buffer.append((state, action_coord, reward, next_state, next_mask, done, tau_steps))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class SpatialQTrainer:
    def __init__(self, lr=1e-4, gamma=0.99, buffer_size=2000, device='cpu'):
        self.device = device
        self.gamma = gamma

        # 主干网络参数与“参考权重”一起训练；其中参考权重保存在 scale_fusion_logits 中。
        self.policy_net = SpatialFCN().to(device)
        self.target_net = SpatialFCN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = SMDPReplayBuffer(capacity=buffer_size)

    def update_step(self, batch_size=16):
        if len(self.memory) < batch_size:
            return 0.0

        transitions = self.memory.sample(batch_size)
        loss_val = 0.0

        self.optimizer.zero_grad()

        for state, action_coord, reward, next_state, next_mask, done, tau in transitions:
            state = state.unsqueeze(0).to(self.device)
            next_state = next_state.unsqueeze(0).to(self.device)
            next_mask = next_mask.to(self.device)

            q_raw = self.policy_net(state).squeeze(0).squeeze(0)
            pred_q = q_raw[action_coord[0], action_coord[1]]

            with torch.no_grad():
                if done:
                    target_q = torch.tensor(reward, dtype=torch.float32, device=self.device)
                else:
                    q_next_raw = self.target_net(next_state).squeeze(0).squeeze(0)
                    q_next_masked = q_next_raw.clone()
                    q_next_masked[next_mask < 0.5] = -1e9

                    max_next_q = torch.max(q_next_masked)
                    gamma_tau = self.gamma ** tau
                    target_q = reward + gamma_tau * max_next_q

            loss = self.criterion(pred_q, target_q)
            loss.backward()
            loss_val += loss.item()

        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss_val / batch_size

    def soft_update_target(self, tau=0.01):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_all_weights(self, checkpoint_path: str, fusion_weight_path: str) -> None:
        torch.save(self.policy_net.state_dict(), checkpoint_path)
        self.policy_net.save_fusion_weights(fusion_weight_path)


def train_main():
    print("=== 正在启动多尺度空间强化学习 (Multi-Scale Spatial FCN RL) 仿真训练引擎 ===")

    map_size = TRAIN_MAP_SIZE
    max_episodes = 120
    max_macro_steps_per_epi = max_macro_steps_for(map_size)
    batch_size = 16
    epsilon_start = 1.0
    epsilon_end = 0.08
    epsilon_decay = 0.99
    experiment_name = f"spatial_fcn_rl_m{map_size}_{MAP_GENERATOR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的底层计算加速设备: {device}")

    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    FUSION_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    FUSION_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "map_size": map_size,
        "max_episodes": max_episodes,
        "max_macro_steps_per_epi": max_macro_steps_per_epi,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "device": str(device),
        "map_generator": MAP_GENERATOR,
        "model_type": "MultiScaleSpatialFCN",
    }
    with open(CONFIG_SNAPSHOTS_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"本次实验名: {experiment_name}")
    print(f"输出目录: {CURRENT_RUN_DIR}")

    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, map_generator=MAP_GENERATOR)
    pipeline = StateTensorPipeline(map_size=map_size, sigma=SIGMA, device=device)
    trainer = SpatialQTrainer(lr=3e-4, gamma=0.98, device=device)

    # 训练开始先保存一份初始权重，方便你确认 run 目录下能立即看到文件。
    trainer.save_all_weights(
        str(CHECKPOINTS_DIR / "startup_checkpoint.pth"),
        str(FUSION_WEIGHTS_DIR / "startup_fusion_weights.pth"),
    )

    epsilon = epsilon_start
    best_reward = float("-inf")
    best_run_reward = float("-inf")

    for episode in range(1, max_episodes + 1):
        explored, pos = env.reset()
        state = pipeline.process(explored, pos)

        decision = SpatialActionDecision(trainer.policy_net, epsilon=epsilon, device=device)

        total_reward = 0.0
        total_loss = 0.0
        update_cnt = 0

        for step in range(1, max_macro_steps_per_epi + 1):
            action_coord, _, _ = decision.select_action(state, explore=True)
            if action_coord is None:
                break

            next_explored, next_pos, reward, tau_steps, done, _ = env.step_macro(action_coord)
            next_state = pipeline.process(next_explored, next_pos)
            next_mask = next_state[2]

            trainer.memory.push(state.cpu(), action_coord, reward, next_state.cpu(), next_mask.cpu(), done, tau_steps)

            loss = trainer.update_step(batch_size=batch_size)
            trainer.soft_update_target(tau=0.02)

            state = next_state
            total_reward += reward
            if loss > 0:
                total_loss += loss
                update_cnt += 1

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        avg_loss = (total_loss / update_cnt) if update_cnt > 0 else 0.0

        print(f"Episode [{episode:02d}/{max_episodes:02d}] | 累积物理回报: {total_reward:7.2f} | 探索率(eps): {epsilon:.3f} | 平均 MSE 损失: {avg_loss:.4f} | 回合步数: {step:2d}")

        log_line = {
            "episode": episode,
            "reward": total_reward,
            "avg_loss": avg_loss,
            "epsilon": epsilon,
            "steps": step,
        }
        with open(LOGS_DIR / "train_progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        if total_reward > best_reward:
            best_reward = total_reward
            trainer.save_all_weights(str(BEST_CHECKPOINT_PATH), str(BEST_FUSION_WEIGHTS_PATH))
            trainer.save_all_weights(
                str(CHECKPOINTS_DIR / "best_checkpoint.pth"),
                str(FUSION_WEIGHTS_DIR / "best_fusion_weights.pth"),
            )
            print(f"--> 当前最优模型已更新，reward={best_reward:.2f}")

        best_run_reward = max(best_run_reward, total_reward)

    # 训练结束后更新全局 latest，并在 run 目录下保留最终版本。
    trainer.save_all_weights(str(LATEST_CHECKPOINT_PATH), str(LATEST_FUSION_WEIGHTS_PATH))
    trainer.save_all_weights(
        str(CHECKPOINTS_DIR / "final_checkpoint.pth"),
        str(FUSION_WEIGHTS_DIR / "final_fusion_weights.pth"),
    )
    print(f"--> 训练结束，最终主干权重已保存到: {LATEST_CHECKPOINT_PATH}")
    print(f"--> 训练结束，最终参考融合权重已保存到: {LATEST_FUSION_WEIGHTS_PATH}")
    print(f"--> run 目录最终主干权重: {CHECKPOINTS_DIR / 'final_checkpoint.pth'}")
    print(f"--> run 目录最终参考融合权重: {FUSION_WEIGHTS_DIR / 'final_fusion_weights.pth'}")


if __name__ == "__main__":
    train_main()
