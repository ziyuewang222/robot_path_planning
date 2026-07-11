import json
from datetime import datetime

import torch

from config import (
    TRAIN_MAP_SIZE,
    max_episodes_for,
    max_macro_steps_for,
    replay_buffer_size_for,
    SENSOR_RANGE,
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
    AREA_REWARD_SCALE,
    STEP_PENALTY,
    COLLISION_PENALTY,
    EPSILON_CONVERGE_RATIO,
    EPSILON_STABILIZE_RATIO,
    EPSILON_STABILIZE_END,
)
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline
from models import SpatialActionDecision
from trainer_core import SpatialQTrainer
from train_monitor import (
    PARAM_REPORT_INTERVAL,
    build_probe_states,
    print_update_monitor,
    probe_actions,
    snapshot_params,
)
from train_reference_eval import TRAIN_MAP_REPEAT_EPISODES, train_seed_for_episode


def train_main():
    print("=== 正在启动多尺度空间强化学习 (Multi-Scale Spatial FCN RL) 仿真训练引擎 ===")

    map_size = TRAIN_MAP_SIZE
    max_episodes = max_episodes_for(map_size)
    max_macro_steps_per_epi = max_macro_steps_for(map_size)
    batch_size = 16
    update_every = 4
    replay_buffer_size = replay_buffer_size_for(map_size)
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_stabilize_end = EPSILON_STABILIZE_END
    epsilon_converge_ratio = EPSILON_CONVERGE_RATIO
    epsilon_stabilize_ratio = EPSILON_STABILIZE_RATIO
    epsilon_converge_episode = max(1, int(round(max_episodes * epsilon_converge_ratio)))
    epsilon_stabilize_episode = max(
        epsilon_converge_episode + 1,
        min(max_episodes, int(round(max_episodes * epsilon_stabilize_ratio))),
    )
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / epsilon_converge_episode)
    stabilize_decay_steps = max(1, max_episodes - epsilon_stabilize_episode + 1)
    epsilon_stabilize_decay = (epsilon_stabilize_end / epsilon_end) ** (1.0 / stabilize_decay_steps)
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
        "update_every": update_every,
        "replay_buffer_size": replay_buffer_size,
        "area_reward_scale": AREA_REWARD_SCALE,
        "step_penalty": STEP_PENALTY,
        "collision_penalty": COLLISION_PENALTY,
        "success_bonus": 0.0,
        "success_bonus_per_area": 0.0,
        "reward_terminal_bonus_removed": True,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_stabilize_end": epsilon_stabilize_end,
        "epsilon_converge_episode": epsilon_converge_episode,
        "epsilon_converge_ratio": epsilon_converge_ratio,
        "epsilon_decay": epsilon_decay,
        "epsilon_stabilize_episode": epsilon_stabilize_episode,
        "epsilon_stabilize_ratio": epsilon_stabilize_ratio,
        "epsilon_stabilize_decay": epsilon_stabilize_decay,
        "device": str(device),
        "map_generator": MAP_GENERATOR,
        "model_type": "MultiScaleSpatialFCN",
        "state_channel_3": "bfs_reachable_distance_field_vC",
        "train_map_repeat_episodes": TRAIN_MAP_REPEAT_EPISODES,
    }
    with open(CONFIG_SNAPSHOTS_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"本次实验名: {experiment_name}")
    print(f"输出目录: {CURRENT_RUN_DIR}")
    print(
        f"训练地图尺寸: {map_size} | 最大 Episode 数: {max_episodes} | replay buffer: {replay_buffer_size} | "
        f"epsilon 将在前 {epsilon_converge_episode} 个 Episode 下降到 {epsilon_end:.2f}，"
        f"第 {epsilon_stabilize_episode} 个 Episode 后继续下降到 {epsilon_stabilize_end:.2f}"
    )
    print(f"训练地图复用: 每张训练图连续重复 {TRAIN_MAP_REPEAT_EPISODES} 个 Episode")

    pipeline = StateTensorPipeline(map_size=map_size, device=device)
    trainer = SpatialQTrainer(lr=3e-4, gamma=0.98, buffer_size=replay_buffer_size, device=device)
    probe_states = build_probe_states(map_size=map_size, device=device)
    param_snapshot = snapshot_params(trainer.policy_net)
    previous_probe_actions = probe_actions(trainer.policy_net, probe_states, device)

    # 训练开始先保存一份初始权重，方便你确认 run 目录下能立即看到文件。
    trainer.save_all_weights(
        str(CHECKPOINTS_DIR / "startup_checkpoint.pth"),
        str(FUSION_WEIGHTS_DIR / "startup_fusion_weights.pth"),
    )

    epsilon = epsilon_start
    best_reward = float("-inf")
    best_run_reward = float("-inf")
    completed_episodes = 0
    global_macro_step = 0

    final_training_end_reason = f"已完成预设最大 Episode 数量: {max_episodes}"

    for episode in range(1, max_episodes + 1):
        train_seed = train_seed_for_episode(episode)
        env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=train_seed, map_generator=MAP_GENERATOR)
        explored, pos = env.reset()
        state = pipeline.process(explored, pos)

        decision = SpatialActionDecision(trainer.policy_net, epsilon=epsilon, device=device)

        total_reward = 0.0
        total_loss = 0.0
        update_cnt = 0
        episode_end_reason = "达到本回合最大宏步数限制"
        steps = 0
        free_square = 0
        delta_area_sum = 0.0
        collision_count = 0
        blocked_count = 0
        macro_step = 0

        for macro_step in range(1, max_macro_steps_per_epi + 1):
            global_macro_step += 1
            action_coord, _, _ = decision.select_action(state, explore=True, return_maps=False)
            if action_coord is None:
                episode_end_reason = "没有可选择的有效探索目标"
                break

            next_explored, next_pos, reward, tau_steps, done, info = env.step_macro(action_coord)
            steps += tau_steps
            delta_area_sum += info.get("delta_area", 0.0)
            if info.get("collision"):
                collision_count += 1
            if info.get("blocked"):
                blocked_count += 1

            free_square = int((next_explored == 0).sum())

            next_state = pipeline.process(next_explored, next_pos)
            next_mask = next_state[StateTensorPipeline.FRONTIER_CHANNEL]

            trainer.memory.push(state.cpu(), action_coord, reward, next_state.cpu(), next_mask.cpu(), done, tau_steps)

            loss = 0.0
            if global_macro_step % update_every == 0:
                loss = trainer.update_step(batch_size=batch_size)
                if loss > 0:
                    trainer.soft_update_target(tau=0.02)

            state = next_state
            total_reward += reward
            if loss > 0:
                total_loss += loss
                update_cnt += 1

            if done:
                episode_end_reason = "已探索至少 98% 的可通行自由空间"
                break
            if info.get("collision"):
                episode_end_reason = "宏动作执行中发生碰撞"
            elif info.get("blocked"):
                episode_end_reason = "目标路径不可达或目标被确认为障碍物"

        if episode >= epsilon_stabilize_episode:
            epsilon = max(epsilon_stabilize_end, epsilon * epsilon_stabilize_decay)
        else:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        avg_loss = (total_loss / update_cnt) if update_cnt > 0 else 0.0

        print(f"Episode [{episode:02d}/{max_episodes:02d}] | seed: {train_seed} | repeat_idx: {(episode - 1) % TRAIN_MAP_REPEAT_EPISODES + 1}/{TRAIN_MAP_REPEAT_EPISODES} | 累积物理回报: {total_reward:7.2f} | 探索率(eps): {epsilon:.3f} | 平均 SmoothL1 损失: {avg_loss:.4f} | steps: {steps:4d} | free_square: {free_square:4d} | delta_area_sum: {delta_area_sum:7.1f} | collision: {collision_count:2d} | blocked: {blocked_count:2d} | 结束原因: {episode_end_reason}")

        log_line = {
            "episode": episode,
            "train_seed": train_seed,
            "train_seed_repeat_idx": (episode - 1) % TRAIN_MAP_REPEAT_EPISODES + 1,
            "train_seed_repeat_episodes": TRAIN_MAP_REPEAT_EPISODES,
            "reward": total_reward,
            "avg_loss": avg_loss,
            "epsilon": epsilon,
            "steps": steps,
            "free_square": free_square,
            "delta_area_sum": delta_area_sum,
            "collision_count": collision_count,
            "blocked_count": blocked_count,
            "macro_steps": macro_step,
            "end_reason": episode_end_reason,
        }
        with open(LOGS_DIR / "train_progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        if episode % PARAM_REPORT_INTERVAL == 0:
            param_snapshot, previous_probe_actions = print_update_monitor(
                episode,
                trainer.policy_net,
                param_snapshot,
                previous_probe_actions,
                probe_states,
                device,
            )

        completed_episodes = episode

        trainer.save_all_weights(str(LATEST_CHECKPOINT_PATH), str(LATEST_FUSION_WEIGHTS_PATH))
        trainer.save_all_weights(
            str(CHECKPOINTS_DIR / "latest_checkpoint.pth"),
            str(FUSION_WEIGHTS_DIR / "latest_fusion_weights.pth"),
        )

        if total_reward > best_reward:
            best_reward = total_reward
            trainer.save_all_weights(str(BEST_CHECKPOINT_PATH), str(BEST_FUSION_WEIGHTS_PATH))
            trainer.save_all_weights(
                str(CHECKPOINTS_DIR / "best_checkpoint.pth"),
                str(FUSION_WEIGHTS_DIR / "best_fusion_weights.pth"),
            )
            print(f"--> 当前最优模型已更新，reward={best_reward:.2f}")

        best_run_reward = max(best_run_reward, total_reward)

    # 训练结束后，run 目录内会保留 startup / best / latest 三类权重。
    print(
        f"--> 训练结束原因: {final_training_end_reason}；"
        f"实际完成 Episode 数: {completed_episodes}/{max_episodes}；"
        f"本次训练最佳 reward: {best_run_reward:.2f}"
    )
    trainer.save_all_weights(str(LATEST_CHECKPOINT_PATH), str(LATEST_FUSION_WEIGHTS_PATH))
    trainer.save_all_weights(
        str(CHECKPOINTS_DIR / "latest_checkpoint.pth"),
        str(FUSION_WEIGHTS_DIR / "latest_fusion_weights.pth"),
    )
    print(f"--> 训练结束，latest 主干权重已保存到: {LATEST_CHECKPOINT_PATH}")
    print(f"--> 训练结束，latest 参考融合权重已保存到: {LATEST_FUSION_WEIGHTS_PATH}")
    print(f"--> run 目录 latest 主干权重: {CHECKPOINTS_DIR / 'latest_checkpoint.pth'}")
    print(f"--> run 目录 latest 参考融合权重: {FUSION_WEIGHTS_DIR / 'latest_fusion_weights.pth'}")


if __name__ == "__main__":
    train_main()
