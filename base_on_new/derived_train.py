import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from datetime import datetime
from pathlib import Path

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
    DERIVED_INITIAL_CHECKPOINT_PATH,
    DERIVED_INITIAL_FUSION_WEIGHTS_PATH,
    DERIVED_EPSILON_START,
    DERIVED_EPSILON_END,
    DERIVED_EPSILON_CONVERGE_RATIO,
    DERIVED_LEARNING_RATE,
    EPSILON_STABILIZE_RATIO,
    EPSILON_STABILIZE_END,
)
from state_pipeline import StateTensorPipeline
from trainer_core import SpatialQTrainer
from parallel_env import (
    EpisodeStats,
    SubprocGridWorldVecEnv,
    default_num_envs,
    format_episode_line,
    initial_free_square,
)
from train_monitor import (
    PARAM_REPORT_INTERVAL,
    build_probe_states,
    print_update_monitor,
    probe_actions,
    snapshot_params,
)
from train_reference_eval import TRAIN_MAP_REPEAT_EPISODES, train_seed_for_episode


def _path_or_none(path):
    if path is None:
        return None
    path = Path(path)
    return path if str(path) else None


def derived_train_main():
    print("=== 正在启动派生策略训练 (Derived Multi-Scale Spatial FCN RL) ===")

    initial_checkpoint_path = _path_or_none(DERIVED_INITIAL_CHECKPOINT_PATH)
    initial_fusion_weights_path = _path_or_none(DERIVED_INITIAL_FUSION_WEIGHTS_PATH)
    if initial_checkpoint_path is None and initial_fusion_weights_path is None:
        raise ValueError("derived_train.py 需要至少配置一个初始权重路径")

    map_size = TRAIN_MAP_SIZE
    max_episodes = max_episodes_for(map_size)
    max_macro_steps_per_epi = max_macro_steps_for(map_size)
    num_envs = default_num_envs()
    batch_size = 24
    update_every = 100
    update_burst = 4
    soft_update_every = 4
    replay_buffer_size = max(replay_buffer_size_for(map_size), 30000)
    learning_rate = DERIVED_LEARNING_RATE
    epsilon_start = DERIVED_EPSILON_START
    epsilon_end = DERIVED_EPSILON_END
    epsilon_stabilize_end = EPSILON_STABILIZE_END
    epsilon_converge_ratio = DERIVED_EPSILON_CONVERGE_RATIO
    epsilon_stabilize_ratio = EPSILON_STABILIZE_RATIO
    epsilon_converge_episode = max(1, int(round(max_episodes * epsilon_converge_ratio)))
    epsilon_stabilize_episode = max(
        epsilon_converge_episode + 1,
        min(max_episodes, int(round(max_episodes * epsilon_stabilize_ratio))),
    )
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / epsilon_converge_episode)
    stabilize_decay_steps = max(1, max_episodes - epsilon_stabilize_episode + 1)
    epsilon_stabilize_decay = (epsilon_stabilize_end / epsilon_end) ** (1.0 / stabilize_decay_steps)
    experiment_name = f"derived_spatial_fcn_rl_m{map_size}_{MAP_GENERATOR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    print(f"本次实验名: {experiment_name}")
    print(f"输出目录: {CURRENT_RUN_DIR}")
    print(f"目标地图生成器: {MAP_GENERATOR}")
    print(f"初始主策略权重: {initial_checkpoint_path}")
    print(f"初始融合权重: {initial_fusion_weights_path}")
    print(f"派生训练学习率: {learning_rate:.1e}")
    print(
        f"训练地图尺寸: {map_size} | 最大 Episode 数: {max_episodes} | 并行环境: {num_envs} | "
        f"batch: {batch_size} | update_every: {update_every} | update_burst: {update_burst} | replay buffer: {replay_buffer_size} | "
        f"epsilon 将在前 {epsilon_converge_episode} 个 Episode 从 {epsilon_start:.2f} 下降到 {epsilon_end:.2f}，"
        f"第 {epsilon_stabilize_episode} 个 Episode 后继续下降到 {epsilon_stabilize_end:.2f}"
    )
    print(f"训练地图复用: 每张训练图连续重复 {TRAIN_MAP_REPEAT_EPISODES} 个 Episode")
    print(
        f"[Parallel-DQN] 即将启动 {num_envs} 个 GridWorld 子进程；"
        f"主进程负责 batch action selection / replay buffer / Q update；"
        f"worker 只负责 reset 和 step_macro。"
    )

    pipeline = StateTensorPipeline(map_size=map_size, device=device)
    trainer = SpatialQTrainer(lr=learning_rate, gamma=0.98, buffer_size=replay_buffer_size, device=device)
    initial_policy_mode = trainer.load_initial_policy(
        checkpoint_path=initial_checkpoint_path,
        fusion_weight_path=initial_fusion_weights_path,
    )
    print(f"初始策略加载模式: {initial_policy_mode}")
    probe_states = build_probe_states(map_size=map_size, device=device)
    param_snapshot = snapshot_params(trainer.policy_net)
    previous_probe_actions = probe_actions(trainer.policy_net, probe_states, device)

    run_meta = {
        "experiment_name": experiment_name,
        "train_mode": "derived",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "map_size": map_size,
        "max_episodes": max_episodes,
        "max_macro_steps_per_epi": max_macro_steps_per_epi,
        "num_envs": num_envs,
        "batch_size": batch_size,
        "update_every": update_every,
        "soft_update_every": soft_update_every,
        "update_burst": update_burst,
        "replay_buffer_size": replay_buffer_size,
        "learning_rate": learning_rate,
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
        "initial_policy_mode": initial_policy_mode,
        "initial_checkpoint_path": str(initial_checkpoint_path) if initial_checkpoint_path else None,
        "initial_fusion_weights_path": str(initial_fusion_weights_path) if initial_fusion_weights_path else None,
    }
    with open(CONFIG_SNAPSHOTS_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    trainer.save_all_weights(
        str(CHECKPOINTS_DIR / "startup_checkpoint.pth"),
        str(FUSION_WEIGHTS_DIR / "startup_fusion_weights.pth"),
    )

    epsilon = epsilon_start
    best_reward = float("-inf")
    best_run_reward = float("-inf")
    completed_episodes = 0
    global_macro_step = 0
    update_step_count = 0
    next_episode = 1
    final_training_end_reason = f"已完成预设最大 Episode 数量: {max_episodes}"

    active_states = [None] * num_envs
    active_stats = [None] * num_envs
    active_done = [True] * num_envs

    def start_episode(env_idx: int):
        nonlocal next_episode
        if next_episode > max_episodes:
            active_done[env_idx] = True
            active_states[env_idx] = None
            active_stats[env_idx] = None
            return
        episode = next_episode
        next_episode += 1
        train_seed = train_seed_for_episode(episode)
        explored, pos = vec_env.reset_at(env_idx, train_seed)
        active_states[env_idx] = pipeline.process(explored, pos)
        active_stats[env_idx] = EpisodeStats(
            episode=episode,
            train_seed=train_seed,
            repeat_idx=(episode - 1) % TRAIN_MAP_REPEAT_EPISODES + 1,
            free_square=initial_free_square(explored),
            extra={"train_mode": "derived", "initial_policy_mode": initial_policy_mode},
        )
        active_done[env_idx] = False

    def finish_episode(stats: EpisodeStats, env_idx: int):
        nonlocal epsilon, best_reward, best_run_reward, completed_episodes, param_snapshot, previous_probe_actions
        if stats.episode >= epsilon_stabilize_episode:
            epsilon = max(epsilon_stabilize_end, epsilon * epsilon_stabilize_decay)
        else:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        active_env_count = sum(
            1 for idx in range(num_envs) if not active_done[idx] and active_states[idx] is not None
        )
        print(
            format_episode_line(
                stats,
                max_episodes,
                epsilon,
                TRAIN_MAP_REPEAT_EPISODES,
                prefix="DerivedEpisode",
                env_idx=env_idx,
                num_envs=num_envs,
                active_envs=active_env_count,
                global_macro_step=global_macro_step,
                update_step_count=update_step_count,
                replay_size=len(trainer.memory),
            )
        )
        log_line = stats.to_log_line(
            epsilon,
            TRAIN_MAP_REPEAT_EPISODES,
            env_idx=env_idx,
            num_envs=num_envs,
            active_envs=active_env_count,
            global_macro_step=global_macro_step,
            update_step_count=update_step_count,
            replay_size=len(trainer.memory),
        )
        with open(LOGS_DIR / "train_progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        if stats.episode % PARAM_REPORT_INTERVAL == 0:
            param_snapshot, previous_probe_actions = print_update_monitor(
                stats.episode,
                trainer.policy_net,
                param_snapshot,
                previous_probe_actions,
                probe_states,
                device,
            )

        completed_episodes = max(completed_episodes, stats.episode)
        trainer.save_all_weights(str(LATEST_CHECKPOINT_PATH), str(LATEST_FUSION_WEIGHTS_PATH))
        trainer.save_all_weights(
            str(CHECKPOINTS_DIR / "latest_checkpoint.pth"),
            str(FUSION_WEIGHTS_DIR / "latest_fusion_weights.pth"),
        )

        if stats.reward > best_reward:
            best_reward = stats.reward
            trainer.save_all_weights(str(BEST_CHECKPOINT_PATH), str(BEST_FUSION_WEIGHTS_PATH))
            trainer.save_all_weights(
                str(CHECKPOINTS_DIR / "best_checkpoint.pth"),
                str(FUSION_WEIGHTS_DIR / "best_fusion_weights.pth"),
            )
            print(f"--> 当前派生训练最优模型已更新，reward={best_reward:.2f}")

        best_run_reward = max(best_run_reward, stats.reward)

    with SubprocGridWorldVecEnv(num_envs, map_size, SENSOR_RANGE, MAP_GENERATOR) as vec_env:
        worker_pids = [process.pid for process in vec_env.processes]
        print(f"[Parallel-DQN] 子进程已启动: num_envs={num_envs}, worker_pids={worker_pids}")
        for env_idx in range(num_envs):
            start_episode(env_idx)

        while completed_episodes < max_episodes:
            active_indices = [idx for idx in range(num_envs) if not active_done[idx] and active_states[idx] is not None]
            if not active_indices:
                break

            states_for_action = [active_states[idx] for idx in active_indices]
            selected_actions = trainer.select_actions_batch(states_for_action, epsilon=epsilon)
            actions = [None] * num_envs
            for env_idx, action in zip(active_indices, selected_actions):
                actions[env_idx] = action
                if action is None:
                    stats = active_stats[env_idx]
                    stats.end_reason = "没有可选择的有效探索目标"
                    finish_episode(stats, env_idx)
                    start_episode(env_idx)

            step_results = vec_env.step(actions)
            loss_sum = 0.0
            loss_count = 0
            for env_idx, (next_explored, next_pos, reward, tau_steps, done, info) in step_results.items():
                stats = active_stats[env_idx]
                state = active_states[env_idx]
                stats.macro_steps += 1
                stats.steps += tau_steps
                stats.reward += reward
                stats.delta_area_sum += info.get("delta_area", 0.0)
                stats.free_square = int((next_explored == 0).sum())
                if info.get("collision"):
                    stats.collision_count += 1
                if info.get("blocked"):
                    stats.blocked_count += 1

                next_state = pipeline.process(next_explored, next_pos)
                next_mask = next_state[StateTensorPipeline.FRONTIER_CHANNEL]
                trainer.memory.push(state.cpu(), actions[env_idx], reward, next_state.cpu(), next_mask.cpu(), done, tau_steps)
                active_states[env_idx] = next_state
                global_macro_step += 1

                if len(trainer.memory) >= batch_size and global_macro_step % update_every < num_envs:
                    for _ in range(update_burst):
                        loss = trainer.update_step(batch_size=batch_size)
                        if loss <= 0:
                            break
                        loss_sum += loss
                        loss_count += 1
                        update_step_count += 1
                        if update_step_count % soft_update_every == 0:
                            trainer.soft_update_target(tau=0.02)

                if done:
                    stats.end_reason = "已探索至少 98% 的可通行自由空间"
                elif stats.macro_steps >= max_macro_steps_per_epi:
                    stats.end_reason = "达到本回合最大宏步数限制"
                    done = True
                elif info.get("collision"):
                    stats.end_reason = "宏动作执行中发生碰撞"
                elif info.get("blocked"):
                    stats.end_reason = "目标路径不可达或目标被确认为障碍物"

                if done:
                    stats.add_loss(loss_sum, loss_count)
                    finish_episode(stats, env_idx)
                    start_episode(env_idx)

            if loss_count > 0:
                for stats in active_stats:
                    if stats is not None:
                        stats.add_loss(loss_sum, loss_count)

    print(
        f"--> 派生训练结束原因: {final_training_end_reason}；"
        f"实际完成 Episode 数: {completed_episodes}/{max_episodes}；"
        f"本次派生训练最佳 reward: {best_run_reward:.2f}"
    )
    trainer.save_all_weights(str(LATEST_CHECKPOINT_PATH), str(LATEST_FUSION_WEIGHTS_PATH))
    trainer.save_all_weights(
        str(CHECKPOINTS_DIR / "latest_checkpoint.pth"),
        str(FUSION_WEIGHTS_DIR / "latest_fusion_weights.pth"),
    )
    print(f"--> 派生训练结束，latest 主干权重已保存到: {LATEST_CHECKPOINT_PATH}")
    print(f"--> 派生训练结束，latest 参考融合权重已保存到: {LATEST_FUSION_WEIGHTS_PATH}")
    print(f"--> run 目录 latest 主干权重: {CHECKPOINTS_DIR / 'latest_checkpoint.pth'}")
    print(f"--> run 目录 latest 参考融合权重: {FUSION_WEIGHTS_DIR / 'latest_fusion_weights.pth'}")


if __name__ == "__main__":
    derived_train_main()
