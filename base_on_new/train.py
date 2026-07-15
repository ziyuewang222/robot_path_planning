import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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


INITIAL_FUSION_WEIGHTS = (0.3, 0.2, 0.5)


def train_main():
    print("=== 正在启动多尺度空间强化学习 (Multi-Scale Spatial FCN RL) 仿真训练引擎 ===")

    map_size = TRAIN_MAP_SIZE
    max_episodes = max_episodes_for(map_size)
    max_macro_steps_per_epi = max_macro_steps_for(map_size)
    num_envs = default_num_envs()
    batch_size = 24
    update_every = 100
    update_burst = 4
    soft_update_every = 4
    replay_buffer_size = replay_buffer_size_for(map_size)
    replay_state_dtype = torch.float16
    replay_mask_dtype = torch.bool
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
        "num_envs": num_envs,
        "batch_size": batch_size,
        "update_every": update_every,
        "soft_update_every": soft_update_every,
        "update_burst": update_burst,
        "replay_buffer_size": replay_buffer_size,
        "replay_state_dtype": str(replay_state_dtype),
        "replay_mask_dtype": str(replay_mask_dtype),
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
        f"训练地图尺寸: {map_size} | 最大 Episode 数: {max_episodes} | 并行环境: {num_envs} | "
        f"batch: {batch_size} | update_every: {update_every} | update_burst: {update_burst} | replay buffer: {replay_buffer_size} | "
        f"gamma: 0.98 (macro-step MDP) | "
        f"epsilon 将在前 {epsilon_converge_episode} 个 Episode 下降到 {epsilon_end:.2f}，"
        f"第 {epsilon_stabilize_episode} 个 Episode 后继续下降到 {epsilon_stabilize_end:.2f}"
    )
    print(f"训练地图复用: 每张训练图连续重复 {TRAIN_MAP_REPEAT_EPISODES} 个 Episode")
    print(
        f"[Parallel-DQN] 即将启动 {num_envs} 个 GridWorld 子进程；"
        f"主进程负责 batch action selection / replay buffer / Q update；"
        f"worker 只负责 reset 和 step_macro。"
    )

    _pipeline = StateTensorPipeline(map_size=map_size, device=device)
    trainer = SpatialQTrainer(lr=3e-4, gamma=0.98, buffer_size=replay_buffer_size, device=device)
    trainer.policy_net.set_fusion_weights(INITIAL_FUSION_WEIGHTS)
    trainer.sync_target_net()
    probe_states = build_probe_states(map_size=map_size, device=device)
    param_snapshot = snapshot_params(trainer.policy_net)
    previous_probe_actions = probe_actions(trainer.policy_net, probe_states, device)

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
    total_update_loss = 0.0
    total_update_count = 0
    episode_update_loss = 0.0
    episode_update_count = 0
    last_logged_update_count = 0
    last_logged_update_loss = 0.0
    next_episode = 1
    final_training_end_reason = "训练尚未结束"

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
        reset_payload = vec_env.reset_at(env_idx, train_seed)
        state_numpy, explored, _pos = reset_payload
        active_states[env_idx] = torch.from_numpy(state_numpy).float().to(device)
        active_stats[env_idx] = EpisodeStats(
            episode=episode,
            train_seed=train_seed,
            repeat_idx=(episode - 1) % TRAIN_MAP_REPEAT_EPISODES + 1,
            free_square=initial_free_square(explored),
        )
        active_done[env_idx] = False

    def finish_episode(stats: EpisodeStats, env_idx: int):
        nonlocal epsilon, best_reward, best_run_reward, completed_episodes, param_snapshot, previous_probe_actions
        nonlocal episode_update_loss, episode_update_count, last_logged_update_count, last_logged_update_loss
        if stats.episode >= epsilon_stabilize_episode:
            epsilon = max(epsilon_stabilize_end, epsilon * epsilon_stabilize_decay)
        else:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        active_env_count = sum(1 for idx in range(num_envs) if not active_done[idx] and active_states[idx] is not None)
        logged_update_count = update_step_count - last_logged_update_count
        logged_update_loss = episode_update_loss - last_logged_update_loss
        stats.loss_sum = logged_update_loss
        stats.update_cnt = logged_update_count
        print(
            format_episode_line(
                stats,
                max_episodes,
                epsilon,
                TRAIN_MAP_REPEAT_EPISODES,
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
        last_logged_update_count = update_step_count
        last_logged_update_loss = episode_update_loss
        episode_update_loss = 0.0
        episode_update_count = 0
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
        trainer.save_all_weights(str(CHECKPOINTS_DIR / "latest_checkpoint.pth"), str(FUSION_WEIGHTS_DIR / "latest_fusion_weights.pth"))

        if stats.reward > best_reward:
            best_reward = stats.reward
            trainer.save_all_weights(str(BEST_CHECKPOINT_PATH), str(BEST_FUSION_WEIGHTS_PATH))
            trainer.save_all_weights(str(CHECKPOINTS_DIR / "best_checkpoint.pth"), str(FUSION_WEIGHTS_DIR / "best_fusion_weights.pth"))
            print(f"--> 当前最优模型已更新，reward={best_reward:.2f}")

        best_run_reward = max(best_run_reward, stats.reward)

    def resolve_end_reason(done: bool, info: dict, macro_steps: int) -> str:
        if done and info.get("success"):
            return "已探索至少 99.5% 的可通行自由空间"
        if macro_steps >= max_macro_steps_per_epi:
            return "达到本回合最大宏步数限制"
        if info.get("collision"):
            return "宏动作执行中发生碰撞"
        if info.get("blocked"):
            return "目标路径不可达或目标被确认为障碍物"
        if done:
            return str(info.get("end_reason", "环境返回结束信号"))
        return "进行中"

    try:
        with SubprocGridWorldVecEnv(num_envs, map_size, SENSOR_RANGE, MAP_GENERATOR) as vec_env:
            worker_pids = [process.pid for process in vec_env.processes]
            print(f"[Parallel-DQN] 子进程已启动: num_envs={num_envs}, worker_pids={worker_pids}")
            for env_idx in range(num_envs):
                start_episode(env_idx)

            while completed_episodes < max_episodes:
                active_indices = [idx for idx in range(num_envs) if not active_done[idx] and active_states[idx] is not None]
                if not active_indices:
                    final_training_end_reason = "所有环境都已结束"
                    break

                states_for_action = [active_states[idx] for idx in active_indices]
                selected_actions = trainer.select_actions_batch(states_for_action, epsilon=epsilon)
                actions = [None] * num_envs
                terminated_now = []
                for env_idx, action in zip(active_indices, selected_actions):
                    if action is None:
                        stats = active_stats[env_idx]
                        stats.end_reason = "没有可选择的有效探索目标"
                        terminated_now.append(env_idx)
                        finish_episode(stats, env_idx)
                        start_episode(env_idx)
                        continue
                    actions[env_idx] = action

                step_results = vec_env.step(actions)
                for env_idx, (next_state_numpy, next_explored, _next_pos, reward, tau_steps, done, info) in step_results.items():
                    stats = active_stats[env_idx]
                    state = active_states[env_idx]

                    stats.macro_steps += 1
                    stats.steps += int(tau_steps)
                    stats.reward += reward
                    stats.delta_area_sum += info.get("delta_area", 0.0)
                    stats.free_square = int((next_explored == 0).sum())
                    if info.get("collision"):
                        stats.collision_count += 1
                    if info.get("blocked"):
                        stats.blocked_count += 1

                    next_state = torch.from_numpy(next_state_numpy).float().to(device)
                    next_mask = next_state[StateTensorPipeline.FRONTIER_CHANNEL]
                    trainer.memory.push(
                        state.detach().to(dtype=replay_state_dtype, device="cpu"),
                        actions[env_idx],
                        reward,
                        next_state.detach().to(dtype=replay_state_dtype, device="cpu"),
                        next_mask.detach().to(dtype=replay_mask_dtype, device="cpu"),
                        done,
                        tau_steps,
                    )
                    active_states[env_idx] = next_state
                    global_macro_step += 1

                    stats.end_reason = resolve_end_reason(done, info, stats.macro_steps)

                    if done or stats.macro_steps >= max_macro_steps_per_epi:
                        finish_episode(stats, env_idx)
                        start_episode(env_idx)

                if len(trainer.memory) >= batch_size and global_macro_step % update_every < num_envs:
                    for _ in range(update_burst):
                        loss = trainer.update_step(batch_size=batch_size)
                        if loss <= 0:
                            break
                        total_update_loss += loss
                        total_update_count += 1
                        update_step_count += 1
                        episode_update_loss += loss
                        episode_update_count += 1
                        if update_step_count % soft_update_every == 0:
                            trainer.soft_update_target(tau=0.02)

            if completed_episodes >= max_episodes:
                final_training_end_reason = f"已完成预设最大 Episode 数量: {max_episodes}"
            elif final_training_end_reason == "训练尚未结束":
                final_training_end_reason = "所有环境都已结束"
    except KeyboardInterrupt:
        final_training_end_reason = "训练被用户中断"
        raise
    except Exception as exc:
        final_training_end_reason = f"训练异常中断: {type(exc).__name__}: {exc}"
        raise
    finally:
        print(
            f"--> 训练结束原因: {final_training_end_reason}；"
            f"实际完成 Episode 数: {completed_episodes}/{max_episodes}；"
            f"本次训练最佳 reward: {best_run_reward:.2f}；"
            f"全局 update: {total_update_count} 次，累计 loss: {total_update_loss:.4f}"
        )
        trainer.save_all_weights(str(LATEST_CHECKPOINT_PATH), str(LATEST_FUSION_WEIGHTS_PATH))
        trainer.save_all_weights(str(CHECKPOINTS_DIR / "latest_checkpoint.pth"), str(FUSION_WEIGHTS_DIR / "latest_fusion_weights.pth"))
        print(f"--> 训练结束，latest 主干权重已保存到: {LATEST_CHECKPOINT_PATH}")
        print(f"--> 训练结束，latest 参考融合权重已保存到: {LATEST_FUSION_WEIGHTS_PATH}")
        print(f"--> run 目录 latest 主干权重: {CHECKPOINTS_DIR / 'latest_checkpoint.pth'}")
        print(f"--> run 目录 latest 参考融合权重: {FUSION_WEIGHTS_DIR / 'latest_fusion_weights.pth'}")


if __name__ == "__main__":
    train_main()
