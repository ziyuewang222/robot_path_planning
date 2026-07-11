import json
from pathlib import Path

import numpy as np
import torch

from bfs import BFSFrontierPlanner, max_bfs_compare_rounds
from grid_env import GridWorldEnv
from models import SpatialActionDecision

TRAIN_MAP_REPEAT_EPISODES = 3
REFERENCE_EVAL_INTERVAL = 20
REFERENCE_SEED_START = 50
REFERENCE_SEED_COUNT = 30
TRAIN_SEED_START = 10000


def train_seed_for_episode(episode: int, seed_start: int = TRAIN_SEED_START, repeat: int = TRAIN_MAP_REPEAT_EPISODES) -> int:
    return seed_start + (episode - 1) // repeat


def reference_seeds(seed_start: int = REFERENCE_SEED_START, count: int = REFERENCE_SEED_COUNT):
    return [seed_start + idx for idx in range(count)]


def _step_one_policy(env, target_coord, total_steps: int, total_area: float):
    next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(target_coord)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    total_steps += n_steps
    total_area += info["delta_area"]
    return total_steps, total_area, done, info


def run_bfs_steps(map_size: int, sensor_range: int, map_generator: str, seed: int):
    env = GridWorldEnv(
        map_size=map_size,
        sensor_range=sensor_range,
        seed=seed,
        map_generator=map_generator,
    )
    planner = BFSFrontierPlanner()
    explored, pos = env.reset()
    total_steps = 0
    total_area = 0.0
    done = False

    for _round_idx in range(1, max_bfs_compare_rounds(map_size) + 1):
        target = planner.find_nearest_frontier(explored, pos)
        if target is None:
            break
        explored, pos, _reward, n_steps, done, info = env.step_macro(target)
        total_steps += n_steps
        total_area += info["delta_area"]
        if done:
            break

    return {
        "seed": seed,
        "steps": int(total_steps),
        "area": float(total_area),
        "done": bool(done),
    }


def build_bfs_reference_baselines(map_size: int, sensor_range: int, map_generator: str, seeds):
    baselines = {}
    print("=== 正在计算 30 张参考图的 BFS baseline ===")
    for idx, seed in enumerate(seeds, start=1):
        result = run_bfs_steps(map_size, sensor_range, map_generator, seed)
        baselines[seed] = result
        status = "done" if result["done"] else "not_done"
        print(f"[BFS baseline {idx:02d}/{len(seeds):02d}] seed={seed} | steps={result['steps']:4d} | {status}")
    valid_steps = [item["steps"] for item in baselines.values() if item["done"]]
    if valid_steps:
        print(f"BFS baseline 有效参考图: {len(valid_steps)}/{len(seeds)} | 平均 steps: {np.mean(valid_steps):.2f}")
    else:
        print("BFS baseline 没有有效完成样本，后续参考评估将无法计算平均提升。")
    return baselines


def evaluate_policy_against_bfs(
    policy_net,
    pipeline,
    device,
    map_size: int,
    sensor_range: int,
    map_generator: str,
    bfs_baselines,
    max_rounds: int,
):
    was_training = policy_net.training
    policy_net.eval()
    decision = SpatialActionDecision(policy_net, epsilon=0.0, device=device)
    results = []

    with torch.no_grad():
        for seed, baseline in bfs_baselines.items():
            env = GridWorldEnv(
                map_size=map_size,
                sensor_range=sensor_range,
                seed=seed,
                map_generator=map_generator,
            )
            explored, pos = env.reset()
            total_steps = 0
            total_area = 0.0
            current_target = None
            done = False
            alive = True
            rounds = 0

            for round_idx in range(1, max_rounds + 1):
                if current_target is None:
                    state_tensor = pipeline.process(explored, pos)
                    current_target, _q_masked, _q_raw = decision.select_action(state_tensor, explore=False)
                    if current_target is None:
                        alive = False
                        break

                total_steps, total_area, done, info = _step_one_policy(env, current_target, total_steps, total_area)
                explored = env.explored_map.copy()
                pos = env.robot_pos.copy()
                rounds = round_idx
                reached = pos[0] == current_target[0] and pos[1] == current_target[1]
                if reached or (info["blocked"] and not reached):
                    current_target = None
                if done:
                    break

            if baseline["done"] and done and baseline["steps"] > 0:
                improvement = (baseline["steps"] - total_steps) / baseline["steps"] * 100.0
            else:
                improvement = None
            results.append(
                {
                    "seed": seed,
                    "bfs_steps": baseline["steps"],
                    "bfs_done": baseline["done"],
                    "rl_steps": int(total_steps),
                    "rl_area": float(total_area),
                    "rl_done": bool(done),
                    "rl_alive": bool(alive),
                    "rounds": int(rounds),
                    "improvement_pct": improvement,
                }
            )

    if was_training:
        policy_net.train()

    valid = [item for item in results if item["improvement_pct"] is not None]
    avg_improvement = float(np.mean([item["improvement_pct"] for item in valid])) if valid else 0.0
    rl_better = sum(1 for item in valid if item["rl_steps"] < item["bfs_steps"])
    bfs_better = sum(1 for item in valid if item["rl_steps"] > item["bfs_steps"])
    tied = sum(1 for item in valid if item["rl_steps"] == item["bfs_steps"])
    return {
        "avg_improvement_pct": avg_improvement,
        "valid_count": len(valid),
        "total_count": len(results),
        "rl_better_count": rl_better,
        "bfs_better_count": bfs_better,
        "tied_count": tied,
        "results": results,
    }


def print_reference_eval_summary(episode: int, summary):
    print(
        f"[REFERENCE EVAL {episode}ep] "
        f"valid={summary['valid_count']}/{summary['total_count']} | "
        f"RL平均相对BFS优化={summary['avg_improvement_pct']:.2f}% | "
        f"RL优={summary['rl_better_count']} | BFS优={summary['bfs_better_count']} | 相同={summary['tied_count']}"
    )


def append_reference_eval_log(log_path: Path, episode: int, epsilon: float, summary):
    payload = {
        "episode": episode,
        "epsilon": epsilon,
        **summary,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
