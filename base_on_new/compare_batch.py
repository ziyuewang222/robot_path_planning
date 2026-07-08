"""
批量对比 BFS 与 RL 在多张随机地图上的探索步数。

使用方法：

1. 默认使用 config.py 中 TEST_CHECKPOINT_PATH 和 TEST_FUSION_WEIGHTS_PATH，对比 NUM_COMPARE_SEEDS 张地图：
   python compare_batch.py

2. 使用 best 权重：
   python compare_batch.py --use-best

3. 指定 RL 主干权重和融合权重：
   python compare_batch.py --weights /path/to/checkpoint.pth --fusion-weights /path/to/fusion_weights.pth
"""

import argparse

import numpy as np

from bfs import BFSFrontierPlanner, max_bfs_compare_rounds
from config import (
    BEST_CHECKPOINT_PATH,
    BEST_FUSION_WEIGHTS_PATH,
    TEST_CHECKPOINT_PATH,
    TEST_FUSION_WEIGHTS_PATH,
    MAP_GENERATOR,
    SENSOR_RANGE,
    TEST_MAP_SIZE,
)
from grid_env import GridWorldEnv
from rl_runtime import RLRuntime

# 批量对比的随机地图数量，修改这里即可调整测试规模
NUM_COMPARE_SEEDS = 30

# 随机种子起点。实际使用的种子为：SEED_START, SEED_START + 1, ...
SEED_START = 50


class CompareResult:
    def __init__(self, name, total_steps, total_area, done, rounds):
        self.name = name
        self.total_steps = total_steps
        self.total_area = total_area
        self.done = done
        self.rounds = rounds


def _step_one_policy(env, target_coord, total_steps: int, total_area: float):
    next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(target_coord)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    total_steps += n_steps
    total_area += info["delta_area"]
    return total_steps, total_area, done, info


def _run_one_seed(seed: int, rl_runtime: RLRuntime):
    bfs_env = GridWorldEnv(
        map_size=TEST_MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=seed,
        map_generator=MAP_GENERATOR,
    )
    rl_env = GridWorldEnv(
        map_size=TEST_MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=seed,
        map_generator=MAP_GENERATOR,
    )

    _, bfs_pos = bfs_env.reset()
    _, rl_pos = rl_env.reset()
    if not (np.array_equal(bfs_env.ground_truth, rl_env.ground_truth) and bfs_pos == rl_pos):
        raise RuntimeError(f"seed={seed} 时 BFS 和 RL 没有初始化到同一张地图")

    bfs_planner = BFSFrontierPlanner()

    bfs_total_steps = 0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_alive = True
    bfs_round = 0
    bfs_target = None

    rl_total_steps = 0
    rl_total_area = 0.0
    rl_done = False
    rl_alive = True
    rl_round = 0
    rl_target = None

    max_rounds = max_bfs_compare_rounds(TEST_MAP_SIZE)
    for step_idx in range(1, max_rounds + 1):
        if bfs_alive and not bfs_done:
            if bfs_target is None:
                bfs_target = bfs_planner.find_nearest_frontier(bfs_env.explored_map, bfs_env.robot_pos)
                if bfs_target is None:
                    bfs_alive = False
            if bfs_target is not None:
                bfs_total_steps, bfs_total_area, bfs_done, info = _step_one_policy(
                    bfs_env,
                    bfs_target,
                    bfs_total_steps,
                    bfs_total_area,
                )
                bfs_round = step_idx
                reached = bfs_env.robot_pos[0] == bfs_target[0] and bfs_env.robot_pos[1] == bfs_target[1]
                if reached or (info["blocked"] and not reached):
                    bfs_target = None

        if rl_alive and not rl_done:
            if rl_target is None:
                _state_tensor, rl_target = rl_runtime.select_target(rl_env.explored_map, rl_env.robot_pos)
                if rl_target is None:
                    rl_alive = False
            if rl_target is not None:
                rl_total_steps, rl_total_area, rl_done, info = _step_one_policy(
                    rl_env,
                    rl_target,
                    rl_total_steps,
                    rl_total_area,
                )
                rl_round = step_idx
                reached = rl_env.robot_pos[0] == rl_target[0] and rl_env.robot_pos[1] == rl_target[1]
                if reached or (info["blocked"] and not reached):
                    rl_target = None

        if (not bfs_alive or bfs_done) and (not rl_alive or rl_done):
            break

    bfs_result = CompareResult("BFS", bfs_total_steps, bfs_total_area, bfs_done, bfs_round)
    rl_result = CompareResult("RL", rl_total_steps, rl_total_area, rl_done, rl_round)
    return bfs_result, rl_result


def _print_summary(results):
    valid_results = [item for item in results if item["bfs"].done and item["rl"].done]
    invalid_results = [item for item in results if not (item["bfs"].done and item["rl"].done)]

    rl_better = 0
    bfs_better = 0
    tied = 0
    improvement_rates = []
    rl_worse_seeds = []

    for item in valid_results:
        seed = item["seed"]
        bfs_steps = item["bfs"].total_steps
        rl_steps = item["rl"].total_steps

        if rl_steps < bfs_steps:
            rl_better += 1
        elif rl_steps > bfs_steps:
            bfs_better += 1
            rl_worse_seeds.append(seed)
        else:
            tied += 1

        improvement_rates.append((bfs_steps - rl_steps) / bfs_steps * 100.0)

    total_valid = len(valid_results)
    total_all = len(results)
    avg_improvement = float(np.mean(improvement_rates)) if improvement_rates else 0.0

    print("\n=== 批量对比汇总 ===")
    print(f"测试地图数量: {total_all}")
    print(f"双方均成功达到 95% 的地图数量: {total_valid}")
    print(f"存在未完成情况的地图数量: {len(invalid_results)}")

    if total_valid > 0:
        print("\n=== 优劣比例，按 steps 判断 ===")
        print(f"RL 更优: {rl_better}/{total_valid} = {rl_better / total_valid * 100.0:.2f}%")
        print(f"BFS 更优: {bfs_better}/{total_valid} = {bfs_better / total_valid * 100.0:.2f}%")
        print(f"二者相同: {tied}/{total_valid} = {tied / total_valid * 100.0:.2f}%")
        print(f"\nRL 平均相比 BFS 优的百分比: {avg_improvement:.2f}%")
    else:
        print("\n没有双方均完成的有效样本，无法计算优劣比例和平均提升百分比。")

    if rl_worse_seeds:
        print("\n=== RL steps 多于 BFS 的随机种子 ===")
        print(", ".join(str(seed) for seed in rl_worse_seeds))
    else:
        print("\n没有出现 RL steps 多于 BFS 的有效样本。")

    if invalid_results:
        print("\n=== 未纳入平均提升统计的地图 ===")
        for item in invalid_results:
            bfs = item["bfs"]
            rl = item["rl"]
            print(
                f"seed={item['seed']} | "
                f"BFS done={bfs.done}, steps={bfs.total_steps} | "
                f"RL done={rl.done}, steps={rl.total_steps}"
            )


def main():
    parser = argparse.ArgumentParser(description="Batch compare BFS baseline with RL policy")
    parser.add_argument("--weights", type=str, default=str(TEST_CHECKPOINT_PATH))
    parser.add_argument("--fusion-weights", type=str, default=str(TEST_FUSION_WEIGHTS_PATH))
    parser.add_argument("--use-best", action="store_true")
    args = parser.parse_args()

    if args.use_best:
        weight_path = str(BEST_CHECKPOINT_PATH)
        fusion_weight_path = str(BEST_FUSION_WEIGHTS_PATH)
    else:
        weight_path = args.weights
        fusion_weight_path = args.fusion_weights

    rl_runtime = RLRuntime(weight_path, fusion_weight_path, map_size=TEST_MAP_SIZE)
    seeds = [SEED_START + idx for idx in range(NUM_COMPARE_SEEDS)]
    results = []

    print("=== 开始批量对比 BFS vs RL ===")
    print(f"地图数量: {NUM_COMPARE_SEEDS}")
    print(f"随机种子范围: {seeds[0]} - {seeds[-1]}")
    print(f"地图尺寸: {TEST_MAP_SIZE}")
    print(f"地图生成器: {MAP_GENERATOR}")
    print(f"RL 主干权重: {weight_path}")
    print(f"RL 融合权重: {fusion_weight_path}\n")

    for idx, seed in enumerate(seeds, start=1):
        bfs_result, rl_result = _run_one_seed(seed, rl_runtime)
        results.append({"seed": seed, "bfs": bfs_result, "rl": rl_result})

        if bfs_result.done and rl_result.done:
            improvement = (bfs_result.total_steps - rl_result.total_steps) / bfs_result.total_steps * 100.0
            verdict = "RL优" if rl_result.total_steps < bfs_result.total_steps else "BFS优" if rl_result.total_steps > bfs_result.total_steps else "相同"
            print(
                f"[{idx:02d}/{NUM_COMPARE_SEEDS}] seed={seed} | "
                f"BFS steps={bfs_result.total_steps:4d} | "
                f"RL steps={rl_result.total_steps:4d} | "
                f"RL提升={improvement:7.2f}% | {verdict}"
            )
        else:
            print(
                f"[{idx:02d}/{NUM_COMPARE_SEEDS}] seed={seed} | "
                f"BFS done={bfs_result.done}, steps={bfs_result.total_steps:4d} | "
                f"RL done={rl_result.done}, steps={rl_result.total_steps:4d} | 未纳入有效均值"
            )

    _print_summary(results)


if __name__ == "__main__":
    main()
