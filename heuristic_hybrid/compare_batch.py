"""
批量对比 BFS 与启发式策略在多张随机地图上的探索步数。

使用方法：

1. 对比 NUM_COMPARE_SEEDS 张地图：
   python compare_batch.py

2. 使用 best 参数：
   python compare_batch.py --use-best
"""

import argparse

import numpy as np

from bfs import BFSFrontierPlanner, max_bfs_compare_rounds
from config import MAP_GENERATOR, SENSOR_RANGE, TEST_MAP_SIZE, TEST_SEED
from grid_env import GridWorldEnv
from heuristic_runtime import HeuristicRuntime

# 批量对比的随机地图数量，修改这里即可调整测试规模
NUM_COMPARE_SEEDS = 30

# 随机种子起点。实际使用的种子为：SEED_START, SEED_START + 1, ...
SEED_START = TEST_SEED

class CompareResult:
    def __init__(self, name, total_steps, total_price_of_steps, total_area, done, rounds):
        self.name = name
        self.total_steps = total_steps
        self.total_price_of_steps = total_price_of_steps
        self.total_area = total_area
        self.done = done
        self.rounds = rounds


def _step_one_policy(env, target_coord, total_steps: int, total_price_of_steps: float, total_area: float):
    # price=steps：按宏动作对应的物理步数统计，物理步数由 A* 规划得到的路径长度决定。
    planned_path = env.planner.plan(env.explored_map, env.robot_pos, target_coord)
    physical_steps = float(max(0, len(planned_path) - 1)) if planned_path is not None else 0.0

    next_explored, next_pos, _reward, _n_steps, done, info = env.step_macro(target_coord)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    total_steps += physical_steps
    total_price_of_steps += physical_steps
    total_area += info.get("coverage_gain", 0.0)
    return total_steps, total_price_of_steps, total_area, done, info


def _run_one_seed(seed: int, heuristic_runtime: HeuristicRuntime):
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
    _, heuristic_pos = rl_env.reset()
    if not (np.array_equal(bfs_env.ground_truth, rl_env.ground_truth) and bfs_pos == heuristic_pos):
        raise RuntimeError(f"seed={seed} 时 BFS 和启发式策略没有初始化到同一张地图")

    bfs_planner = BFSFrontierPlanner()
    heuristic_runtime.previous_direction = None

    bfs_total_steps = 0
    bfs_total_price_of_steps = 0.0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_alive = True
    bfs_round = 0
    bfs_target = None

    rl_total_steps = 0
    rl_total_price_of_steps = 0.0
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
                bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, info = _step_one_policy(
                    bfs_env,
                    bfs_target,
                    bfs_total_steps,
                    bfs_total_price_of_steps,
                    bfs_total_area,
                )
                bfs_round = step_idx
                reached = bfs_env.robot_pos[0] == bfs_target[0] and bfs_env.robot_pos[1] == bfs_target[1]
                if reached or info.get("collision", False):
                    bfs_target = None

        if rl_alive and not rl_done:
            if rl_target is None:
                _state_tensor, rl_target = heuristic_runtime.select_target(rl_env.explored_map, rl_env.robot_pos)
                if rl_target is None:
                    rl_alive = False
            if rl_target is not None:
                prev_pos = rl_env.robot_pos.copy()
                rl_total_steps, rl_total_price_of_steps, rl_total_area, rl_done, info = _step_one_policy(
                    rl_env,
                    rl_target,
                    rl_total_steps,
                    rl_total_price_of_steps,
                    rl_total_area,
                )
                move_vec = (
                    float(rl_env.robot_pos[0] - prev_pos[0]),
                    float(rl_env.robot_pos[1] - prev_pos[1]),
                )
                if np.linalg.norm(move_vec) > 1e-6:
                    heuristic_runtime.previous_direction = move_vec
                rl_round = step_idx
                reached = rl_env.robot_pos[0] == rl_target[0] and rl_env.robot_pos[1] == rl_target[1]
                if reached or info.get("collision", False):
                    rl_target = None

        if (not bfs_alive or bfs_done) and (not rl_alive or rl_done):
            break

    bfs_result = CompareResult("BFS", bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, bfs_round)
    heuristic_result = CompareResult("Heuristic", rl_total_steps, rl_total_price_of_steps, rl_total_area, rl_done, rl_round)
    return bfs_result, heuristic_result


def _print_summary(results):
    valid_results = [item for item in results if item["bfs"].done and item["heuristic"].done]
    invalid_results = [item for item in results if not (item["bfs"].done and item["heuristic"].done)]

    heuristic_better = 0
    bfs_better = 0
    tied = 0
    improvement_rates = []
    heuristic_worse_seeds = []

    for item in valid_results:
        seed = item["seed"]
        bfs_price = item["bfs"].total_price_of_steps
        heuristic_price = item["heuristic"].total_price_of_steps

        if heuristic_price < bfs_price:
            heuristic_better += 1
        elif heuristic_price > bfs_price:
            bfs_better += 1
            heuristic_worse_seeds.append(seed)
        else:
            tied += 1

        if bfs_price > 0:
            improvement_rates.append((bfs_price - heuristic_price) / bfs_price * 100.0)

    total_valid = len(valid_results)
    total_all = len(results)
    avg_improvement = float(np.mean(improvement_rates)) if improvement_rates else 0.0

    print("\n=== 批量对比汇总 ===")
    print(f"测试地图数量: {total_all}")
    print(f"双方均成功达到 95% 的地图数量: {total_valid}")
    print(f"存在未完成情况的地图数量: {len(invalid_results)}")

    if total_valid > 0:
        print("\n=== 优劣比例，按 price_of_steps 判断 ===")
        print(f"启发式更优: {heuristic_better}/{total_valid} = {heuristic_better / total_valid * 100.0:.2f}%")
        print(f"BFS 更优: {bfs_better}/{total_valid} = {bfs_better / total_valid * 100.0:.2f}%")
        print(f"二者相同: {tied}/{total_valid} = {tied / total_valid * 100.0:.2f}%")
        print(f"\n启发式平均相比 BFS 优的百分比: {avg_improvement:.2f}%")
    else:
        print("\n没有双方均完成的有效样本，无法计算优劣比例和平均提升百分比。")

    if heuristic_worse_seeds:
        print("\n=== 启发式 price_of_steps 多于 BFS 的随机种子 ===")
        print(", ".join(str(seed) for seed in heuristic_worse_seeds))
    else:
        print("\n没有出现启发式 price_of_steps 多于 BFS 的有效样本。")

    if invalid_results:
        print("\n=== 未纳入平均提升统计的地图 ===")
        for item in invalid_results:
            bfs = item["bfs"]
            heuristic = item["heuristic"]
            print(
                f"seed={item['seed']} | "
                f"BFS done={bfs.done}, steps={bfs.total_steps}, price_of_steps={bfs.total_price_of_steps:.0f} | "
                f"启发式 done={heuristic.done}, steps={heuristic.total_steps}, price_of_steps={heuristic.total_price_of_steps:.0f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Batch compare BFS baseline with heuristic policy")
    parser.parse_args()

    heuristic_runtime = HeuristicRuntime(map_size=TEST_MAP_SIZE, debug_print=False)
    seeds = [SEED_START + idx for idx in range(NUM_COMPARE_SEEDS)]
    results = []

    print("=== 开始批量对比 BFS vs 启发式策略 ===")
    print(f"地图数量: {NUM_COMPARE_SEEDS}")
    print(f"随机种子范围: {seeds[0]} - {seeds[-1]}")
    print(f"地图尺寸: {TEST_MAP_SIZE}")
    print(f"地图生成器: {MAP_GENERATOR}")
    print(f"启发式运行参数来源: config.py")
    print()

    for idx, seed in enumerate(seeds, start=1):
        bfs_result, rl_result = _run_one_seed(seed, heuristic_runtime)
        results.append({"seed": seed, "bfs": bfs_result, "heuristic": rl_result})

        if bfs_result.done and rl_result.done:
            if bfs_result.total_price_of_steps > 0:
                improvement = (bfs_result.total_price_of_steps - rl_result.total_price_of_steps) / bfs_result.total_price_of_steps * 100.0
                improvement_text = f"启发式提升={improvement:7.2f}%"
            else:
                improvement = None
                improvement_text = "启发式提升=N/A( BFS price=0 )"
            verdict = "启发式优" if rl_result.total_price_of_steps < bfs_result.total_price_of_steps else "BFS优" if rl_result.total_price_of_steps > bfs_result.total_price_of_steps else "相同"
            print(
                f"[{idx:02d}/{NUM_COMPARE_SEEDS}] seed={seed} | "
                f"BFS price={bfs_result.total_price_of_steps:4.0f} | "
                f"启发式 price={rl_result.total_price_of_steps:4.0f} | "
                f"{improvement_text} | {verdict}"
            )
        else:
            print(
                f"[{idx:02d}/{NUM_COMPARE_SEEDS}] seed={seed} | "
                f"BFS done={bfs_result.done}, steps={bfs_result.total_steps:4d}, price={bfs_result.total_price_of_steps:4.0f} | "
                f"启发式 done={rl_result.done}, steps={rl_result.total_steps:4d}, price={rl_result.total_price_of_steps:4.0f} | 未纳入有效均值"
            )

    _print_summary(results)


if __name__ == "__main__":
    main()
