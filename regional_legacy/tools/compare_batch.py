"""regional_legacy 的 BFS / heuristic 批量对比脚本。

该脚本直接复用 `core.separating_runtime` 的双策略执行逻辑，只是在外层批量遍历多个随机种子。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAP_SIZE, resolve_test_checkpoint_path
from core.cv_extractor import SemanticExtractor
from core.separating_runtime import run_dual_comparison


NUM_COMPARE_SEEDS = 30
SEED_START = 35


class CompareResult:
    def __init__(self, name, total_steps, total_price_of_steps, total_area, done, rounds):
        self.name = name
        self.total_steps = total_steps
        self.total_price_of_steps = total_price_of_steps
        self.total_area = total_area
        self.done = done
        self.rounds = rounds


def _print_summary(results):
    valid_results = [item for item in results if item["bfs"].done and item["heuristic"].done]
    invalid_results = [item for item in results if not (item["bfs"].done and item["heuristic"].done)]

    rl_better = 0
    bfs_better = 0
    tied = 0
    improvement_rates = []
    rl_worse_seeds = []

    for item in valid_results:
        seed = item["seed"]
        bfs_price = item["bfs"].total_price_of_steps
        rl_price = item["heuristic"].total_price_of_steps

        if rl_price < bfs_price:
            rl_better += 1
        elif rl_price > bfs_price:
            bfs_better += 1
            rl_worse_seeds.append(seed)
        else:
            tied += 1

        if bfs_price > 0:
            improvement_rates.append((bfs_price - rl_price) / bfs_price * 100.0)

    total_valid = len(valid_results)
    total_all = len(results)
    avg_improvement = float(np.mean(improvement_rates)) if improvement_rates else 0.0

    print("\n=== 批量对比汇总 ===")
    print(f"测试地图数量: {total_all}")
    print(f"双方均成功达到 99.5% 的地图数量: {total_valid}")
    print(f"存在未完成情况的地图数量: {len(invalid_results)}")

    if total_valid > 0:
        print("\n=== 优劣比例，按 price_of_steps 判断 ===")
        print(f"Heuristic 更优: {rl_better}/{total_valid} = {rl_better / total_valid * 100.0:.2f}%")
        print(f"BFS 更优: {bfs_better}/{total_valid} = {bfs_better / total_valid * 100.0:.2f}%")
        print(f"二者相同: {tied}/{total_valid} = {tied / total_valid * 100.0:.2f}%")
        print(f"\nHeuristic 平均相比 BFS 优的百分比: {avg_improvement:.2f}%")
    else:
        print("\n没有双方均完成的有效样本，无法计算优劣比例和平均提升百分比。")

    if rl_worse_seeds:
        print("\n=== Heuristic price_of_steps 多于 BFS 的随机种子 ===")
        print(", ".join(str(seed) for seed in rl_worse_seeds))
    else:
        print("\n没有出现 Heuristic price_of_steps 多于 BFS 的有效样本。")

    if invalid_results:
        print("\n=== 未纳入平均提升统计的地图 ===")
        for item in invalid_results:
            bfs = item["bfs"]
            heuristic = item["heuristic"]
            print(
                f"seed={item['seed']} | "
                f"BFS done={bfs.done}, steps={bfs.total_steps}, price_of_steps={bfs.total_price_of_steps:.0f} | "
                f"Heuristic done={heuristic.done}, steps={heuristic.total_steps}, price_of_steps={heuristic.total_price_of_steps:.0f}"
            )


def main():
    seeds = [SEED_START + idx for idx in range(NUM_COMPARE_SEEDS)]
    results = []
    semantic_extractor = SemanticExtractor(resolve_test_checkpoint_path())

    print("=== 开始批量对比 BFS vs Heuristic ===")
    print(f"地图数量: {NUM_COMPARE_SEEDS}")
    print(f"随机种子范围: {seeds[0]} - {seeds[-1]}")
    print(f"地图尺寸: {DEFAULT_MAP_SIZE}")
    print(f"地图生成器: {semantic_extractor.__class__.__name__ if False else 'easy_big_room'}")
    print(f"特征提取模型权重: {resolve_test_checkpoint_path()}")
    print(f"\n")

    for idx, seed in enumerate(seeds, start=1):
        result = run_dual_comparison(seed, DEFAULT_MAP_SIZE, semantic_extractor)
        bfs_result = CompareResult("BFS", result.bfs.total_steps, result.bfs.total_price_of_steps, result.bfs.total_area, result.bfs.done, result.bfs.rounds)
        heuristic_result = CompareResult("Heuristic", result.heuristic.total_steps, result.heuristic.total_price_of_steps, result.heuristic.total_area, result.heuristic.done, result.heuristic.rounds)
        results.append({"seed": seed, "bfs": bfs_result, "heuristic": heuristic_result})

        if bfs_result.done and heuristic_result.done:
            improvement = (bfs_result.total_price_of_steps - heuristic_result.total_price_of_steps) / bfs_result.total_price_of_steps * 100.0 if bfs_result.total_price_of_steps > 0 else 0.0
            verdict = "Heuristic优" if heuristic_result.total_price_of_steps < bfs_result.total_price_of_steps else "BFS优" if heuristic_result.total_price_of_steps > bfs_result.total_price_of_steps else "相同"
            print(
                f"[{idx:02d}/{NUM_COMPARE_SEEDS}] seed={seed} | "
                f"BFS steps={bfs_result.total_steps:7.1f}, macro={bfs_result.rounds:4d} | "
                f"Heuristic steps={heuristic_result.total_steps:7.1f}, macro={heuristic_result.rounds:4d} | "
                f"Heuristic节省={improvement:7.2f}% | {verdict}"
            )
        else:
            print(
                f"[{idx:02d}/{NUM_COMPARE_SEEDS}] seed={seed} | "
                f"BFS steps={bfs_result.total_steps:7.1f}, macro={bfs_result.rounds:4d} | "
                f"Heuristic steps={heuristic_result.total_steps:7.1f}, macro={heuristic_result.rounds:4d} | 未纳入有效均值"
            )

    _print_summary(results)


if __name__ == "__main__":
    main()
