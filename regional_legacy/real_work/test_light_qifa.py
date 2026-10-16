from __future__ import annotations

"""light_qifa 轻量几何启发式探索策略的验收测试。

地图模板固定使用 easy_big_room（real_config.MAP_GENERATOR），
在若干固定随机种子上跑完整探索，要求每个种子都探索完成。

用法（在 real_work 目录下执行）：
    python test_light_qifa.py                # 使用 real_config 中的种子/参数
    python test_light_qifa.py 150 1 7 42 99  # 指定种子
退出码为 0 表示全部通过；否则打印失败明细并返回 1。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPLORATION_SUCCESS_THRESHOLD
from env.grid_env import GridWorldEnv
from real_work.light_qifa import LightHeuristicConfig, decide_frontier
from real_work.real_config import (
    HEURISTIC_CONFIG,
    MAP_GENERATOR,
    MAP_SIZE,
    MAX_MACRO_STEPS,
    MAX_ROUNDS,
    SEED,
    SENSOR_RANGE,
    validate_config,
)

# easy_big_room 模板下要求所有测试种子都能探索完成。
SUCCESS_RATE_THRESHOLD = 1.0
DEFAULT_SEEDS = (SEED, 1, 7)


@dataclass
class EpisodeResult:
    seed: int
    done: bool = False
    rounds: int = 0
    physical_steps: float = 0.0
    collisions: int = 0
    explored_free: int = 0
    total_free: int = 0

    @property
    def coverage(self) -> float:
        if self.total_free <= 0:
            return 0.0
        return self.explored_free / self.total_free


@dataclass
class EvaluationSummary:
    results: List[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.done) / len(self.results)

    @property
    def mean_steps(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.physical_steps for r in self.results]))


def _new_env(seed: int) -> GridWorldEnv:
    return GridWorldEnv(
        map_size=MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=seed,
        map_generator=MAP_GENERATOR,
    )


def run_episode(
    seed: int,
    config: Optional[LightHeuristicConfig] = None,
    show_visualization: bool = False,
) -> EpisodeResult:
    """用轻量几何启发式在 easy_big_room 地图上跑一轮完整探索。"""
    env = _new_env(seed)
    env.reset()

    result = EpisodeResult(
        seed=seed,
        total_free=int(np.sum(env.ground_truth == 0)),
    )
    previous_direction = None
    visualizer = None
    if show_visualization:
        import matplotlib.pyplot as plt

        plt.ion()
        visualizer = plt

    try:
        for _ in range(MAX_ROUNDS):
            decision = decide_frontier(
                env.explored_map,
                env.robot_pos,
                previous_direction=previous_direction,
                config=config or HEURISTIC_CONFIG,
            )
            if decision.target is None:
                break

            robot_before = tuple(env.robot_pos)
            path = env.planner.plan(env.explored_map, env.robot_pos, decision.target)
            result.physical_steps += float(max(0, len(path) - 1)) if path is not None else 0.0

            explored_map, robot_pos, _reward, _steps, _done, info = env.step_macro(
                decision.target,
                max_steps=MAX_MACRO_STEPS,
            )
            env.explored_map = explored_map
            env.robot_pos = robot_pos
            result.rounds += 1
            result.collisions += int(bool(info.get("collision", False)))

            direction = np.asarray(env.robot_pos, dtype=np.float32) - np.asarray(
                robot_before, dtype=np.float32
            )
            if np.linalg.norm(direction) > 1e-6:
                previous_direction = direction

            result.explored_free = int(np.sum(env.explored_map == 0))
            if result.coverage >= EXPLORATION_SUCCESS_THRESHOLD:
                result.done = True
                break

            if visualizer is not None:
                env.render(result.rounds)
    finally:
        if visualizer is not None:
            import matplotlib.pyplot as plt

            plt.close("all")

    result.explored_free = int(np.sum(env.explored_map == 0))
    result.done = result.done or result.coverage >= EXPLORATION_SUCCESS_THRESHOLD
    return result


def evaluate(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config: Optional[LightHeuristicConfig] = None,
) -> EvaluationSummary:
    summary = EvaluationSummary()
    for seed in seeds:
        summary.results.append(run_episode(seed, config=config))
    return summary


def _print_summary(summary: EvaluationSummary) -> None:
    print(f"generator={MAP_GENERATOR} | size={MAP_SIZE} | sensor_range={SENSOR_RANGE}")
    print(f"{'seed':>8} | {'done':>5} | {'rounds':>7} | {'steps':>8} | {'coverage':>9} | collisions")
    for result in summary.results:
        print(
            f"{result.seed:>8} | {str(result.done):>5} | {result.rounds:>7} | "
            f"{result.physical_steps:>8.1f} | {result.coverage:>9.4f} | {result.collisions}"
        )
    print(
        f"success_rate={summary.success_rate:.2%} | mean_steps={summary.mean_steps:.1f} | "
        f"required={SUCCESS_RATE_THRESHOLD:.0%}"
    )


def test_light_heuristic_completes_easy_big_room() -> None:
    """集成测试：easy_big_room 模板下多个种子都必须探索完成。"""
    validate_config()
    summary = evaluate(DEFAULT_SEEDS)
    _print_summary(summary)
    assert summary.success_rate >= SUCCESS_RATE_THRESHOLD, (
        f"探索成功率 {summary.success_rate:.2%} 低于要求的 {SUCCESS_RATE_THRESHOLD:.0%}"
    )


def test_frontier_mask_and_decision_on_toy_map() -> None:
    """单元测试：在手工小地图上验证前沿提取与决策接口。"""
    explored_map = np.full((7, 7), -1, dtype=np.int8)
    explored_map[2:5, 2:5] = 0
    explored_map[2, 1] = 1

    decision = decide_frontier(explored_map, (3, 3))
    assert decision.frontier_mask.ndim == 2
    assert np.any(decision.frontier_mask == 1)
    assert decision.target is not None
    tx, ty = decision.target
    assert explored_map[tx, ty] == 0
    assert decision.frontier_mask[tx, ty] == 1

    fully_explored = np.zeros((5, 5), dtype=np.int8)
    empty_decision = decide_frontier(fully_explored, (2, 2))
    assert empty_decision.target is None


def main(argv: Sequence[str]) -> int:
    seeds = tuple(int(value) for value in argv[1:]) or DEFAULT_SEEDS
    validate_config()
    print(
        f"HEURISTIC_CONFIG: bfs={HEURISTIC_CONFIG.bfs_weight} "
        f"turn={HEURISTIC_CONFIG.turn_weight} "
        f"obstacle={HEURISTIC_CONFIG.obstacle_weight} "
        f"safe_distance={HEURISTIC_CONFIG.obstacle_safe_distance}"
    )
    summary = evaluate(seeds)
    _print_summary(summary)

    if summary.success_rate >= SUCCESS_RATE_THRESHOLD:
        print("PASS: 所有种子均探索完成")
        return 0

    failed = [r.seed for r in summary.results if not r.done]
    print(f"FAIL: 未完成的种子 = {failed}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
