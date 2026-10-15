from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPLORATION_SUCCESS_THRESHOLD
from core.bfs import find_nearest_frontier
from core.pretreatment import build_frontier_mask
from env.grid_env import GridWorldEnv
from real_work.light_qifa import FrontierDecision, decide_frontier
from real_work.real_config import (
    HEURISTIC_CONFIG,
    MAP_GENERATOR,
    MAP_SIZE,
    MAX_MACRO_STEPS,
    MAX_ROUNDS,
    SEED,
    SENSOR_RANGE,
    SHOW_VISUALIZATION,
    validate_config,
)


@dataclass
class PolicyStats:
    name: str
    physical_steps: float = 0.0
    explored_area: float = 0.0
    rounds: int = 0
    collisions: int = 0
    blocked_rounds: int = 0
    done: bool = False


class ComparisonVisualizer:
    def __init__(self, map_size: int):
        plt.ion()
        self.fig, self.axes = plt.subplots(
            1, 2,
            figsize=(14, 7),
            constrained_layout=True,
        )
        self.closed = False
        self.images = []
        self.robot_marks = []
        for axis in self.axes:
            axis.set_xticks([])
            axis.set_yticks([])
            image = axis.imshow(
                np.zeros((map_size, map_size), dtype=np.float32),
                cmap="viridis",
                origin="upper",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            mark, = axis.plot([], [], "ro", markersize=7)
            self.images.append(image)
            self.robot_marks.append(mark)
        self.fig.suptitle("BFS baseline vs light geometric heuristic")
        self.fig.canvas.mpl_connect(
            "close_event",
            lambda _event: setattr(self, "closed", True),
        )

    @staticmethod
    def _display(explored_map: np.ndarray) -> np.ndarray:
        display = np.full(explored_map.shape, 0.5, dtype=np.float32)
        display[explored_map == 0] = 1.0
        display[explored_map == 1] = 0.0
        return display

    def is_alive(self) -> bool:
        return not self.closed and plt.fignum_exists(self.fig.number)

    def update(
        self,
        bfs_env: GridWorldEnv,
        light_env: GridWorldEnv,
        step_idx: int,
        bfs_stats: PolicyStats,
        light_stats: PolicyStats,
        light_decision: Optional[FrontierDecision],
    ) -> bool:
        if not self.is_alive():
            self.closed = True
            return False

        environments = (bfs_env, light_env)
        stats = (bfs_stats, light_stats)
        titles = ("BFS baseline", "Light heuristic")
        for index, (env, policy_stats, title) in enumerate(zip(environments, stats, titles)):
            self.images[index].set_data(self._display(env.explored_map))
            self.robot_marks[index].set_data(
                [env.robot_pos[1]],
                [env.robot_pos[0]],
            )
            self.axes[index].set_title(
                f"{title} | round={policy_stats.rounds} | "
                f"steps={policy_stats.physical_steps:.0f} | "
                f"area={policy_stats.explored_area:.0f}"
            )

        if light_decision is not None and light_decision.target is not None:
            self.axes[1].plot(
                [light_decision.target[1]],
                [light_decision.target[0]],
                marker="x",
                color="red",
                markersize=9,
            )

        self.fig.suptitle(f"BFS baseline vs light geometric heuristic | round={step_idx}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        return self.is_alive()

    def close(self) -> None:
        self.closed = True
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


def _step_policy(env: GridWorldEnv, target: tuple[int, int]) -> tuple[float, bool, dict]:
    path = env.planner.plan(env.explored_map, env.robot_pos, target)
    physical_steps = float(max(0, len(path) - 1)) if path is not None else 0.0
    explored_map, robot_pos, _reward, _n_steps, done, info = env.step_macro(
        target,
        max_steps=MAX_MACRO_STEPS,
    )
    env.explored_map = explored_map
    env.robot_pos = robot_pos
    return physical_steps, done, info


def _new_env() -> GridWorldEnv:
    return GridWorldEnv(
        map_size=MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=SEED,
        map_generator=MAP_GENERATOR,
    )


def _update_stats(stats: PolicyStats, physical_steps: float, info: dict) -> None:
    stats.physical_steps += physical_steps
    stats.explored_area += float(info.get("delta_area", 0.0))
    stats.rounds += 1
    stats.collisions += int(bool(info.get("collision", False)))
    stats.blocked_rounds += int(bool(info.get("blocked", False)))
    stats.done = bool(info.get("success", False))


def run_comparison(show_visualization: bool = SHOW_VISUALIZATION) -> tuple[PolicyStats, PolicyStats]:
    validate_config()
    bfs_env = _new_env()
    light_env = _new_env()
    bfs_env.reset()
    light_env.reset()

    if not np.array_equal(bfs_env.ground_truth, light_env.ground_truth):
        raise RuntimeError("BFS 和轻量启发式没有使用同一张地图")
    if bfs_env.robot_pos != light_env.robot_pos:
        raise RuntimeError("BFS 和轻量启发式没有使用同一个起点")

    bfs_stats = PolicyStats(name="BFS")
    light_stats = PolicyStats(name="LightHeuristic")
    visualizer = ComparisonVisualizer(MAP_SIZE) if show_visualization else None
    previous_direction = None
    last_decision = None

    try:
        for step_idx in range(1, MAX_ROUNDS + 1):
            if not bfs_stats.done:
                bfs_frontier = build_frontier_mask(bfs_env.explored_map)
                bfs_target = find_nearest_frontier(
                    bfs_env.explored_map,
                    bfs_frontier,
                    bfs_env.robot_pos,
                )
                if bfs_target is None:
                    bfs_stats.done = _coverage_finished(bfs_env)
                else:
                    physical_steps, done, info = _step_policy(bfs_env, bfs_target)
                    _update_stats(bfs_stats, physical_steps, info)
                    bfs_stats.done = done or _coverage_finished(bfs_env)

            if not light_stats.done:
                last_decision = decide_frontier(
                    light_env.explored_map,
                    light_env.robot_pos,
                    previous_direction=previous_direction,
                    config=HEURISTIC_CONFIG,
                )
                light_target = last_decision.target
                if light_target is None:
                    light_stats.done = _coverage_finished(light_env)
                else:
                    robot_before = tuple(light_env.robot_pos)
                    physical_steps, done, info = _step_policy(light_env, light_target)
                    _update_stats(light_stats, physical_steps, info)
                    light_stats.done = done or _coverage_finished(light_env)
                    robot_after = tuple(light_env.robot_pos)
                    direction = np.asarray(robot_after, dtype=np.float32) - np.asarray(robot_before, dtype=np.float32)
                    if np.linalg.norm(direction) > 1e-6:
                        previous_direction = direction

            if visualizer is not None:
                if not visualizer.update(
                    bfs_env,
                    light_env,
                    step_idx,
                    bfs_stats,
                    light_stats,
                    last_decision,
                ):
                    break

            if bfs_stats.done and light_stats.done:
                break
    finally:
        if visualizer is not None:
            visualizer.close()

    return bfs_stats, light_stats


def _coverage_finished(env: GridWorldEnv) -> bool:
    free_cells = int(np.sum(env.ground_truth == 0))
    explored_free = int(np.sum(env.explored_map == 0))
    return explored_free >= free_cells * EXPLORATION_SUCCESS_THRESHOLD


def print_summary(bfs_stats: PolicyStats, light_stats: PolicyStats) -> None:
    print("\n=== BFS vs Light Heuristic ===")
    for stats in (bfs_stats, light_stats):
        print(
            f"{stats.name:>16} | done={stats.done} | "
            f"rounds={stats.rounds:4d} | "
            f"physical_steps={stats.physical_steps:8.1f} | "
            f"explored_area={stats.explored_area:8.1f} | "
            f"collisions={stats.collisions:3d} | "
            f"blocked={stats.blocked_rounds:3d}"
        )

    if bfs_stats.physical_steps > 0:
        saving = (bfs_stats.physical_steps - light_stats.physical_steps) / bfs_stats.physical_steps * 100.0
        print(f"Light heuristic step difference vs BFS: {saving:.2f}%")


def main() -> None:
    print(
        f"generator={MAP_GENERATOR} | size={MAP_SIZE} | seed={SEED} | "
        f"sensor_range={SENSOR_RANGE}"
    )
    bfs_stats, light_stats = run_comparison()
    print_summary(bfs_stats, light_stats)


if __name__ == "__main__":
    main()
