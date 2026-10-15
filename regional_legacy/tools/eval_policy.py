"""regional_legacy 的策略评估脚本。

展示三张图：观测输入、语义拓扑图、价值热力图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAP_SIZE, MAP_GENERATOR, SENSOR_RANGE, TEST_SEED, resolve_test_checkpoint_path
from core.cv_extractor import SemanticExtractor
from core.explorer_policy import decide_frontier
from env.grid_env import GridWorldEnv


DEFAULT_CHECKPOINT_PATH = resolve_test_checkpoint_path()
VISUALIZE_MAP_SIZE = DEFAULT_MAP_SIZE
VISUALIZE_SEED = TEST_SEED
INPUT_CMAP = ListedColormap(["#3b3b3b", "#f4d03f", "#9aa0a6"])
INPUT_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], INPUT_CMAP.N)
SEMANTIC_CMAP = ListedColormap(["#000000", "#202020", "#2e86de", "#f4d03f"])
SEMANTIC_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], SEMANTIC_CMAP.N)
SCORE_CMAP = plt.get_cmap("magma").copy(); SCORE_CMAP.set_bad(alpha=0.0)


class PolicyVisualizer:
    def __init__(self, map_size: int):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
        self.closed = False
        self.mode_text = self.fig.suptitle("", fontsize=14, fontweight="bold")
        self.input_img = self.axes[0].imshow(np.zeros((map_size, map_size)), cmap=INPUT_CMAP, norm=INPUT_NORM, interpolation="nearest")
        self.semantic_img = self.axes[1].imshow(np.zeros((map_size, map_size)), cmap=SEMANTIC_CMAP, norm=SEMANTIC_NORM, interpolation="nearest")
        self.score_bg = self.axes[2].imshow(np.zeros((map_size, map_size)), cmap=INPUT_CMAP, norm=INPUT_NORM, interpolation="nearest")
        self.score_img = self.axes[2].imshow(np.zeros((map_size, map_size), dtype=np.float32), cmap=SCORE_CMAP, interpolation="nearest")
        self.robot_marks = []
        for ax in self.axes:
            ax.set_xticks([]); ax.set_yticks([])
            self.robot_marks.append(ax.plot([], [], "ro", markersize=7)[0])
        self.axes[0].set_title("Observed Input")
        self.axes[1].set_title("Predicted Semantic Map")
        self.axes[2].set_title("Frontier Scores")
        self.fig.canvas.mpl_connect("close_event", lambda _event: setattr(self, "closed", True))

    def is_alive(self) -> bool:
        return (not self.closed) and plt.fignum_exists(self.fig.number)

    @staticmethod
    def _known_semantic_display(semantic_map: np.ndarray, explored_map: np.ndarray) -> np.ndarray:
        display = np.array(semantic_map, dtype=np.int64, copy=True)
        display[explored_map == -1] = -1
        return display

    @staticmethod
    def _score_display(scores: np.ndarray, frontier_mask: np.ndarray) -> np.ndarray:
        display = np.array(scores, dtype=np.float32, copy=True)
        display[frontier_mask != 1] = np.nan
        finite_mask = np.isfinite(display)
        if not finite_mask.any():
            return display
        valid = display[finite_mask]
        vmin = float(valid.min()); vmax = float(valid.max())
        if abs(vmax - vmin) < 1e-6:
            display[finite_mask] = 0.5
            return display
        display[finite_mask] = (display[finite_mask] - vmin) / (vmax - vmin)
        return display

    def update(self, explored_map, semantic_map, score_map, frontier_mask, robot_pos, step_idx: int, chosen_frontier, chosen_score, explored_area: float) -> bool:
        if not self.is_alive():
            self.closed = True
            return False
        self.input_img.set_data(explored_map)
        self.semantic_img.set_data(self._known_semantic_display(semantic_map, explored_map))
        self.score_bg.set_data(explored_map)
        self.score_img.set_data(self._score_display(score_map, frontier_mask))
        for mark in self.robot_marks:
            mark.set_data([robot_pos[1]], [robot_pos[0]])
        self.mode_text.set_text(f"step={step_idx} | area={explored_area:.0f}")
        if chosen_frontier is None:
            self.axes[2].set_title("Frontier Scores | no reachable frontier")
        else:
            self.axes[2].set_title(f"Frontier Scores | target=({chosen_frontier[0]}, {chosen_frontier[1]}) | score={chosen_score:.2f}")
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events(); plt.pause(0.001)
        return self.is_alive()

    def close(self):
        self.closed = True
        try:
            if plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
        except Exception:
            pass


def _step_one_policy(env: GridWorldEnv, target_coord):
    goal = [int(target_coord[0]), int(target_coord[1])]
    planned_path = env.planner.plan(env.explored_map, env.robot_pos, goal)
    physical_steps = float(max(0, len(planned_path) - 1)) if planned_path is not None else 0.0
    next_explored, next_pos, _reward, _n_steps, done, info = env.step_macro(goal)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    return physical_steps, done, info


def evaluate_policy(model_path: Path, map_size: int, seed: int | None = None) -> None:
    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    explored_map, _robot_pos = env.reset()
    semantic_extractor = SemanticExtractor(model_path)
    visualizer = PolicyVisualizer(map_size)

    total_steps = 0.0
    total_price_of_steps = 0.0
    total_area = 0.0
    step_idx = 0

    try:
        while True:
            semantic_map = semantic_extractor.predict(explored_map)
            decision = decide_frontier(explored_map, env.robot_pos, semantic_map)
            chosen_frontier = decision.chosen_frontier
            score_map = decision.score_map
            frontier_mask = decision.frontier_mask
            chosen_score = None if chosen_frontier is None else float(score_map[chosen_frontier[0], chosen_frontier[1]])

            if not visualizer.update(explored_map, semantic_map, score_map, frontier_mask, env.robot_pos, step_idx, chosen_frontier, chosen_score, total_area):
                break
            if chosen_frontier is None:
                print("没有可达 frontier，评估结束。")
                break

            physical_steps, done, info = _step_one_policy(env, chosen_frontier)
            total_steps += physical_steps
            total_price_of_steps += physical_steps
            total_area += float(info.get("coverage_gain", info.get("delta_area", 0.0)))
            explored_map = env.explored_map
            step_idx += 1
            if done:
                print("环境已经完成探索，评估结束。")
                break
    finally:
        visualizer.close()

    print("\n=== Policy Evaluation Summary ===")
    print(f"steps={total_steps:.1f} | price_of_steps={total_price_of_steps:.1f} | area={total_area:.1f}")


if __name__ == "__main__":
    evaluate_policy(DEFAULT_CHECKPOINT_PATH, VISUALIZE_MAP_SIZE, VISUALIZE_SEED)
