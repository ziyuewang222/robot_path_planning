"""regional_legacy 的 BFS / heuristic 对比脚本。

运行参数统一由 `config.py` 与本文件顶部常量决定；不接受终端参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAP_SIZE, MAP_GENERATOR, SENSOR_RANGE, TEST_SEED, resolve_test_checkpoint_path
from core.bfs import find_nearest_frontier
from core.cv_extractor import SemanticExtractor
from core.heuristic import score_frontiers
from core.pretreatment import build_frontier_mask
from core.separating_runtime import HeuristicRuntimeState, step_one_policy
from env.grid_env import GridWorldEnv


SHOW_VISUALIZATION = True
DEFAULT_CHECKPOINT_PATH = resolve_test_checkpoint_path()
VISUALIZE_MAP_SIZE = DEFAULT_MAP_SIZE
VISUALIZE_SEED = TEST_SEED


class DualMapVisualizer:
    def __init__(self, map_size: int):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        self.images = {}
        self.robot_marks = {}
        self.closed = False
        for idx, ax in enumerate(self.axes):
            ax.set_xticks([])
            ax.set_yticks([])
            self.images[idx] = ax.imshow(np.zeros((map_size, map_size)), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
            self.robot_marks[idx], = ax.plot([], [], "ro", markersize=7)
        self.fig.suptitle("Same-map comparison: BFS vs Heuristic", fontsize=15, fontweight="bold")
        self.fig.canvas.mpl_connect("close_event", lambda _event: setattr(self, "closed", True))

    def is_alive(self) -> bool:
        return (not self.closed) and plt.fignum_exists(self.fig.number)

    @staticmethod
    def _to_display(explored_map):
        display = np.zeros_like(explored_map, dtype=np.float32)
        display[explored_map == -1] = 0.5
        display[explored_map == 0] = 1.0
        display[explored_map == 1] = 0.0
        return display

    def update(self, bfs_env: GridWorldEnv, heuristic_env: GridWorldEnv, step_idx: int, bfs_stats, heuristic_stats) -> bool:
        if not self.is_alive():
            self.closed = True
            return False
        self.images[0].set_data(self._to_display(bfs_env.explored_map))
        self.images[1].set_data(self._to_display(heuristic_env.explored_map))
        self.robot_marks[0].set_data([bfs_env.robot_pos[1]], [bfs_env.robot_pos[0]])
        self.robot_marks[1].set_data([heuristic_env.robot_pos[1]], [heuristic_env.robot_pos[0]])
        self.axes[0].set_title(f"BFS | step={step_idx} | area={bfs_stats[1]:.0f} | price={bfs_stats[0]:.0f}")
        self.axes[1].set_title(f"Heuristic | step={step_idx} | area={heuristic_stats[1]:.0f} | price={heuristic_stats[0]:.0f}")
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events(); plt.pause(0.001)
        return self.is_alive()

    def close(self):
        self.closed = True
        try:
            if plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
        except Exception:
            pass


def _print_results(bfs_result, heuristic_result):
    print("\n=== 对比结果 ===")
    for res in (bfs_result, heuristic_result):
        print(f"{res[0]:>10} | steps={res[1]:7.1f} | price_of_steps={res[2]:7.1f} | area={res[3]:7.1f} | done={res[4]} | rounds={res[5]}")
    if not bfs_result[4]:
        print(f"BFS done=False 原因: {bfs_result[6]}")
    if not heuristic_result[4]:
        print(f"启发式 done=False 原因: {heuristic_result[6]}")


def _run_comparison(show_vis: bool, seed: int = TEST_SEED):
    HeuristicRuntimeState.reset()
    bfs_env = GridWorldEnv(map_size=VISUALIZE_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    heuristic_env = GridWorldEnv(map_size=VISUALIZE_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    _, bfs_pos = bfs_env.reset()
    _, heuristic_pos = heuristic_env.reset()
    if not (np.array_equal(bfs_env.ground_truth, heuristic_env.ground_truth) and list(bfs_pos) == list(heuristic_pos)):
        raise RuntimeError("BFS 和启发式环境没有初始化到同一张地图，请检查 seed / MAP_GENERATOR 配置")

    bfs_total_steps = 0.0
    bfs_total_price_of_steps = 0.0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_alive = True
    bfs_round = 0
    bfs_target = None
    bfs_end_reason = "达到最大对比轮数仍未完成 95% 探索"

    heuristic_total_steps = 0.0
    heuristic_total_price_of_steps = 0.0
    heuristic_total_area = 0.0
    heuristic_done = False
    heuristic_alive = True
    heuristic_round = 0
    heuristic_target = None
    heuristic_end_reason = "达到最大对比轮数仍未完成 95% 探索"

    semantic_extractor = SemanticExtractor(DEFAULT_CHECKPOINT_PATH)
    visualizer = DualMapVisualizer(VISUALIZE_MAP_SIZE) if show_vis else None
    max_rounds = 200

    try:
        for step_idx in range(1, max_rounds + 1):
            if bfs_alive and not bfs_done:
                frontier = build_frontier_mask(bfs_env.explored_map)
                if bfs_target is None:
                    bfs_target = find_nearest_frontier(bfs_env.explored_map, frontier, bfs_env.robot_pos)
                    if bfs_target is None:
                        bfs_alive = False
                        bfs_end_reason = "BFS 找不到可达 frontier，但尚未达到 95% 探索"
                if bfs_target is not None:
                    step_result = step_one_policy(bfs_env, bfs_target)
                    bfs_total_steps += step_result.physical_steps
                    bfs_total_price_of_steps += step_result.physical_steps
                    bfs_total_area += float(step_result.info.get("delta_area", 0.0))
                    bfs_round = step_idx
                    bfs_done = step_result.done
                    if bfs_done:
                        bfs_end_reason = "BFS 已达到 95% 探索"
                    reached = (bfs_env.robot_pos[0] == bfs_target[0] and bfs_env.robot_pos[1] == bfs_target[1])
                    if reached or step_result.info.get("collision", False) or step_result.info.get("blocked", False):
                        bfs_target = None

            if heuristic_alive and not heuristic_done:
                frontier = build_frontier_mask(heuristic_env.explored_map)
                
                # 移除 heuristic_target is None 判定，每步强制重规划
                semantic_map = semantic_extractor.predict(heuristic_env.explored_map)
                heuristic_target, _score_map = score_frontiers(heuristic_env.explored_map, frontier, heuristic_env.robot_pos, semantic_map)
                
                if heuristic_target is None:
                    heuristic_alive = False
                    heuristic_end_reason = "启发式策略找不到可达 frontier"
                
                if heuristic_target is not None:
                    step_result = step_one_policy(heuristic_env, heuristic_target)
                    heuristic_total_steps += step_result.physical_steps
                    heuristic_total_price_of_steps += step_result.physical_steps
                    heuristic_total_area += float(step_result.info.get("delta_area", 0.0))
                    heuristic_round = step_idx
                    heuristic_done = step_result.done
                    if heuristic_done:
                        heuristic_end_reason = "启发式策略已达到 95% 探索"
                    # 这里不再需要依据 reached 状态清空 heuristic_target，下一轮循环会重新覆盖

            if visualizer is not None:
                if not visualizer.update(bfs_env, heuristic_env, step_idx, (bfs_total_steps, bfs_total_area), (heuristic_total_steps, heuristic_total_area)):
                    break
            if (not bfs_alive or bfs_done) and (not heuristic_alive or heuristic_done):
                break
    finally:
        if visualizer is not None:
            visualizer.close()

    return (
        "BFS", bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, bfs_round, bfs_end_reason
    ), (
        "Heuristic", heuristic_total_steps, heuristic_total_price_of_steps, heuristic_total_area, heuristic_done, heuristic_round, heuristic_end_reason
    )


def main():
    bfs_result, heuristic_result = _run_comparison(SHOW_VISUALIZATION)
    _print_results(bfs_result, heuristic_result)


if __name__ == "__main__":
    main()