"""
使用方法

1. 默认对比 BFS 与启发式策略，并使用 `config.py` 中的测试地图参数：
   python compare_eva.py

2. 开启同屏双栏可视化：
   python compare_eva.py --vis
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from bfs import BFSFrontierPlanner, max_bfs_compare_rounds
from config import MAP_GENERATOR, SENSOR_RANGE, TEST_MAP_SIZE, TEST_SEED
from grid_env import GridWorldEnv
from heuristic_runtime import HeuristicRuntime


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

        def _on_close(_event):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", _on_close)

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
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.03)
        except Exception:
            self.closed = True
            return False
        if not self.is_alive():
            self.closed = True
            return False
        return True

    def close(self):
        self.closed = True
        try:
            if plt.fignum_exists(self.fig.number):
                plt.close(self.fig)
        except Exception:
            pass


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


def _run_comparison(show_vis: bool):
    bfs_env = GridWorldEnv(map_size=TEST_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    heuristic_env = GridWorldEnv(map_size=TEST_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    _, bfs_pos = bfs_env.reset()
    _, heuristic_pos = heuristic_env.reset()
    if not (np.array_equal(bfs_env.ground_truth, heuristic_env.ground_truth) and bfs_pos == heuristic_pos):
        raise RuntimeError("BFS 和启发式环境没有初始化到同一张地图，请检查 TEST_SEED / MAP_GENERATOR 配置")

    bfs_planner = BFSFrontierPlanner()
    # 强制关闭 debug_print，保持终端清爽
    heuristic_runtime = HeuristicRuntime(map_size=TEST_MAP_SIZE, debug_print=False)

    bfs_total_steps = 0
    bfs_total_price_of_steps = 0.0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_alive = True
    bfs_round = 0
    bfs_target = None
    bfs_end_reason = "达到最大对比轮数仍未完成 95% 探索"

    heuristic_total_steps = 0
    heuristic_total_price_of_steps = 0.0
    heuristic_total_area = 0.0
    heuristic_done = False
    heuristic_alive = True
    heuristic_round = 0
    heuristic_target = None
    heuristic_end_reason = "达到最大对比轮数仍未完成 95% 探索"

    visualizer = DualMapVisualizer(TEST_MAP_SIZE) if show_vis else None
    max_rounds = max_bfs_compare_rounds(TEST_MAP_SIZE)
    
    try:
        for step_idx in range(1, max_rounds + 1):
            if bfs_alive and not bfs_done:
                if bfs_target is None:
                    bfs_target = bfs_planner.find_nearest_frontier(bfs_env.explored_map, bfs_env.robot_pos)
                    if bfs_target is None:
                        bfs_alive = False
                        bfs_end_reason = "BFS 找不到可达 frontier，但尚未达到 95% 探索"
                if bfs_target is not None:
                    bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, info = _step_one_policy(bfs_env, bfs_target, bfs_total_steps, bfs_total_price_of_steps, bfs_total_area)
                    bfs_round = step_idx
                    if bfs_done:
                        bfs_end_reason = "BFS 已达到 95% 探索"
                    reached = (bfs_env.robot_pos[0] == bfs_target[0] and bfs_env.robot_pos[1] == bfs_target[1])
                    if reached or info.get("collision", False):
                        bfs_target = None

            if heuristic_alive and not heuristic_done:
                if heuristic_target is None:
                    _state_tensor, heuristic_target = heuristic_runtime.select_target(heuristic_env.explored_map, heuristic_env.robot_pos)
                    if heuristic_target is None:
                        heuristic_alive = False
                        heuristic_end_reason = "启发式策略找不到可达 frontier"

                if heuristic_target is not None:
                    prev_pos = heuristic_env.robot_pos.copy()
                    heuristic_total_steps, heuristic_total_price_of_steps, heuristic_total_area, heuristic_done, info = _step_one_policy(
                        heuristic_env,
                        heuristic_target,
                        heuristic_total_steps,
                        heuristic_total_price_of_steps,
                        heuristic_total_area,
                    )
                    move_vec = (
                        float(heuristic_env.robot_pos[0] - prev_pos[0]),
                        float(heuristic_env.robot_pos[1] - prev_pos[1]),
                    )
                    if np.linalg.norm(move_vec) > 1e-6:
                        heuristic_runtime.previous_direction = move_vec
                    heuristic_round = step_idx
                    if heuristic_done:
                        heuristic_end_reason = "启发式策略已达到 95% 探索"
                    reached = (heuristic_env.robot_pos[0] == heuristic_target[0] and heuristic_env.robot_pos[1] == heuristic_target[1])
                    if reached or info.get("collision", False):
                        heuristic_target = None

            if visualizer is not None:
                still_open = visualizer.update(
                    bfs_env,
                    heuristic_env,
                    step_idx,
                    (bfs_total_steps, bfs_total_area),
                    (heuristic_total_steps, heuristic_total_area),
                )
                if not still_open:
                    break

            if (not bfs_alive or bfs_done) and (not heuristic_alive or heuristic_done):
                break

    finally:
        if visualizer is not None:
            visualizer.close()

    return ("BFS", bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, bfs_round, bfs_end_reason), ("Heuristic", heuristic_total_steps, heuristic_total_price_of_steps, heuristic_total_area, heuristic_done, heuristic_round, heuristic_end_reason)


def _print_results(bfs_result, heuristic_result):
    print("\n=== 对比结果 ===")
    for res in (bfs_result, heuristic_result):
        print(f"{res[0]:>10} | steps={res[1]:7.1f} | price_of_steps={res[2]:7.1f} | area={res[3]:7.1f} | done={res[4]} | rounds={res[5]}")
    if not bfs_result[4]:
        print(f"BFS done=False 原因: {bfs_result[6]}")
    if not heuristic_result[4]:
        print(f"启发式 done=False 原因: {heuristic_result[6]}")


def main():
    parser = argparse.ArgumentParser(description="Compare BFS baseline with heuristic policy on the same test map")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    bfs_result, heuristic_result = _run_comparison(args.vis)
    _print_results(bfs_result, heuristic_result)


if __name__ == "__main__":
    main()