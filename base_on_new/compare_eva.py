"""
使用方法

1. 默认对比 BFS 与 RL，并使用 `config.py` 中的测试地图参数和测试权重路径：
   python compare_eva.py

2. 指定 RL 主干权重文件：
   python compare_eva.py --weights /path/to/your_checkpoint.pth

3. 指定融合权重文件：
   python compare_eva.py --fusion-weights /path/to/your_fusion_weights.pth

4. 开启同屏双栏可视化：
   python compare_eva.py --vis
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from bfs import BFSFrontierPlanner, max_bfs_compare_rounds
from config import (
    BEST_CHECKPOINT_PATH,
    BEST_FUSION_WEIGHTS_PATH,
    MAP_GENERATOR,
    SENSOR_RANGE,
    TEST_CHECKPOINT_PATH,
    TEST_FUSION_WEIGHTS,
    TEST_MAP_SIZE,
    TEST_SEED,
    resolve_test_fusion_weights_path,
)
from grid_env import GridWorldEnv
from rl_runtime import RLRuntime


class DualMapVisualizer:
    def __init__(self, map_size: int):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        self.images = {}
        self.robot_marks = {}
        for idx, ax in enumerate(self.axes):
            ax.set_xticks([])
            ax.set_yticks([])
            self.images[idx] = ax.imshow(np.zeros((map_size, map_size)), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
            self.robot_marks[idx], = ax.plot([], [], "ro", markersize=7)
        self.fig.suptitle("Same-map comparison: BFS vs RL", fontsize=15, fontweight="bold")

    @staticmethod
    def _to_display(explored_map):
        display = np.zeros_like(explored_map, dtype=np.float32)
        display[explored_map == -1] = 0.5
        display[explored_map == 0] = 1.0
        display[explored_map == 1] = 0.0
        return display

    def update(self, bfs_env: GridWorldEnv, rl_env: GridWorldEnv, step_idx: int, bfs_stats, rl_stats):
        self.images[0].set_data(self._to_display(bfs_env.explored_map))
        self.images[1].set_data(self._to_display(rl_env.explored_map))
        self.robot_marks[0].set_data([bfs_env.robot_pos[1]], [bfs_env.robot_pos[0]])
        self.robot_marks[1].set_data([rl_env.robot_pos[1]], [rl_env.robot_pos[0]])
        self.axes[0].set_title(f"BFS | step={step_idx} | area={bfs_stats[1]:.0f} | price={bfs_stats[0]:.0f}")
        self.axes[1].set_title(f"RL  | step={step_idx} | area={rl_stats[1]:.0f} | price={rl_stats[0]:.0f}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.03)

    def close(self):
        plt.close(self.fig)


def _calc_price_of_steps(info, map_generator: str):
    if map_generator != "easy_big_room":
        return float(info["n_steps"])
    room_steps = float(info.get("room_steps", 0.0))
    corridor_steps = float(info.get("corridor_steps", 0.0))
    return room_steps + corridor_steps * 2.0


def _step_one_policy(env, target_coord, total_steps: int, total_price_of_steps: float, total_area: float):
    next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(target_coord)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    total_steps += n_steps
    total_price_of_steps += _calc_price_of_steps(info, env.map_generator)
    total_area += info["delta_area"]
    return total_steps, total_price_of_steps, total_area, done, info


def _run_comparison(weight_path: str, fusion_weight_path: str, show_vis: bool):
    bfs_env = GridWorldEnv(map_size=TEST_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    rl_env = GridWorldEnv(map_size=TEST_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    _, bfs_pos = bfs_env.reset()
    _, rl_pos = rl_env.reset()
    if not (np.array_equal(bfs_env.ground_truth, rl_env.ground_truth) and bfs_pos == rl_pos):
        raise RuntimeError("BFS 和 RL 环境没有初始化到同一张地图，请检查 TEST_SEED / MAP_GENERATOR 配置")

    bfs_planner = BFSFrontierPlanner()
    rl_runtime = RLRuntime(weight_path, fusion_weight_path, map_size=TEST_MAP_SIZE)

    bfs_total_steps = 0
    bfs_total_price_of_steps = 0.0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_alive = True
    bfs_round = 0
    bfs_target = None
    bfs_end_reason = "达到最大对比轮数仍未完成 95% 探索"

    rl_total_steps = 0
    rl_total_price_of_steps = 0.0
    rl_total_area = 0.0
    rl_done = False
    rl_alive = True
    rl_round = 0
    rl_target = None

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
                    if reached or (info["blocked"] and not reached):
                        bfs_target = None

            if rl_alive and not rl_done:
                if rl_target is None:
                    _, rl_target = rl_runtime.select_target(rl_env.explored_map, rl_env.robot_pos)
                    if rl_target is None:
                        rl_alive = False
                if rl_target is not None:
                    rl_total_steps, rl_total_price_of_steps, rl_total_area, rl_done, info = _step_one_policy(rl_env, rl_target, rl_total_steps, rl_total_price_of_steps, rl_total_area)
                    rl_round = step_idx
                    reached = (rl_env.robot_pos[0] == rl_target[0] and rl_env.robot_pos[1] == rl_target[1])
                    if reached or (info["blocked"] and not reached):
                        rl_target = None

            if visualizer is not None:
                visualizer.update(bfs_env, rl_env, step_idx, (bfs_total_steps, bfs_total_area), (rl_total_steps, rl_total_area))

            if (not bfs_alive or bfs_done) and (not rl_alive or rl_done):
                break
    finally:
        if visualizer is not None:
            visualizer.close()

    return ("BFS", bfs_total_steps, bfs_total_price_of_steps, bfs_total_area, bfs_done, bfs_round, bfs_end_reason), ("RL", rl_total_steps, rl_total_price_of_steps, rl_total_area, rl_done, rl_round, "")


def _print_results(bfs_result, rl_result):
    print("\n=== 对比结果 ===")
    for res in (bfs_result, rl_result):
        print(f"{res[0]:>3} | steps={res[1]:4d} | price_of_steps={res[2]:7.1f} | area={res[3]:7.1f} | done={res[4]} | rounds={res[5]}")
    if not bfs_result[3]:
        print(f"BFS done=False 原因: {bfs_result[5]}")


def main():
    parser = argparse.ArgumentParser(description="Compare BFS baseline with RL policy on the same test map")
    parser.add_argument("--weights", type=str, default=str(TEST_CHECKPOINT_PATH))
    parser.add_argument("--fusion-weights", type=str, default=None)
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    if args.use_best:
        weight_path = str(BEST_CHECKPOINT_PATH)
        fusion_weight_path = str(BEST_FUSION_WEIGHTS_PATH)
    else:
        weight_path = args.weights
        fusion_weight_path = args.fusion_weights

    fusion_weight_path = resolve_test_fusion_weights_path(fusion_weight_path)
    if fusion_weight_path is None and TEST_FUSION_WEIGHTS is not None:
        print(f"使用内置三元组融合权重: {tuple(TEST_FUSION_WEIGHTS)}")

    bfs_result, rl_result = _run_comparison(weight_path, fusion_weight_path, args.vis)
    _print_results(bfs_result, rl_result)


if __name__ == "__main__":
    main()
