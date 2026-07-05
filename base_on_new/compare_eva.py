"""
使用方法

1. 默认对比 BFS 与 RL，并使用 `config.py` 中的测试地图参数：
   - `TEST_MAP_SIZE`
   - `TEST_SEED`
   - `MAP_GENERATOR`

2. 直接运行：
   python compare_eva.py

3. 指定 RL 主干权重文件：
   python compare_eva.py --weights /path/to/your_checkpoint.pth

4. 指定融合权重文件（不指定则默认使用 latest）：
   python compare_eva.py --fusion-weights /path/to/your_fusion_weights.pth

5. 开启同屏双栏可视化（左 BFS，右 RL，同步推进，整个过程只创建一个窗口）：
   python compare_eva.py --vis

说明
- 左边显示 BFS 策略在同一张测试地图上的探索过程。
- 右边显示 RL 策略在同一张测试地图上的探索过程。
- 两边是同步推进、连续刷新、同一个窗口内并排显示。
- 不会逐步创建新窗口。
- 你只需要改 `base_on_new/config.py` 里的测试配置，就能统一控制该脚本的地图尺寸和障碍生成逻辑。
- RL 推理时会同时加载“主干权重 + fusion_weights”，其中融合权重默认使用 latest，也可以单独指定。
"""

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from bfs import BFSFrontierPlanner
from config import (
    BEST_CHECKPOINT_PATH,
    BEST_FUSION_WEIGHTS_PATH,
    LATEST_CHECKPOINT_PATH,
    LATEST_FUSION_WEIGHTS_PATH,
    MAP_GENERATOR,
    SENSOR_RANGE,
    SIGMA,
    TEST_MAP_SIZE,
    TEST_SEED,
    max_macro_steps_for,
)
from grid_env import GridWorldEnv
from models import SpatialActionDecision, SpatialFCN
from state_pipeline import StateTensorPipeline


@dataclass
class EvalResult:
    name: str
    total_steps: int
    total_area: float
    reached_done: bool
    episodes: int

    @property
    def score(self):
        return self.total_area - 0.1 * self.total_steps

    @property
    def success(self):
        return self.reached_done


class DualMapVisualizer:
    def __init__(self, map_size: int):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        self.map_size = map_size
        self.images = {}
        self.robot_marks = {}

        titles = ["BFS baseline", "RL policy"]
        for idx, ax in enumerate(self.axes):
            ax.set_title(titles[idx], fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            empty = np.zeros((map_size, map_size), dtype=np.float32)
            self.images[idx] = ax.imshow(empty, cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
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

        self.axes[0].set_title(
            f"BFS | step={step_idx} | pos={tuple(bfs_env.robot_pos)} | area={bfs_stats[1]:.0f} | steps={bfs_stats[0]}"
        )
        self.axes[1].set_title(
            f"RL  | step={step_idx} | pos={tuple(rl_env.robot_pos)} | area={rl_stats[1]:.0f} | steps={rl_stats[0]}"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.03)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def _make_env():
    return GridWorldEnv(
        map_size=TEST_MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=TEST_SEED,
        map_generator=MAP_GENERATOR,
    )


def _load_rl_policy(weight_path: str, fusion_weight_path: str, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到主干权重文件: {weight_path}")
    if not os.path.exists(fusion_weight_path):
        raise FileNotFoundError(f"找不到 fusion_weights 文件: {fusion_weight_path}")

    policy_net = SpatialFCN().to(device)
    state_dict = torch.load(weight_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.load_fusion_weights(fusion_weight_path)
    policy_net.eval()
    return policy_net


def _step_one_policy(env, select_target_fn, total_steps: int, total_area: float):
    explored, pos = env.explored_map.copy(), env.robot_pos.copy()
    chosen_coord = select_target_fn(explored, pos)
    if chosen_coord is None:
        return total_steps, total_area, True, False

    next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(chosen_coord)
    env.explored_map = next_explored
    env.robot_pos = next_pos
    total_steps += n_steps
    total_area += info["delta_area"]
    return total_steps, total_area, done, True


def _run_comparison(weight_path: str, fusion_weight_path: str, show_vis: bool):
    bfs_env = _make_env()
    rl_env = _make_env()

    _, bfs_pos = bfs_env.reset()
    _, rl_pos = rl_env.reset()

    if not (np.array_equal(bfs_env.ground_truth, rl_env.ground_truth) and bfs_pos == rl_pos):
        raise RuntimeError("BFS 和 RL 环境没有初始化到同一张地图，请检查 TEST_SEED / MAP_GENERATOR 配置")

    bfs_planner = BFSFrontierPlanner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_policy = _load_rl_policy(weight_path, fusion_weight_path, device)
    rl_pipeline = StateTensorPipeline(map_size=TEST_MAP_SIZE, sigma=SIGMA, device=device)
    rl_decision = SpatialActionDecision(rl_policy, epsilon=0.0, device=device)

    bfs_total_steps = 0
    bfs_total_area = 0.0
    bfs_done = False
    bfs_alive = True
    bfs_round = 0

    rl_total_steps = 0
    rl_total_area = 0.0
    rl_done = False
    rl_alive = True
    rl_round = 0

    visualizer = DualMapVisualizer(TEST_MAP_SIZE) if show_vis else None

    max_rounds = max_macro_steps_for(TEST_MAP_SIZE) * 2
    try:
        for step_idx in range(1, max_rounds + 1):
            if bfs_alive and not bfs_done:
                def bfs_select(explored, pos):
                    return bfs_planner.find_nearest_frontier(explored, pos)

                chosen_coord = bfs_select(bfs_env.explored_map, bfs_env.robot_pos)
                if chosen_coord is None:
                    bfs_alive = False
                else:
                    bfs_total_steps, bfs_total_area, bfs_done, bfs_alive = _step_one_policy(
                        bfs_env,
                        bfs_select,
                        bfs_total_steps,
                        bfs_total_area,
                    )
                    bfs_round = step_idx

            if rl_alive and not rl_done:
                def rl_select(explored, pos):
                    state_tensor = rl_pipeline.process(explored, pos)
                    chosen_coord, _q_masked, _q_raw = rl_decision.select_action(state_tensor, explore=False)
                    return chosen_coord

                chosen_coord = rl_select(rl_env.explored_map, rl_env.robot_pos)
                if chosen_coord is None:
                    rl_alive = False
                else:
                    rl_total_steps, rl_total_area, rl_done, rl_alive = _step_one_policy(
                        rl_env,
                        rl_select,
                        rl_total_steps,
                        rl_total_area,
                    )
                    rl_round = step_idx

            if visualizer is not None:
                visualizer.update(
                    bfs_env,
                    rl_env,
                    step_idx,
                    (bfs_total_steps, bfs_total_area),
                    (rl_total_steps, rl_total_area),
                )

            if (not bfs_alive or bfs_done) and (not rl_alive or rl_done):
                break
    finally:
        if visualizer is not None:
            visualizer.close()

    bfs_result = EvalResult("BFS", bfs_total_steps, bfs_total_area, bfs_done, bfs_round)
    rl_result = EvalResult("RL", rl_total_steps, rl_total_area, rl_done, rl_round)
    return bfs_result, rl_result


def _pick_better(bfs_result: EvalResult, rl_result: EvalResult):
    bfs_success = bfs_result.success
    rl_success = rl_result.success

    if bfs_success and not rl_success:
        return bfs_result, "BFS 完成任务，而 RL 未完成"
    if rl_success and not bfs_success:
        return rl_result, "RL 完成任务，而 BFS 未完成"

    if bfs_success and rl_success:
        if bfs_result.total_steps < rl_result.total_steps:
            return bfs_result, "两者都完成任务，但 BFS 步数更少"
        if rl_result.total_steps < bfs_result.total_steps:
            return rl_result, "两者都完成任务，但 RL 步数更少"
        if bfs_result.total_area > rl_result.total_area:
            return bfs_result, "两者步数相同，但 BFS 探索面积更大"
        if rl_result.total_area > bfs_result.total_area:
            return rl_result, "两者步数相同，但 RL 探索面积更大"
        return bfs_result, "两者步数和探索面积都相同，按平局处理"

    if bfs_result.total_area > rl_result.total_area:
        return bfs_result, "两者都未完成任务，但 BFS 探索面积更大"
    if rl_result.total_area > bfs_result.total_area:
        return rl_result, "两者都未完成任务，但 RL 探索面积更大"
    if bfs_result.total_steps < rl_result.total_steps:
        return bfs_result, "两者都未完成任务，但 BFS 步数更少"
    if rl_result.total_steps < bfs_result.total_steps:
        return rl_result, "两者都未完成任务，但 RL 步数更少"
    return bfs_result, "两者未完成且指标相同，按平局处理"


def _print_results(bfs_result: EvalResult, rl_result: EvalResult):
    print("\n=== 对比结果 ===")
    for res in (bfs_result, rl_result):
        print(
            f"{res.name:>3} | steps={res.total_steps:4d} | area={res.total_area:7.1f} | "
            f"done={res.reached_done} | success={res.success}"
        )

    better, reason = _pick_better(bfs_result, rl_result)
    print(f"\n更优策略: {better.name}")
    print(f"判定依据: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Compare BFS baseline with RL policy on the same test map")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(LATEST_CHECKPOINT_PATH),
        help="RL 主干权重文件路径，默认使用 latest",
    )
    parser.add_argument(
        "--fusion-weights",
        type=str,
        default=str(LATEST_FUSION_WEIGHTS_PATH),
        help="RL fusion_weights 文件路径，默认使用 latest",
    )
    parser.add_argument(
        "--use-best",
        action="store_true",
        help="同时使用 best 主干权重和 best fusion_weights（会覆盖 --weights / --fusion-weights 默认值）",
    )
    parser.add_argument("--vis", action="store_true", help="开启同屏左右双栏可视化")
    args = parser.parse_args()

    if args.use_best:
        weight_path = str(BEST_CHECKPOINT_PATH)
        fusion_weight_path = str(BEST_FUSION_WEIGHTS_PATH)
    else:
        weight_path = args.weights
        fusion_weight_path = args.fusion_weights

    print("=== 开始同图对比评估 ===")
    print(f"测试地图: size={TEST_MAP_SIZE}, seed={TEST_SEED}, generator={MAP_GENERATOR}")
    print(f"RL 主干权重: {weight_path}")
    print(f"RL fusion_weights: {fusion_weight_path}")

    bfs_result, rl_result = _run_comparison(weight_path, fusion_weight_path, args.vis)
    _print_results(bfs_result, rl_result)

    if args.vis:
        plt.show()


if __name__ == "__main__":
    main()
