"""
使用方法:
python eval_policy.py

说明:
本文件已剥离所有的强化学习逻辑，重构为纯启发式打分演示脚本。
运行后将基于 config.py 的配置初始化地图，并在每一步动态展示所有 frontier 的启发式价值热力图。
"""

import matplotlib.pyplot as plt
import numpy as np

from config import MAP_GENERATOR, SENSOR_RANGE, TEST_MAP_SIZE, TEST_SEED
from grid_env import GridWorldEnv
from heuristic_runtime import HeuristicRuntime
from state_pipeline import StateTensorPipeline


BAR_COLORS = ["#00C853", "#FFAB00", "#D50000"]


def _build_background(explored_map: np.ndarray) -> np.ndarray:
    bg = np.full(explored_map.shape, np.nan, dtype=np.float32)
    bg[explored_map == 0] = 0.85
    bg[explored_map == 1] = 0.25
    return bg


def _build_frontier_heatmap(explored_map: np.ndarray, frontier_list: np.ndarray, utility: np.ndarray) -> np.ndarray:
    h, w = explored_map.shape
    heatmap_data = np.full((h, w), np.nan, dtype=np.float32)
    for idx, (fy, fx) in enumerate(frontier_list):
        if idx < len(utility):
            heatmap_data[fy, fx] = float(utility[idx])
    return heatmap_data


def _render_step(ax_map, ax_bar, explored_map, robot_pos, frontier_list, utility, chosen_coord, step_idx, done, branch_scores=None):
    ax_map.clear()
    ax_bar.clear()

    bg = _build_background(explored_map)
    ax_map.imshow(bg, cmap="gray", origin="upper", vmin=0, vmax=1)

    heatmap_data = _build_frontier_heatmap(explored_map, frontier_list, utility)
    im = ax_map.imshow(heatmap_data, cmap="plasma", origin="upper", interpolation="nearest")

    ax_map.plot(robot_pos[1], robot_pos[0], "o", color="#00E5FF", markersize=8, markeredgecolor="black", label="Robot")
    if chosen_coord is not None:
        ax_map.plot(chosen_coord[1], chosen_coord[0], "*", color="#76FF03", markersize=14, markeredgecolor="black", label="Best Frontier")

    ax_map.set_title(f"Heuristic Frontier Utility Heatmap - Step {step_idx}{' (Done)' if done else ''}")
    ax_map.legend(loc="upper right")

    if branch_scores is not None:
        labels = ["gain", "cost", "turn"]
        values = [float(branch_scores.get("gain", 0.0)), float(branch_scores.get("cost", 0.0)), float(branch_scores.get("turn", 0.0))]
        ax_bar.bar(labels, values, color=BAR_COLORS, width=0.6)
        ax_bar.axhline(0.0, color="black", linewidth=0.8)
        ax_bar.set_ylim(min(0.0, min(values) * 1.2), max(1e-6, max(values) * 1.2 + 1e-6))
        ax_bar.set_title("Chosen target branch scores")
        ax_bar.set_ylabel("Weighted score")
        for idx, value in enumerate(values):
            ax_bar.text(idx, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top")

    return im


def run_dynamic_heuristic_visualization():
    print("正在初始化地图并启动动态可视化...")
    print("快捷键：空格/右箭头=单步执行，p=暂停/继续，q=退出")

    env = GridWorldEnv(map_size=TEST_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    explored_map, robot_pos = env.reset()
    runtime = HeuristicRuntime(map_size=TEST_MAP_SIZE, debug_print=False)

    plt.ion()
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3.2, 1]})
    colorbar = None
    paused = False
    step_pending = False
    should_exit = False

    def _set_status_text():
        status = "PAUSED" if paused else "RUNNING"
        fig.suptitle(f"Heuristic Evaluation Visualization [{status}] - Space/Right: step, P: pause/resume, Q: quit")

    def _on_key(event):
        nonlocal paused, step_pending, should_exit
        key = (event.key or "").lower()
        if key == "p":
            paused = not paused
            _set_status_text()
            fig.canvas.draw_idle()
        elif key in (" ", "space", "right"):
            step_pending = True
            if paused:
                paused = False
            _set_status_text()
            fig.canvas.draw_idle()
        elif key == "q":
            should_exit = True

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _set_status_text()

    try:
        with np.errstate(all="ignore"):
            for step_idx in range(1, runtime.max_rounds + 1):
                while paused and not step_pending and not should_exit:
                    plt.pause(0.05)

                if should_exit:
                    print("收到退出指令，结束可视化。")
                    break

                state_tensor = runtime.process_state(explored_map, robot_pos)
                frontier_mask = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
                frontier_list = np.argwhere(frontier_mask)

                if len(frontier_list) == 0:
                    ax_map.clear()
                    ax_bar.clear()
                    ax_map.imshow(_build_background(explored_map), cmap="gray", origin="upper", vmin=0, vmax=1)
                    ax_map.set_title(f"Heuristic Frontier Utility Heatmap - Step {step_idx} (No frontier)")
                    ax_bar.set_title("Chosen target branch scores")
                    plt.pause(0.001)
                    print("当前地图没有可用的 frontier，结束可视化。")
                    break

                chosen_idx, chosen_coord, utility = runtime._select_by_heuristic(explored_map, state_tensor, robot_pos)
                if chosen_idx is None or chosen_coord is None or utility is None:
                    print("无法计算 frontier 价值，结束可视化。")
                    break

                frontier_coords, cost_scores = runtime._cost_only_bfs_frontier_scores(explored_map, robot_pos)
                cost_lookup = {tuple(coord.tolist()): float(score) for coord, score in zip(frontier_coords, cost_scores)}
                turn_penalties = runtime._turn_penalty(robot_pos, frontier_list)
                gain_map = state_tensor[StateTensorPipeline.UNKNOWN_FRONTIER_CHANNEL].detach().cpu().numpy()
                gains = np.array([gain_map[y, x] for y, x in frontier_list], dtype=np.float32)
                gain_norm = np.clip(gains / runtime.global_max_gain, 0.0, 1.0).astype(np.float32)
                cost_values = np.array([cost_lookup.get((int(y), int(x)), 1.0) for y, x in frontier_list], dtype=np.float32)
                cost_norm = np.clip(cost_values, 0.0, 1.0).astype(np.float32)
                turn_norm = np.clip(turn_penalties, 0.0, 1.0).astype(np.float32)

                branch_scores = {
                    "gain": runtime.w_gain * float(gain_norm[chosen_idx]),
                    "cost": runtime.w_cost * float(cost_norm[chosen_idx]),
                    "turn": runtime.w_turn * float(turn_norm[chosen_idx]),
                }

                im = _render_step(ax_map, ax_bar, explored_map, robot_pos, frontier_list, utility, chosen_coord, step_idx, False, branch_scores=branch_scores)
                if colorbar is None:
                    colorbar = fig.colorbar(im, ax=ax_map, label="Heuristic Final Utility Score")
                else:
                    colorbar.update_normal(im)

                fig.tight_layout()
                plt.pause(0.2)

                while paused and not step_pending and not should_exit:
                    plt.pause(0.05)

                if should_exit:
                    print("收到退出指令，结束可视化。")
                    break

                next_explored, next_pos, _reward, _n_steps, done, _info = env.step_macro(chosen_coord)
                step_pending = False

                move_vec = (float(next_pos[0] - robot_pos[0]), float(next_pos[1] - robot_pos[1]))
                if np.linalg.norm(move_vec) > 1e-6:
                    runtime.previous_direction = move_vec

                explored_map, robot_pos = next_explored, next_pos

                if done:
                    state_tensor = runtime.process_state(explored_map, robot_pos)
                    frontier_mask = state_tensor[StateTensorPipeline.FRONTIER_CHANNEL].detach().cpu().numpy() > 0.5
                    frontier_list = np.argwhere(frontier_mask)
                    if len(frontier_list) > 0:
                        chosen_idx, chosen_coord, utility = runtime._select_by_heuristic(explored_map, state_tensor, robot_pos)
                        frontier_coords, cost_scores = runtime._cost_only_bfs_frontier_scores(explored_map, robot_pos)
                        cost_lookup = {tuple(coord.tolist()): float(score) for coord, score in zip(frontier_coords, cost_scores)}
                        turn_penalties = runtime._turn_penalty(robot_pos, frontier_list)
                        gain_map = state_tensor[StateTensorPipeline.UNKNOWN_FRONTIER_CHANNEL].detach().cpu().numpy()
                        gains = np.array([gain_map[y, x] for y, x in frontier_list], dtype=np.float32)
                        gain_norm = np.clip(gains / runtime.global_max_gain, 0.0, 1.0).astype(np.float32)
                        cost_values = np.array([cost_lookup.get((int(y), int(x)), 1.0) for y, x in frontier_list], dtype=np.float32)
                        cost_norm = np.clip(cost_values, 0.0, 1.0).astype(np.float32)
                        turn_norm = np.clip(turn_penalties, 0.0, 1.0).astype(np.float32)
                        branch_scores = {
                            "gain": runtime.w_gain * float(gain_norm[chosen_idx]),
                            "cost": runtime.w_cost * float(cost_norm[chosen_idx]),
                            "turn": runtime.w_turn * float(turn_norm[chosen_idx]),
                        }
                    else:
                        chosen_coord, utility, branch_scores = None, None, None
                    im = _render_step(ax_map, ax_bar, explored_map, robot_pos, frontier_list, utility if utility is not None else np.array([]), chosen_coord, step_idx, True, branch_scores=branch_scores)
                    if colorbar is not None:
                        colorbar.update_normal(im)
                    fig.tight_layout()
                    plt.pause(0.001)
                    print(f"任务完成于第 {step_idx} 步。")
                    break
                elif paused:
                    step_pending = False
            else:
                print("达到最大迭代步数，结束可视化。")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    run_dynamic_heuristic_visualization()