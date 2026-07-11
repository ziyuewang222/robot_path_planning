"""
使用方法

1. 默认使用 `config.py` 中的 TEST_CHECKPOINT_PATH 和 TEST_FUSION_WEIGHTS_PATH 进行评估：
   python eval_policy.py

2. 指定主干权重和融合权重：
   python eval_policy.py --weights /path/to/checkpoint.pth --fusion-weights /path/to/fusion.pth

3. 只指定主干权重，融合权重仍默认使用 TEST_FUSION_WEIGHTS_PATH：
   python eval_policy.py --weights /path/to/checkpoint.pth

4. 指定使用 best：
   python eval_policy.py --use-best
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import (
    BEST_CHECKPOINT_PATH,
    BEST_FUSION_WEIGHTS_PATH,
    TEST_CHECKPOINT_PATH,
    TEST_FUSION_WEIGHTS_PATH,
    MAP_GENERATOR,
    SENSOR_RANGE,
    TEST_MAP_SIZE,
    TEST_SEED,
)
from grid_env import GridWorldEnv
from rl_runtime import RLRuntime
from state_pipeline import StateTensorPipeline


class PolicyInspector:
    def __init__(self, map_size: int):
        plt.ion()
        self.map_size = map_size
        self.local_radius = 45
        self.fig = plt.figure(figsize=(28, 16))
        self.fig.canvas.manager.set_window_title("Spatial FCN RL - Policy Inspector")
        gs = self.fig.add_gridspec(
            4,
            5,
            height_ratios=[1.1, 1.05, 1.05, 0.75],
            width_ratios=[1, 1, 1, 1, 1],
        )
        layout = [
            "map_overview", "ch0", "ch3", "ch4", "q_masked_global",
            "q_raw_local", "q_masked_local", "q_full", "q_half", "q_quarter",
            "full_patch", "half_patch", "quarter_patch", "q_hist", "fusion_weights",
            "feature_line", "status", "status_extra_1", "status_extra_2", "status_extra_3",
        ]
        self.axes = {k: self.fig.add_subplot(gs[i // 5, i % 5]) for i, k in enumerate(layout)}
        self.images = {}
        self.markers = {}

        self.titles = {
            "map_overview": "Map overview: known free / obstacle / unknown",
            "ch0": "Visibility-aware gain",
            "ch3": "BFS reachable distance",
            "ch4": "Valid frontier mask",
            "q_masked_global": "Global masked Q over frontiers",
            "q_raw_local": "Raw Q local crop",
            "q_masked_local": "Masked Q local crop",
            "q_full": "Full-scale branch contribution",
            "q_half": "Half-scale branch contribution",
            "q_quarter": "Quarter-scale branch contribution",
            "full_patch": "Full-scale feature mean",
            "half_patch": "Half-scale feature mean",
            "quarter_patch": "Quarter-scale upsampled feature mean",
            "q_hist": "Valid-frontier Q histogram",
            "fusion_weights": "Learned scale fusion weights",
            "feature_line": "Local row Q profile through target",
            "status": "Status / controls",
        }
        for key, ax in self.axes.items():
            ax.set_title(self.titles.get(key, ""), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        map_cmap = plt.cm.gray.copy()
        map_cmap.set_bad(color="#232323")
        q_cmap = plt.cm.plasma.copy()
        q_cmap.set_bad(color="#d8d8d8")

        state_specs = [
            ("map_overview", map_cmap, 0.0, 1.0),
            ("ch0", "gray", 0.0, 1.0),
            ("ch3", "viridis", 0.0, 1.0),
            ("ch4", "gray", 0.0, 1.0),
        ]
        for key, cmap, vmin, vmax in state_specs:
            self.images[key] = self.axes[key].imshow(
                np.zeros((map_size, map_size), dtype=np.float32),
                cmap=cmap,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            self._set_grid_view(key, (0, map_size, 0, map_size))

        for key in ["q_masked_global", "q_raw_local", "q_masked_local", "q_full", "q_half", "q_quarter"]:
            self.images[key] = self.axes[key].imshow(
                np.zeros((map_size, map_size), dtype=np.float32),
                cmap=q_cmap,
                origin="upper",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )

        for key in ["full_patch", "half_patch", "quarter_patch"]:
            self.images[key] = self.axes[key].imshow(
                np.zeros((map_size, map_size), dtype=np.float32),
                cmap="magma",
                origin="upper",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )

        self.fusion_bar = self.axes["fusion_weights"].bar(
            [0, 1, 2],
            [1 / 3, 1 / 3, 1 / 3],
            color=["#4C72B0", "#55A868", "#C44E52"],
        )
        self.axes["fusion_weights"].set_xticks([0, 1, 2])
        self.axes["fusion_weights"].set_xticklabels(["1x", "1/2", "1/4"])
        self.axes["fusion_weights"].set_ylim(0, 1)
        self.axes["fusion_weights"].grid(axis="y", alpha=0.25)

        for key in ["map_overview", "ch0", "ch3", "ch4", "q_masked_global", "q_raw_local", "q_masked_local"]:
            self.markers[f"robot_{key}"], = self.axes[key].plot([], [], "o", color="#00E5FF", markersize=6, markeredgecolor="black")
            self.markers[f"target_{key}"], = self.axes[key].plot([], [], "*", color="#76FF03", markersize=12, markeredgecolor="black")

        self.markers["local_box"], = self.axes["q_masked_global"].plot([], [], color="#00E5FF", lw=1.5)
        self.feature_line, = self.axes["feature_line"].plot([], [], lw=2, color="#4C72B0")
        self.axes["feature_line"].grid(alpha=0.25)
        self.status_box = self.axes["status"].text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            transform=self.axes["status"].transAxes,
            fontsize=10,
            family="monospace",
        )
        self.axes["status"].axis("off")
        for key in ["status_extra_1", "status_extra_2", "status_extra_3"]:
            self.axes[key].axis("off")

        self.paused = False
        self.step_once = False
        self.last_status = ""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.tight_layout()

    def _on_key_press(self, event):
        if event.key == "p":
            self.paused = not self.paused
        elif event.key in {"n", "right", "enter", "space"}:
            self.step_once = True
            self.paused = False
        elif event.key == "q":
            plt.close(self.fig)

    @staticmethod
    def _normalize_for_display(x):
        x = np.asarray(x, dtype=np.float32)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x, dtype=np.float32)
        valid_data = x[finite_mask]
        lo = np.nanpercentile(valid_data, 1)
        hi = np.nanpercentile(valid_data, 99)
        if abs(hi - lo) < 1e-6:
            out = np.zeros_like(x, dtype=np.float32)
            out[finite_mask] = 0.5
            out[~finite_mask] = np.nan
            return out
        out = (np.clip(x, lo, hi) - lo) / (hi - lo)
        out[~finite_mask] = np.nan
        return out

    @staticmethod
    def _crop_around(x, center, radius=45):
        x = np.asarray(x, dtype=np.float32)
        h, w = x.shape
        if center is None:
            cy, cx = h // 2, w // 2
        else:
            cy, cx = int(center[0]), int(center[1])
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        return x[y0:y1, x0:x1], (x0, x1, y0, y1)

    @staticmethod
    def _title_with_range(base, x):
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x)
        if not np.any(finite):
            return f"{base} | no finite values"
        vals = x[finite]
        return f"{base} | min {vals.min():.2g}, max {vals.max():.2g}"

    @staticmethod
    def _overview_from_explored(explored):
        arr = np.asarray(explored, dtype=np.float32)
        overview = np.full(arr.shape, np.nan, dtype=np.float32)
        overview[arr == 0] = 0.85
        overview[arr == 1] = 0.05
        return overview

    def _set_grid_view(self, key, extent):
        x0, x1, y0, y1 = extent
        self.images[key].set_extent((x0, x1, y1, y0))
        self.axes[key].set_xlim(x0, x1)
        self.axes[key].set_ylim(y1, y0)
        self.axes[key].set_aspect("equal", adjustable="box")

    def _set_image(self, key, arr, title, extent, normalize=True):
        display_arr = self._normalize_for_display(arr) if normalize else np.asarray(arr, dtype=np.float32)
        self.images[key].set_data(np.ma.masked_invalid(display_arr))
        self.images[key].set_clim(vmin=0.0, vmax=1.0)
        self._set_grid_view(key, extent)
        self.axes[key].set_title(self._title_with_range(title, arr), fontsize=9)

    def _set_markers(self, key, pos, chosen_coord):
        robot = self.markers.get(f"robot_{key}")
        target = self.markers.get(f"target_{key}")
        if robot is not None:
            robot.set_data([pos[1]], [pos[0]])
        if target is None:
            return
        if chosen_coord is None:
            target.set_data([], [])
        else:
            target.set_data([chosen_coord[1]], [chosen_coord[0]])

    def _update_local_box(self, extent):
        x0, x1, y0, y1 = extent
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        self.markers["local_box"].set_data(xs, ys)

    def update(self, step_num, explored, pos, state_tensor, model_info, chosen_coord):
        state_np = state_tensor.detach().cpu().numpy()
        full_extent = (0, self.map_size, 0, self.map_size)
        crop_center = chosen_coord if chosen_coord is not None else pos
        _, local_extent = self._crop_around(model_info["target_patch"], crop_center, radius=self.local_radius)

        self._set_image("map_overview", self._overview_from_explored(explored), "Map overview: white free, black obstacle, dark unknown", full_extent, normalize=False)
        self._set_image("ch0", state_np[0], "Visibility-aware gain", full_extent, normalize=False)
        self._set_image("ch3", state_np[3], "BFS reachable distance", full_extent, normalize=False)
        self._set_image("ch4", state_np[4], "Valid frontier mask", full_extent, normalize=False)
        self._set_image("q_masked_global", model_info["q_masked_vis"], "Global masked Q over valid frontiers", full_extent)

        local_items = [
            ("q_raw_local", "Raw Q local crop", model_info["target_patch"]),
            ("q_masked_local", "Masked Q local crop", model_info["q_masked_vis"]),
            ("q_full", "Full-scale branch contribution", model_info["q_full"]),
            ("q_half", "Half-scale branch contribution", model_info["q_half"]),
            ("q_quarter", "Quarter-scale branch contribution", model_info["q_quarter"]),
            ("full_patch", "Full-scale feature mean", model_info["full_patch"]),
            ("half_patch", "Half-scale feature mean", model_info["half_patch"]),
            ("quarter_patch", "Quarter-scale upsampled feature mean", model_info["quarter_patch"]),
        ]
        for key, title, raw_arr in local_items:
            display_arr, extent = self._crop_around(raw_arr, crop_center, radius=self.local_radius)
            self._set_image(key, display_arr, title, extent)

        for key in ["map_overview", "ch0", "ch3", "ch4", "q_masked_global", "q_raw_local", "q_masked_local"]:
            self._set_markers(key, pos, chosen_coord)
        self._update_local_box(local_extent)

        for bar, val in zip(self.fusion_bar, model_info["fusion_weights"]):
            bar.set_height(float(val))
        self.axes["fusion_weights"].set_title(
            f"Scale fusion weights | {np.round(model_info['fusion_weights'], 3).tolist()}",
            fontsize=9,
        )

        self.axes["q_hist"].cla()
        q_flat = np.asarray(model_info["q_masked_flat"], dtype=np.float32)
        if q_flat.size > 0:
            self.axes["q_hist"].hist(q_flat, bins=32, color="#4C72B0", alpha=0.85)
            self.axes["q_hist"].axvline(float(np.nanmax(q_flat)), color="#D62728", lw=1.5, label="max")
            lo = np.nanpercentile(q_flat, 1)
            hi = np.nanpercentile(q_flat, 99)
            if hi > lo:
                self.axes["q_hist"].set_xlim(lo, hi)
        self.axes["q_hist"].set_title("Valid-frontier Q histogram", fontsize=9)
        self.axes["q_hist"].grid(alpha=0.2)

        raw_q = np.asarray(model_info["target_patch"], dtype=np.float32)
        profile_y = int(crop_center[0]) if crop_center is not None else raw_q.shape[0] // 2
        profile_y = int(np.clip(profile_y, 0, raw_q.shape[0] - 1))
        x0, x1, _, _ = local_extent
        profile = raw_q[profile_y, x0:x1]
        if profile.size > 0:
            norm_profile = self._normalize_for_display(profile)
            self.feature_line.set_data(np.arange(x0, x1), norm_profile)
            self.axes["feature_line"].set_xlim(x0, max(x0 + 1, x1 - 1))
            self.axes["feature_line"].set_ylim(-0.05, 1.05)
        self.axes["feature_line"].set_title(f"Raw Q profile at row {profile_y}", fontsize=9)

        valid_q = q_flat[np.isfinite(q_flat)]
        q_summary = "no valid frontier"
        if valid_q.size > 0:
            q_summary = f"valid_q min/mean/max: {valid_q.min():.2f}/{valid_q.mean():.2f}/{valid_q.max():.2f}"
        self.last_status = (
            f"step: {step_num}\n"
            f"pos(row,col): {tuple(pos)}\n"
            f"target(row,col): {chosen_coord}\n"
            f"local crop xy: x[{local_extent[0]},{local_extent[1]}), y[{local_extent[2]},{local_extent[3]})\n"
            f"fusion: {np.round(model_info['fusion_weights'], 3).tolist()}\n"
            f"{q_summary}\n"
            f"controls: p pause/resume, n next step, q quit"
        )
        self.status_box.set_text(self.last_status)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        while self.paused and not self.step_once:
            self.fig.canvas.flush_events()
            plt.pause(0.1)

        self.step_once = False
        plt.pause(0.03)

    def close(self):
        plt.close(self.fig)


class InspectableRuntime(RLRuntime):
    def build_model_info(self, state_tensor):
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        _, _, h, w = state_tensor.shape

        full_proj, half_up, quarter_up, fused_feat = self.policy_net.extract_features(state_tensor)

        fusion_weights = self.policy_net.get_display_fusion_weights()
        q_map = self.policy_net.head(fused_feat)
        q_raw = q_map.squeeze(0).squeeze(0)
        q_quarter = self.policy_net.head(quarter_up).squeeze(0).squeeze(0)
        q_masked = q_raw.clone()
        q_masked[state_tensor[0, StateTensorPipeline.FRONTIER_CHANNEL] < 0.5] = -1e9

        return {
            "full_patch": full_proj.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "half_patch": half_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "quarter_patch": quarter_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "q_full": full_proj.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "q_half": half_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "q_quarter": q_quarter.detach().cpu().numpy(),
            "q_masked_vis": q_masked.masked_fill(q_masked <= -1e8, float("nan")).detach().cpu().numpy(),
            "q_masked_flat": q_masked[q_masked > -1e8].detach().cpu().numpy(),
            "target_patch": q_raw.detach().cpu().numpy(),
            "feature_profile": q_raw.flatten().detach().cpu().numpy(),
            "fusion_weights": fusion_weights.detach().cpu().numpy(),
        }


def _resolve_paths(args):
    if args.use_best:
        return str(BEST_CHECKPOINT_PATH), str(BEST_FUSION_WEIGHTS_PATH)
    return args.weights, args.fusion_weights


def evaluate_model(weight_path, fusion_weight_path):
    env = GridWorldEnv(map_size=TEST_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=TEST_SEED, map_generator=MAP_GENERATOR)
    runtime = InspectableRuntime(weight_path, fusion_weight_path, map_size=TEST_MAP_SIZE)
    vis = PolicyInspector(TEST_MAP_SIZE)

    def on_step(step_idx, current_target, next_pos, n_steps, done, info, reached_target, model_info):
        if info["blocked"] and not reached_target:
            print("目标点被打断，释放目标锁")

    try:
        result = runtime.run_locked(env, on_step=on_step, render_hook=vis)
    finally:
        vis.close()

    print(f"评估总结 -> 累计物理步数: {result.total_steps} | 累计探明新区域: {result.total_area:.0f}")
    print("按图窗中的 p 可暂停/继续，n 可单步推进，q 可退出")
    time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect RL policy internals with visualization")
    parser.add_argument("--weights", type=str, default=str(TEST_CHECKPOINT_PATH))
    parser.add_argument("--fusion-weights", type=str, default=str(TEST_FUSION_WEIGHTS_PATH))
    parser.add_argument("--use-best", action="store_true")
    args = parser.parse_args()
    weight_path, fusion_weight_path = _resolve_paths(args)
    evaluate_model(weight_path, fusion_weight_path)