"""
使用方法

1. 默认使用 `config.py` 中的 latest 权重进行评估：
   python eval_policy.py

2. 指定主干权重和融合权重：
   python eval_policy.py --weights /path/to/checkpoint.pth --fusion-weights /path/to/fusion.pth

3. 只指定主干权重，融合权重仍默认使用 latest：
   python eval_policy.py --weights /path/to/checkpoint.pth

4. 指定使用 best：
   python eval_policy.py --use-best
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    BEST_CHECKPOINT_PATH,
    BEST_FUSION_WEIGHTS_PATH,
    LATEST_CHECKPOINT_PATH,
    LATEST_FUSION_WEIGHTS_PATH,
    MAP_GENERATOR,
    SENSOR_RANGE,
    TEST_MAP_SIZE,
    TEST_SEED,
)
from grid_env import GridWorldEnv
from rl_runtime import RLRuntime


class PolicyInspector:
    def __init__(self, map_size: int):
        plt.ion()
        self.fig = plt.figure(figsize=(24, 15))
        self.fig.canvas.manager.set_window_title("Spatial FCN RL - Policy Inspector")
        gs = self.fig.add_gridspec(4, 4, height_ratios=[1.15, 1.05, 1.0, 0.9])
        layout = [
            "main", "ch0", "ch1", "ch2",
            "full_patch", "half_patch", "quarter_patch", "fusion_weights",
            "q_full", "q_half", "q_quarter", "q_masked",
            "patch_zoom", "feature_line", "q_hist", "reserved",
        ]
        self.axes = {k: self.fig.add_subplot(gs[i // 4, i % 4]) for i, k in enumerate(layout)}

        titles = {
            "main": "Main exploration view",
            "ch0": "State ch0: explored map",
            "ch1": "State ch1: robot gaussian",
            "ch2": "State ch2: frontier mask",
            "full_patch": "Full-res patch",
            "half_patch": "Half-res patch",
            "quarter_patch": "Quarter-res patch",
            "fusion_weights": "Fusion weights",
            "q_full": "Full branch Q map",
            "q_half": "Half branch Q map",
            "q_quarter": "Quarter branch Q map",
            "q_masked": "Final masked Q map",
            "patch_zoom": "Target-centered patch",
            "feature_line": "Feature profile",
            "q_hist": "Masked Q histogram",
            "reserved": "Status / help",
        }
        for name, ax in self.axes.items():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[name], fontsize=11)

        self.images = {}
        self.stars = {}
        self.images["main"] = self.axes["main"].imshow(np.zeros((map_size, map_size)), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
        self.robot, = self.axes["main"].plot([], [], "ro", markersize=7)
        self.stars["main"], = self.axes["main"].plot([], [], "r*", markersize=12)

        for key, cmap, vmin, vmax in [("ch0", "viridis", 0.0, 1.0), ("ch1", "magma", 0.0, 1.0), ("ch2", "gray", 0.0, 1.0)]:
            self.images[key] = self.axes[key].imshow(np.zeros((map_size, map_size)), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        for key in ["full_patch", "half_patch", "quarter_patch", "patch_zoom"]:
            self.images[key] = self.axes[key].imshow(np.zeros((11, 11)), cmap="plasma" if key != "patch_zoom" else "viridis", origin="upper")

        self.fusion_bar = self.axes["fusion_weights"].bar([0, 1, 2], [1 / 3, 1 / 3, 1 / 3], color=["#4C72B0", "#55A868", "#C44E52"])
        self.axes["fusion_weights"].set_xticks([0, 1, 2])
        self.axes["fusion_weights"].set_xticklabels(["1x", "1/2", "1/4"])
        self.axes["fusion_weights"].set_ylim(0, 1)

        q_cmap = plt.cm.plasma.copy()
        q_cmap.set_bad(color="#d9d9d9")
        for key in ["q_full", "q_half", "q_quarter", "q_masked"]:
            self.images[key] = self.axes[key].imshow(np.zeros((map_size, map_size)), cmap=q_cmap, origin="upper", interpolation="nearest")
        self.stars["q_masked"], = self.axes["q_masked"].plot([], [], "g*", markersize=11)
        self.feature_line, = self.axes["feature_line"].plot([], [], lw=2)
        self.axes["q_hist"].set_xlim(0, 1)
        self.status_box = self.axes["reserved"].text(
            0.02,
            0.98,
            "",
            va="top",
            ha="left",
            transform=self.axes["reserved"].transAxes,
            fontsize=11,
            family="monospace",
        )
        self.axes["reserved"].axis("off")

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
    def _display_grid(explored):
        display = np.zeros_like(explored, dtype=np.float32)
        display[explored == -1] = 0.5
        display[explored == 0] = 1.0
        display[explored == 1] = 0.0
        return display

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
            return np.clip(x, lo - 1e-6, hi + 1e-6)
            
        clipped = np.clip(x, lo, hi)
        return (clipped - lo) / (hi - lo)

    @staticmethod
    def _crop_around(x, center, radius=35):
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

    def update(self, step_num, explored, pos, state_tensor, model_info, chosen_coord):
        self.images["main"].set_data(self._display_grid(explored))
        self.images["ch0"].set_data(state_tensor[0].cpu().numpy())
        self.images["ch1"].set_data(state_tensor[1].cpu().numpy())
        self.images["ch2"].set_data(state_tensor[2].cpu().numpy())

        crop_center = chosen_coord if chosen_coord is not None else pos
        display_items = [
            ("full_patch", "Full-res local feature", model_info["full_patch"]),
            ("half_patch", "Half-res local feature", model_info["half_patch"]),
            ("quarter_patch", "Quarter-res local feature", model_info["quarter_patch"]),
            ("q_full", "Full branch local Q", model_info["q_full"]),
            ("q_half", "Half branch local Q", model_info["q_half"]),
            ("q_quarter", "Quarter branch local Q", model_info["q_quarter"]),
            ("q_masked", "Final masked local Q", model_info["q_masked_vis"]),
            ("patch_zoom", "Raw Q local zoom", model_info["target_patch"]),
        ]
        for key, title, raw_arr in display_items:
            local_arr, extent = self._crop_around(raw_arr, crop_center, radius=45)
            norm_arr = self._normalize_for_display(local_arr)
            masked_arr = np.ma.masked_invalid(norm_arr)
            
            self.images[key].set_data(masked_arr)
            self.images[key].set_extent(extent)
            
            vmin = np.nanmin(masked_arr) if masked_arr.count() > 0 else 0.0
            vmax = np.nanmax(masked_arr) if masked_arr.count() > 0 else 1.0
            if vmin == vmax:
                vmax += 1e-5
            self.images[key].set_clim(vmin=vmin, vmax=vmax)
            
            self.axes[key].set_xlim(extent[0], extent[1])
            self.axes[key].set_ylim(extent[3], extent[2])
            self.axes[key].set_title(self._title_with_range(title, local_arr), fontsize=9)

        for bar, val in zip(self.fusion_bar, model_info["fusion_weights"]):
            bar.set_height(float(val))

        self.robot.set_data([pos[1]], [pos[0]])
        if chosen_coord is not None:
            self.stars["main"].set_data([chosen_coord[1]], [chosen_coord[0]])
            self.stars["q_masked"].set_data([chosen_coord[1]], [chosen_coord[0]])
            self.stars["q_masked"].set_markersize(14)
        else:
            self.stars["main"].set_data([], [])
            self.stars["q_masked"].set_data([], [])

        fp = np.asarray(model_info["feature_profile"], dtype=np.float32)
        if fp.size > 0:
            fp = (fp - np.min(fp)) / (np.max(fp) - np.min(fp) + 1e-8)
        self.feature_line.set_data(np.arange(len(fp)), fp)
        self.axes["feature_line"].set_xlim(0, max(1, len(fp) - 1))
        self.axes["feature_line"].set_ylim(0, 1.05)

        self.axes["q_hist"].cla()
        q_flat = np.asarray(model_info["q_masked_flat"], dtype=np.float32)
        if q_flat.size > 0:
            self.axes["q_hist"].hist(q_flat, bins=30, color="#4C72B0", alpha=0.85)
        self.axes["q_hist"].set_xlim(np.nanpercentile(q_flat, 2) if q_flat.size > 0 else 0, np.nanpercentile(q_flat, 98) if q_flat.size > 0 else 1)

        self.last_status = (
            f"step: {step_num}\n"
            f"pos: {tuple(pos)}\n"
            f"target: {chosen_coord}\n"
            f"fusion: {np.round(model_info['fusion_weights'], 3).tolist()}\n"
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

        full_feat = self.policy_net.full_res_branch(state_tensor)
        full_proj = self.policy_net.fuse_proj_full(full_feat)

        half_x = F.avg_pool2d(state_tensor, kernel_size=2, stride=2)
        half_feat = self.policy_net.half_res_branch(half_x)
        half_proj = self.policy_net.fuse_proj_half(half_feat)
        half_up = F.interpolate(half_proj, size=(h, w), mode="bilinear", align_corners=False)

        quarter_x = F.avg_pool2d(state_tensor, kernel_size=4, stride=4)
        quarter_feat = self.policy_net.quarter_res_branch(quarter_x)
        quarter_proj = self.policy_net.fuse_proj_quarter(quarter_feat)
        quarter_up = F.interpolate(quarter_proj, size=(h, w), mode="bilinear", align_corners=False)

        fusion_weights = self.policy_net.get_fusion_weights()
        q_map = self.policy_net.head(
            fusion_weights[0] * full_proj + fusion_weights[1] * half_up + fusion_weights[2] * quarter_up
        )
        q_raw = q_map.squeeze(0).squeeze(0)
        q_masked = q_raw.clone()
        q_masked[state_tensor[0, 2] < 0.5] = -1e9

        return {
            "full_patch": full_proj.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "half_patch": half_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "quarter_patch": quarter_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "q_full": full_proj.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "q_half": half_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
            "q_quarter": quarter_up.mean(dim=1).squeeze(0).detach().cpu().numpy(),
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
    parser.add_argument("--weights", type=str, default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--fusion-weights", type=str, default=str(LATEST_FUSION_WEIGHTS_PATH))
    parser.add_argument("--use-best", action="store_true")
    args = parser.parse_args()
    weight_path, fusion_weight_path = _resolve_paths(args)
    evaluate_model(weight_path, fusion_weight_path)