import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from config import MAP_GENERATOR, SENSOR_RANGE, TEST_MAP_SIZE, TRAIN_MAP_SIZE
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline


# -----------------------------------------------------------------------------
# Edit these constants directly, then run:
#   python test_state.py
# -----------------------------------------------------------------------------
MAP_SIZE = TEST_MAP_SIZE if TEST_MAP_SIZE else TRAIN_MAP_SIZE
SEED = 42
STEPS = 20
ACTION_SEQUENCE = [3] * 10 + [1] * 10
SHOW_INTERVAL_SEC = 0.3
PAUSE_SEC_AFTER_DONE = 5


def _normalize_for_display(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)
    if abs(hi - lo) < 1e-6:
        out = np.zeros_like(x, dtype=np.float32)
        out[finite] = 0.5
        out[~finite] = np.nan
        return out
    out = (np.clip(x, lo, hi) - lo) / (hi - lo)
    out[~finite] = np.nan
    return out


def _crop_around(x, center, radius=45):
    x = np.asarray(x, dtype=np.float32)
    h, w = x.shape
    cy, cx = int(center[0]), int(center[1])
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    return x[y0:y1, x0:x1], (x0, x1, y0, y1)


class StateViewer:
    def __init__(self, map_size: int, show_interval_sec: float):
        plt.ion()
        self.map_size = map_size
        self.show_interval_sec = show_interval_sec
        self.fig = plt.figure(figsize=(24, 12))
        self.fig.canvas.manager.set_window_title("State Tensor Visualization")
        gs = self.fig.add_gridspec(2, 4)
        self.axes = [self.fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
        self.images = [None] * 8
        self.titles = [
            "Overview",
            "Ch0: Gain map",
            "Ch1: Free mask",
            "Ch2: Obstacle mask",
            "Ch3: BFS distance",
            "Ch4: Frontier mask",
            "Ch0 local crop",
            "Ch4 local crop",
        ]
        for ax, title in zip(self.axes, self.titles):
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.cmaps = ["gray", "magma", "gray", "gray", "viridis", "gray", "magma", "gray"]

    def update(self, step_idx, explored, pos, state_tensor):
        state = state_tensor.detach().cpu().numpy()
        overview = np.full(explored.shape, np.nan, dtype=np.float32)
        overview[explored == 0] = 0.85
        overview[explored == 1] = 0.05
        overview_disp = np.ma.masked_invalid(overview)

        local_crop_0, _ = _crop_around(state[0], pos, radius=45)
        local_crop_4, _ = _crop_around(state[4], pos, radius=45)

        data_list = [
            overview_disp,
            _normalize_for_display(state[0]),
            state[1],
            state[2],
            state[3],
            state[4],
            _normalize_for_display(local_crop_0),
            local_crop_4,
        ]

        for idx, (ax, data, cmap) in enumerate(zip(self.axes, data_list, self.cmaps)):
            ax.cla()
            ax.set_title(self.titles[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(data, cmap=cmap, origin="upper")

        self.fig.suptitle(f"State Tensor Visualization - Step {step_idx}", fontsize=14)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.show_interval_sec)


def main():
    parser = argparse.ArgumentParser(description="Visualize state tensor channels")
    parser.add_argument("--map-size", type=int, default=MAP_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--show-interval", type=float, default=SHOW_INTERVAL_SEC)
    parser.add_argument("--pause-sec", type=int, default=PAUSE_SEC_AFTER_DONE)
    args = parser.parse_args()

    env = GridWorldEnv(
        map_size=args.map_size,
        sensor_range=SENSOR_RANGE,
        seed=args.seed,
        map_generator=MAP_GENERATOR,
    )
    pipeline = StateTensorPipeline(map_size=args.map_size)
    viewer = StateViewer(args.map_size, args.show_interval)

    explored, pos = env.reset()
    actions = ACTION_SEQUENCE[:args.steps]
    if len(actions) < args.steps:
        actions = actions + [actions[-1]] * (args.steps - len(actions))

    print(f"state tensor type: {type(pipeline.process(explored, pos))}, shape: {pipeline.process(explored, pos).shape}")
    for step_idx, act in enumerate(actions, start=1):
        explored, pos, _ = env.step(act)
        state_tensor = pipeline.process(explored, pos)
        viewer.update(step_idx, explored, pos, state_tensor)

    print("Done. Close the window or wait for auto exit.")
    time.sleep(args.pause_sec)


if __name__ == "__main__":
    main()
