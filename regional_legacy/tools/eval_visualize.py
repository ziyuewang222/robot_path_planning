"""regional_legacy 的语义地图可视化评估脚本。

输入观测地图，预测拓扑语义图并可视化对比。运行参数统一由
`config.py` 与本文件顶部常量决定；不接受终端参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAP_SIZE, MAP_GENERATOR, SENSOR_RANGE, TEST_SEED, resolve_test_checkpoint_path
from core.models import SemanticUNet
from env.grid_env import GridWorldEnv


DEFAULT_CHECKPOINT_PATH = resolve_test_checkpoint_path()
VISUALIZE_MAP_SIZE = DEFAULT_MAP_SIZE
VISUALIZE_SEED = TEST_SEED
INPUT_CMAP = ListedColormap(["#3b3b3b", "#f4d03f", "#9aa0a6"])
INPUT_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], INPUT_CMAP.N)
SEMANTIC_CMAP = ListedColormap(["#202020", "#2e86de", "#f4d03f"])
SEMANTIC_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], SEMANTIC_CMAP.N)


def build_input_tensor(explored_map: np.ndarray) -> torch.Tensor:
    """把观测地图转成模型输入。
    直接保留 -1（迷雾）, 0（空地）, 1（障碍物）的物理特征
    """
    x = explored_map.astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


def build_semantic_label(ground_truth: np.ndarray, rooms) -> np.ndarray:
    sem = np.zeros_like(ground_truth, dtype=np.int64)
    sem[ground_truth == 0] = 1
    for room in rooms:
        x, y, w, h = room["bbox"]
        patch_gt = ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h]
        patch_sem = sem[x + 1:x + 1 + w, y + 1:y + 1 + h]
        patch_sem[patch_gt == 0] = 2
        sem[x + 1:x + 1 + w, y + 1:y + 1 + h] = patch_sem
    return sem


def _is_frontier_cell(explored_map: np.ndarray, x: int, y: int) -> bool:
    if explored_map[x, y] != 0:
        return False
    h, w = explored_map.shape
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w and explored_map[nx, ny] == -1:
            return True
    return False


def visualize_prediction(model_path: Path, map_size: int, seed: int | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE, seed=seed, map_generator=MAP_GENERATOR)
    explored_map, robot_pos = env.reset()

    model = SemanticUNet(in_channels=1, num_classes=3).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis("off")

    step_idx = 0
    while True:
        with torch.no_grad():
            input_tensor = build_input_tensor(explored_map).to(device)
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        gt_semantic = build_semantic_label(env.ground_truth, env.rooms)

        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[0].set_title(f"Observed Input | step={step_idx}")
        axes[0].imshow(explored_map, cmap=INPUT_CMAP, norm=INPUT_NORM, interpolation="nearest")
        axes[0].scatter([robot_pos[1]], [robot_pos[0]], c="#e74c3c", s=18)

        axes[1].set_title("Predicted Semantic Map")
        axes[1].imshow(pred, cmap=SEMANTIC_CMAP, norm=SEMANTIC_NORM, interpolation="nearest")

        axes[2].set_title("Ground Truth Semantic Map")
        axes[2].imshow(gt_semantic, cmap=SEMANTIC_CMAP, norm=SEMANTIC_NORM, interpolation="nearest")

        plt.tight_layout()
        plt.pause(0.05)

        target = None
        queue = [(robot_pos[0], robot_pos[1])]
        visited = {tuple(robot_pos)}
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) != tuple(robot_pos) and _is_frontier_cell(explored_map, cx, cy):
                target = [cx, cy]
                break
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < map_size and 0 <= ny < map_size and (nx, ny) not in visited and explored_map[nx, ny] == 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        if target is None:
            break

        explored_map, robot_pos, _reward, _n_steps, done, _info = env.step_macro(target)
        step_idx += 1
        if done:
            break

    plt.show()


if __name__ == "__main__":
    visualize_prediction(DEFAULT_CHECKPOINT_PATH, VISUALIZE_MAP_SIZE, VISUALIZE_SEED)

    visualize_prediction(Path(args.checkpoint), args.map_size, args.seed)