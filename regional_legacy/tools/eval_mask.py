"""regional_legacy 地图标签可视化脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAP_SIZE, MAP_GENERATOR, TEST_SEED
from map_generator import build_map, sample_spawn
from parallel_worker import extract_perfect_labels


VISUALIZE_MAP_SIZE = DEFAULT_MAP_SIZE
VISUALIZE_SEED = TEST_SEED


def show_current_map(map_size: int, seed: int | None = None) -> None:
    if seed is not None:
        np.random.seed(seed)

    ground_truth, map_meta, generator_name = build_map(map_size, MAP_GENERATOR)
    robot_pos = sample_spawn(ground_truth, map_meta, map_size, generator_name)
    semantic_mask = extract_perfect_labels(ground_truth, map_meta, generator_name)

    cmap = ListedColormap([
        "#111111",
        "#3498db",
        "#2ecc71",
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    _, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"Label Mask | generator={generator_name} | size={map_size}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(semantic_mask, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    ax.scatter([robot_pos[1]], [robot_pos[0]], c="#e74c3c", s=24, label="spawn")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="obstacle (0)", markerfacecolor="#111111", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="corridor (1)", markerfacecolor="#3498db", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="room (2)", markerfacecolor="#2ecc71", markersize=10),
    ]
    ax.legend(handles=handles + [plt.Line2D([0], [0], marker="o", color="w", label="spawn", markerfacecolor="#e74c3c", markersize=8)], loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_current_map(VISUALIZE_MAP_SIZE, VISUALIZE_SEED)
