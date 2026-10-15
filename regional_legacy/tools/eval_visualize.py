"""regional_legacy 地图静态可视化脚本。

作用：直接生成当前 `MAP_GENERATOR` 对应的一张地图并展示，不做模型推理，
也不执行任何逐步探索过程。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MAP_SIZE, MAP_GENERATOR, TEST_SEED
from map_generator import build_map, sample_spawn


VISUALIZE_MAP_SIZE = DEFAULT_MAP_SIZE
VISUALIZE_SEED = TEST_SEED


def _to_display(ground_truth: np.ndarray) -> np.ndarray:
    """将完整地图转换为显示图：空地为白色，障碍物为黑色。"""
    display = np.ones_like(ground_truth, dtype=np.float32)
    display[ground_truth == 1] = 0.0
    return display


def show_current_map(map_size: int, seed: int | None = None) -> None:
    if seed is not None:
        np.random.seed(seed)

    ground_truth, map_meta, generator_name = build_map(map_size, MAP_GENERATOR)
    robot_pos = sample_spawn(ground_truth, map_meta, map_size, generator_name)

    _, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"Complete Map | generator={generator_name} | size={map_size}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(_to_display(ground_truth), cmap="gray", origin="upper", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.scatter([robot_pos[1]], [robot_pos[0]], c="#e74c3c", s=24, label="spawn")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_current_map(VISUALIZE_MAP_SIZE, VISUALIZE_SEED)
