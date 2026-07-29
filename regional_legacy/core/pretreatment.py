from __future__ import annotations

import numpy as np


def build_frontier_mask(explored_map: np.ndarray) -> np.ndarray:
    """把 0/1/-1 三值地图转换成 frontier 动作掩码。

    使用 NumPy 向量化邻域判断：frontier 为与未知区域四邻接的已知空地。
    """
    if explored_map.ndim != 2:
        raise ValueError("explored_map 必须是二维栅格地图")

    free_space = explored_map == 0
    fog = explored_map == -1
    has_fog_adjacent = np.zeros_like(explored_map, dtype=bool)

    # 向量化平移检测四个方向
    has_fog_adjacent[:-1, :] |= fog[1:, :]
    has_fog_adjacent[1:, :] |= fog[:-1, :]
    has_fog_adjacent[:, :-1] |= fog[:, 1:]
    has_fog_adjacent[:, 1:] |= fog[:, :-1]

    return (free_space & has_fog_adjacent).astype(np.uint8)