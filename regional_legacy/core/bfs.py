from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np


def find_nearest_frontier(
    explored_map: np.ndarray,
    frontier_mask: np.ndarray,
    robot_pos: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """基于 BFS 可达性距离，返回离小车最近的 frontier 坐标。"""
    if frontier_mask.ndim != 2 or explored_map.ndim != 2:
        raise ValueError("地图和 frontier_mask 必须是二维数组")

    start = (int(robot_pos[0]), int(robot_pos[1]))
    h, w = frontier_mask.shape
    if explored_map.shape != frontier_mask.shape:
        raise ValueError("explored_map 和 frontier_mask 的形状必须一致")
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        raise ValueError("robot_pos 超出地图范围")

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()
        if frontier_mask[x, y] == 1:
            return (x, y)

        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in visited:
                if explored_map[nx, ny] == 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return None
