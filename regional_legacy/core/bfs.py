from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np


def find_nearest_frontier(frontier_mask: np.ndarray, robot_pos: Sequence[int]) -> Optional[Tuple[int, int]]:
    """基于 BFS 可达性距离，返回离小车最近的 frontier 坐标。"""
    if frontier_mask.ndim != 2:
        raise ValueError("frontier_mask 必须是二维地图")

    start = (int(robot_pos[0]), int(robot_pos[1]))
    h, w = frontier_mask.shape
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
                visited.add((nx, ny))
                queue.append((nx, ny))

    return None
