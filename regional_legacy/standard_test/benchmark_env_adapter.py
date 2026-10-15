from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .benchmark_loader import RoomMapCase


@dataclass
class BenchmarkStepInfo:
    coverage_gain: float
    delta_area: float
    collision: bool
    blocked: bool
    done: bool


class BenchmarkEnvAdapter:
    """把 Explore-Bench 房间地图适配成当前策略可用的轻量接口。"""

    def __init__(self, case: RoomMapCase, start_pos: tuple[int, int] | None = None) -> None:
        self.case = case
        self.ground_truth = self._to_occupancy(case.grid)
        self.map_size = int(self.ground_truth.shape[0])
        self.sensor_range = max(1, int(round(7.0 / case.resolution)))
        self.robot_pos = list(start_pos or self._default_start())
        self.explored_map = np.full_like(self.ground_truth, -1, dtype=np.int8)
        self._reveal_local_area()

    @staticmethod
    def _to_occupancy(grid: np.ndarray) -> np.ndarray:
        occupancy = np.full(grid.shape, 1, dtype=np.int8)
        occupancy[grid > 240] = 0
        occupancy[grid < 50] = 1
        occupancy[(grid >= 50) & (grid <= 240)] = 1
        return occupancy

    def _default_start(self) -> tuple[int, int]:
        free = np.argwhere(self.ground_truth == 0)
        if free.size == 0:
            return (0, 0)
        mid = free[len(free) // 2]
        return int(mid[0]), int(mid[1])

    def reset(self, start_pos: Optional[tuple[int, int]] = None):
        self.explored_map = np.full_like(self.ground_truth, -1, dtype=np.int8)
        self.robot_pos = list(start_pos or self._default_start())
        self._reveal_local_area()
        return self.explored_map.copy(), self.robot_pos.copy()

    def _reveal_local_area(self) -> None:
        rx, ry = self.robot_pos
        radius = self.sensor_range
        x0 = max(0, rx - radius)
        x1 = min(self.map_size, rx + radius + 1)
        y0 = max(0, ry - radius)
        y1 = min(self.map_size, ry + radius + 1)
        for x in range(x0, x1):
            for y in range(y0, y1):
                if abs(x - rx) + abs(y - ry) <= radius:
                    self.explored_map[x, y] = self.ground_truth[x, y]

    def step_macro(self, target_coord):
        target = (int(target_coord[0]), int(target_coord[1]))
        before = int(np.sum(self.explored_map != -1))
        collision = self.ground_truth[target] == 1
        blocked = False
        if not collision:
            self.robot_pos = [target[0], target[1]]
            self._reveal_local_area()
        else:
            blocked = True
        after = int(np.sum(self.explored_map != -1))
        delta_area = float(after - before)
        done = bool(np.sum(self.explored_map == 0) >= np.sum(self.ground_truth == 0) * 0.995)
        info = {
            "coverage_gain": delta_area,
            "delta_area": delta_area,
            "collision": collision,
            "blocked": blocked,
            "done": done,
        }
        return self.explored_map.copy(), self.robot_pos.copy(), 0.0, 1, done, info
