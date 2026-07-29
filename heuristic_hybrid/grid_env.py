import numpy as np
import matplotlib.pyplot as plt
from local_planner import AStarPlanner
from config import (
    SENSOR_RANGE,
    MAP_GENERATOR,
    AREA_REWARD_SCALE,
)
from map_generator import build_map, sample_spawn


class GridWorldEnv:
    def __init__(self, map_size=100, sensor_range=SENSOR_RANGE, seed=None, map_generator=None):
        self.map_size = map_size
        self.sensor_range = sensor_range
        self.seed = seed
        self.map_generator = map_generator or MAP_GENERATOR
        self.action_space = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.ground_truth = None
        self.explored_map = None
        self.robot_pos = None
        self.planner = AStarPlanner()

    def reset(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.explored_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)
        self.ground_truth, self.map_meta, _ = build_map(self.map_size, self.map_generator)
        self.robot_pos = list(sample_spawn(self.ground_truth, self.map_meta, self.map_size, self.map_generator))
        self._update_fov_bresenham()
        return self.explored_map.copy(), self.robot_pos.copy()

    def step(self, action):
        dx, dy = self.action_space[action]
        nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
        collision = not (0 <= nx < self.map_size and 0 <= ny < self.map_size and self.ground_truth[nx, ny] == 0)
        if not collision:
            self.robot_pos = [nx, ny]
        self._update_fov_bresenham()
        return self.explored_map.copy(), self.robot_pos.copy(), collision

    def extract_heuristic_matrix(self, frontier_list, max_distance=None, max_efficiency_scale=None, local_clear_radius=2.0):
        frontier_list = list(frontier_list or [])
        if len(frontier_list) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        max_distance = float(max_distance if max_distance is not None else max(1.0, self.map_size * 1.5))
        max_efficiency_scale = float(max_efficiency_scale if max_efficiency_scale is not None else max(1.0, self.sensor_range * 10.0))
        eps = 1e-6
        out = []
        heading = np.array([1.0, 0.0], dtype=np.float32)
        for fx, fy in frontier_list:
            path = self.planner.plan(self.explored_map, list(self.robot_pos), [int(fx), int(fy)])
            dist = max_distance if path is None else float(max(0, len(path) - 1))
            h_dist = max(0.0, 1.0 - (dist / max_distance))
            visible_unknown = float(self._frontier_lightweight_gain((int(fx), int(fy)), radius=self.sensor_range))
            h_eff = min(1.0, (visible_unknown / (dist + eps)) / max_efficiency_scale)
            vec = np.array([fx - self.robot_pos[0], fy - self.robot_pos[1]], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            cos_delta = 1.0 if norm <= eps else float(np.dot(heading, vec / norm))
            h_cons = (cos_delta + 1.0) / 2.0
            h_clear = 1.0 if dist < local_clear_radius else 0.0
            out.append([h_dist, h_eff, h_cons, h_clear])
        return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)

    def step_macro(self, target_coord, max_steps=40, alpha=AREA_REWARD_SCALE, time_penalty=-0.5):
        known_before = self.explored_map != -1
        path = self.planner.plan(self.explored_map, self.robot_pos, target_coord)
        n_steps = 0
        collision = False
        if path is not None:
            for nxt in path[1:max_steps+1]:
                dx, dy = nxt[0] - self.robot_pos[0], nxt[1] - self.robot_pos[1]
                act = next(k for k, v in self.action_space.items() if v == (dx, dy))
                _, _, collision = self.step(act)
                n_steps += 1
                if collision:
                    break
        coverage_gain = float(np.sum(self.explored_map != -1) - np.sum(known_before))
        reward = float(alpha * coverage_gain + time_penalty)
        done = bool(np.sum(self.explored_map == 0) >= np.sum(self.ground_truth == 0) * 0.995)
        info = {"coverage_gain": coverage_gain, "n_steps": n_steps, "collision": collision}
        return self.explored_map.copy(), self.robot_pos.copy(), reward, n_steps, done, info

    def _frontier_lightweight_gain(self, pos, radius=None):
        return 0.0

    def _update_fov_bresenham(self):
        rx, ry = self.robot_pos
        R = self.sensor_range
        for tx in range(max(0, rx-R), min(self.map_size-1, rx+R)+1):
            for ty in range(max(0, ry-R), min(self.map_size-1, ry+R)+1):
                if (tx-rx)**2 + (ty-ry)**2 > R**2:
                    continue
                for lx, ly in self._bresenham_line(rx, ry, tx, ty):
                    self.explored_map[lx, ly] = self.ground_truth[lx, ly]
                    if self.ground_truth[lx, ly] == 1:
                        break

    def _bresenham_line(self, x0, y0, x1, y1):
        pts=[]; dx=abs(x1-x0); dy=abs(y1-y0); sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1; err=dx-dy
        while True:
            pts.append((x0,y0))
            if x0==x1 and y0==y1: break
            e2=2*err
            if e2>-dy: err-=dy; x0+=sx
            if e2<dx: err+=dx; y0+=sy
        return pts
