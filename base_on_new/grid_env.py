import numpy as np
import matplotlib.pyplot as plt
from local_planner import AStarPlanner
from config import (
    SENSOR_RANGE,
    MAP_GENERATOR,
    AREA_REWARD_SCALE,
    STEP_PENALTY,
    COLLISION_PENALTY,
)
from map_generator import build_map, sample_spawn


class GridWorldEnv:
    def __init__(self, map_size=100, sensor_range=SENSOR_RANGE, seed=None, map_generator=None):
        """
        全知室内地图仿真环境 (纯 2D Grid)
        :param map_size: 地图边长
        :param sensor_range: 传感器视距 (栅格)
        :param map_generator: 地图生成器，"room" 或 "random"；默认读取 config.MAP_GENERATOR
        """
        self.map_size = map_size
        self.sensor_range = sensor_range
        self.seed = seed
        self.map_generator = map_generator or MAP_GENERATOR

        self.action_space = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        self.ground_truth = None
        self.explored_map = None
        self.robot_pos = None
        self.rooms = []
        self.map_meta = {}

        self.fig = None
        self.ax = None
        self.img_plot = None
        self.robot_plot = None

        self.planner = AStarPlanner()

    def reset(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.explored_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)
        self.ground_truth, self.map_meta, _ = build_map(self.map_size, self.map_generator)
        self.rooms = self.map_meta.get("rooms", [])
        self.robot_pos = list(sample_spawn(
            self.ground_truth, self.map_meta, self.map_size, self.map_generator
        ))
        self._update_fov_bresenham()

        return self.explored_map.copy(), self.robot_pos.copy()

    def step(self, action):
        dx, dy = self.action_space[action]
        next_x = self.robot_pos[0] + dx
        next_y = self.robot_pos[1] + dy

        collision = False
        if 0 <= next_x < self.map_size and 0 <= next_y < self.map_size:
            if self.ground_truth[next_x, next_y] == 0:
                self.robot_pos = [next_x, next_y]
            else:
                collision = True
        else:
            collision = True

        self._update_fov_bresenham()
        return self.explored_map.copy(), self.robot_pos.copy(), collision

    def step_macro(
        self,
        target_coord,
        max_steps=40,
        area_reward_scale=AREA_REWARD_SCALE,
        penalty_step=STEP_PENALTY,
        penalty_collision=COLLISION_PENALTY,
    ):
        area_before = np.sum(self.explored_map != -1)

        n_steps = 0
        collision_occurred = False
        path_blocked = False

        path = self.planner.plan(self.explored_map, self.robot_pos, target_coord)

        if path is None or len(path) <= 1:
            path_blocked = True
            n_steps = 1
        else:
            for next_node in path[1:]:
                if n_steps >= max_steps:
                    break

                dx = next_node[0] - self.robot_pos[0]
                dy = next_node[1] - self.robot_pos[1]

                act = -1
                for k, v in self.action_space.items():
                    if v == (dx, dy):
                        act = k
                        break

                _, _, collision = self.step(act)
                n_steps += 1

                if collision:
                    collision_occurred = True
                    break

                if self.explored_map[target_coord[0], target_coord[1]] == 1:
                    path_blocked = True
                    break

        area_after = np.sum(self.explored_map != -1)
        delta_area = float(area_after - area_before)

        macro_reward = area_reward_scale * delta_area - penalty_step * n_steps
        if collision_occurred or path_blocked:
            macro_reward -= penalty_collision

        total_free_space = np.sum(self.ground_truth == 0)
        explored_free_space = np.sum(self.explored_map == 0)
        done = (explored_free_space >= total_free_space * 0.95)

        info = {
            'delta_area': delta_area,
            'n_steps': n_steps,
            'collision': collision_occurred,
            'blocked': path_blocked,
        }

        return self.explored_map.copy(), self.robot_pos.copy(), macro_reward, n_steps, done, info

    def render(self, step_num=0):
        display_grid = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        display_grid[self.explored_map == -1] = 0.5
        display_grid[self.explored_map == 0] = 1.0
        display_grid[self.explored_map == 1] = 0.0

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.set_title("Autonomous Exploration Simulation")
            self.img_plot = self.ax.imshow(display_grid, cmap='viridis', origin='upper', vmin=0.0, vmax=1.0)
            self.robot_plot, = self.ax.plot([], [], 'ro', markersize=8, label='Robot')
            self.ax.legend(loc='upper right')

        self.img_plot.set_data(display_grid)
        self.robot_plot.set_data([self.robot_pos[1]], [self.robot_pos[0]])
        self.ax.set_title(
            f"Step: {step_num} | Generator: {self.map_generator} | "
            f"Robot Pos: ({self.robot_pos[0]}, {self.robot_pos[1]})"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def _update_fov_bresenham(self):
        rx, ry = self.robot_pos
        R = self.sensor_range
        x_min = max(0, rx - R)
        x_max = min(self.map_size - 1, rx + R)
        y_min = max(0, ry - R)
        y_max = min(self.map_size - 1, ry + R)

        for tx in range(x_min, x_max + 1):
            for ty in range(y_min, y_max + 1):
                if (tx - rx)**2 + (ty - ry)**2 > R**2:
                    continue
                line = self._bresenham_line(rx, ry, tx, ty)
                for (lx, ly) in line:
                    self.explored_map[lx, ly] = self.ground_truth[lx, ly]
                    if self.ground_truth[lx, ly] == 1:
                        break

    def _bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points
