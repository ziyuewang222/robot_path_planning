import numpy as np
import matplotlib.pyplot as plt
from core.a_star import AStarPlanner
from config import SENSOR_RANGE, MAP_GENERATOR, EXPLORATION_SUCCESS_THRESHOLD
from map_generator import build_map, sample_spawn


class GridWorldEnv:
    def __init__(self, map_size=100, sensor_range=SENSOR_RANGE, seed=None, map_generator=None):
        """
        全知室内地图仿真环境 (纯 2D Grid)
        :param map_size: 地图边长
        :param sensor_range: 传感器视距 (栅格)
        :param map_generator: 地图生成器，默认读取 config.MAP_GENERATOR
        """
        self.map_size = map_size
        self.sensor_range = sensor_range
        self.seed = seed
        self.map_generator = map_generator or MAP_GENERATOR
        self.current_room_idx = None
        self.completed_rooms = set()
        self.abandoned_rooms = set()
        self.recent_positions = []
        self.recent_position_window = 12
        self.position_repeat_penalty = 2.5

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
        self.exploration_success_threshold = EXPLORATION_SUCCESS_THRESHOLD

    def reset(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.explored_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)
        self.ground_truth, self.map_meta, _ = build_map(self.map_size, self.map_generator)
        self.rooms = self.map_meta.get("rooms", [])
        self.completed_rooms = set()
        self.abandoned_rooms = set()
        self.recent_positions = []
        self.robot_pos = list(sample_spawn(
            self.ground_truth, self.map_meta, self.map_size, self.map_generator
        ))
        self.current_room_idx = self._locate_room(tuple(self.robot_pos))
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
        self._update_recent_positions(tuple(self.robot_pos))
        return self.explored_map.copy(), self.robot_pos.copy(), collision

    def step_macro(
        self,
        target_coord,
        max_steps=40,
        area_reward_scale=1.0,
        penalty_step=0.0,
        penalty_collision=0.0,
    ):
        robot_pos_before = tuple(self.robot_pos)
        known_before = self.explored_map != -1
        area_before = int(np.sum(known_before))

        path = self.planner.plan(self.explored_map, self.robot_pos, target_coord)
        n_steps = 0
        collision_occurred = False
        path_blocked = False

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

        known_after = self.explored_map != -1
        area_after = int(np.sum(known_after))
        delta_area = float(area_after - area_before)
        newly_known = known_after & ~known_before
        newly_explored_free = int(np.sum((self.explored_map == 0) & newly_known))
        newly_observed_obstacles = int(np.sum((self.explored_map == 1) & newly_known))
        frontier_count_after = self._count_frontier_cells()
        frontier_count_before = self._count_frontier_cells(known_before)
        success = (np.sum(self.explored_map == 0) >= np.sum(self.ground_truth == 0) * self.exploration_success_threshold)
        done = success

        reward = area_reward_scale * delta_area - penalty_step * n_steps
        if collision_occurred or path_blocked:
            reward -= penalty_collision

        self.current_room_idx = self._locate_room(tuple(self.robot_pos))

        info = {
            "delta_area": delta_area,
            "n_steps": n_steps,
            "collision": collision_occurred,
            "blocked": path_blocked,
            "success": success,
            "newly_explored_free": newly_explored_free,
            "newly_observed_obstacles": newly_observed_obstacles,
            "frontier_count_before": frontier_count_before,
            "frontier_count_after": frontier_count_after,
            "robot_pos_before": robot_pos_before,
            "robot_pos_after": tuple(self.robot_pos),
        }

        return self.explored_map.copy(), self.robot_pos.copy(), reward, n_steps, done, info

    def _update_recent_positions(self, pos):
        self.recent_positions.append(pos)
        if len(self.recent_positions) > self.recent_position_window:
            self.recent_positions = self.recent_positions[-self.recent_position_window:]

    def _recent_repeat_penalty(self, pos):
        if not self.recent_positions:
            return 0.0
        repeats = sum(1 for prev in self.recent_positions[:-1] if prev == pos)
        return float(repeats * self.position_repeat_penalty)

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

    def _locate_room(self, pos):
        if not self.rooms:
            return None
        rx, ry = pos
        for idx, room in enumerate(self.rooms):
            x, y, w, h = room['bbox']
            if x + 1 <= rx <= x + w and y + 1 <= ry <= y + h:
                return idx
        return None

    def _count_frontier_cells(self, explored_mask=None):
        explored_mask = self.explored_map != -1 if explored_mask is None else explored_mask
        count = 0
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.explored_map[x, y] != 0:
                    continue
                for dx, dy in self.action_space.values():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if not explored_mask[nx, ny]:
                            count += 1
                            break
        return count

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
