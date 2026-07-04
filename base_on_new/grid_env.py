import numpy as np
import matplotlib.pyplot as plt
from local_planner import AStarPlanner
from config import (
    SENSOR_RANGE,
    CORRIDOR_WIDTH,
    ROOM_MIN_SIZE,
    ROOM_MAX_SIZE,
    ROOM_PADDING,
    ROOM_PLACE_ATTEMPTS,
    OBS_MIN_SIZE,
    OBS_MAX_SIZE,
    OBSTACLES_PER_ROOM_MIN,
    OBSTACLES_PER_ROOM_MAX,
    OBSTACLE_PLACE_ATTEMPTS,
    OBSTACLE_WALL_MARGIN,
    SPAWN_MARGIN,
    rooms_for,
)


class GridWorldEnv:
    def __init__(self, map_size=100, sensor_range=SENSOR_RANGE, seed=None):
        """
        全知室内地图仿真环境 (纯 2D Grid)
        :param map_size: 地图边长
        :param sensor_range: 传感器视距 (栅格)
        """
        self.map_size = map_size
        self.sensor_range = sensor_range
        self.seed = seed

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

        self.fig = None
        self.ax = None
        self.img_plot = None
        self.robot_plot = None

        self.planner = AStarPlanner()

    def reset(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.ground_truth = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.explored_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)

        self._build_structured_map()
        self.robot_pos = list(self._get_valid_spawn_pos())
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

    def step_macro(self, target_coord, max_steps=40, penalty_step=0.1, penalty_collision=10.0):
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

        macro_reward = delta_area - (penalty_step * n_steps)
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
        self.ax.set_title(f"Step: {step_num} | Robot Pos: ({self.robot_pos[0]}, {self.robot_pos[1]})")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def _build_structured_map(self):
        """程序化生成房间-走廊地图：地图变大时房间变多，尺寸保持不变。"""
        S = self.map_size
        self.ground_truth[:] = 1

        self.rooms = self._generate_rooms(S)
        for room in self.rooms:
            self._carve_room(room)
        self._connect_rooms(self.rooms)
        for room in self.rooms:
            self._place_room_obstacles(room)

    def _generate_rooms(self, map_size):
        target = rooms_for(map_size)
        rooms = []
        for _ in range(target):
            for _ in range(ROOM_PLACE_ATTEMPTS):
                w = np.random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE + 1)
                h = np.random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE + 1)
                x_max = map_size - w - 3
                y_max = map_size - h - 3
                if x_max < 1 or y_max < 1:
                    break

                x = np.random.randint(1, x_max + 1)
                y = np.random.randint(1, y_max + 1)
                room = (x, y, w, h)
                if not self._room_overlaps(room, rooms):
                    rooms.append(room)
                    break
        return rooms

    def _room_bbox(self, room):
        x, y, w, h = room
        return x, y, x + w + 2, y + h + 2

    def _room_overlaps(self, room, rooms):
        ax1, ay1, ax2, ay2 = self._room_bbox(room)
        pad = ROOM_PADDING
        for other in rooms:
            bx1, by1, bx2, by2 = self._room_bbox(other)
            if not (ax2 + pad <= bx1 or bx2 + pad <= ax1 or ay2 + pad <= by1 or by2 + pad <= ay1):
                return True
        return False

    def _carve_room(self, room):
        x, y, w, h = room
        self.ground_truth[x + 1:x + 1 + w, y + 1:y + 1 + h] = 0

    def _room_center(self, room):
        x, y, w, h = room
        return x + 1 + w // 2, y + 1 + h // 2

    def _connect_rooms(self, rooms):
        if len(rooms) < 2:
            return

        centers = [self._room_center(room) for room in rooms]
        in_tree = {0}
        while len(in_tree) < len(rooms):
            best_dist = None
            best_pair = None
            for i in in_tree:
                for j in range(len(rooms)):
                    if j in in_tree:
                        continue
                    dist = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_pair = (i, j)
            i, j = best_pair
            self._carve_corridor(centers[i], centers[j])
            in_tree.add(j)

    def _carve_corridor(self, start, end):
        x1, y1 = start
        x2, y2 = end
        half = CORRIDOR_WIDTH // 2
        if np.random.rand() < 0.5:
            self._carve_horizontal_corridor(x1, x2, y1, half)
            self._carve_vertical_corridor(y1, y2, x2, half)
        else:
            self._carve_vertical_corridor(y1, y2, x1, half)
            self._carve_horizontal_corridor(x1, x2, y2, half)

    def _carve_horizontal_corridor(self, x1, x2, y, half):
        xa, xb = sorted((x1, x2))
        y0 = max(1, y - half)
        y1 = min(self.map_size - 1, y - half + CORRIDOR_WIDTH)
        self.ground_truth[xa:xb + 1, y0:y1] = 0

    def _carve_vertical_corridor(self, y1, y2, x, half):
        ya, yb = sorted((y1, y2))
        x0 = max(1, x - half)
        x1 = min(self.map_size - 1, x - half + CORRIDOR_WIDTH)
        self.ground_truth[x0:x1, ya:yb + 1] = 0

    def _place_room_obstacles(self, room):
        x, y, w, h = room
        n_obs = np.random.randint(OBSTACLES_PER_ROOM_MIN, OBSTACLES_PER_ROOM_MAX + 1)
        ix1 = x + 1 + OBSTACLE_WALL_MARGIN
        iy1 = y + 1 + OBSTACLE_WALL_MARGIN
        ix2 = x + w - OBSTACLE_WALL_MARGIN
        iy2 = y + h - OBSTACLE_WALL_MARGIN
        if ix2 <= ix1 or iy2 <= iy1:
            return

        for _ in range(n_obs):
            for _ in range(OBSTACLE_PLACE_ATTEMPTS):
                ow = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
                oh = np.random.randint(OBS_MIN_SIZE, OBS_MAX_SIZE + 1)
                if ix2 - ow + 1 < ix1 or iy2 - oh + 1 < iy1:
                    break

                rx = np.random.randint(ix1, ix2 - ow + 2)
                ry = np.random.randint(iy1, iy2 - oh + 2)
                if np.all(self.ground_truth[rx:rx + ow, ry:ry + oh] == 0):
                    self.ground_truth[rx:rx + ow, ry:ry + oh] = 1
                    break

    def _get_valid_spawn_pos(self):
        """在随机房间内部寻找合法出生点（周围 3x3 空旷）"""
        if not self.rooms:
            raise RuntimeError("地图生成失败：未放置任何房间")

        for _ in range(500):
            room = self.rooms[np.random.randint(len(self.rooms))]
            x, y, w, h = room
            x_lo = x + 1 + SPAWN_MARGIN
            x_hi = x + w - SPAWN_MARGIN
            y_lo = y + 1 + SPAWN_MARGIN
            y_hi = y + h - SPAWN_MARGIN
            if x_lo >= x_hi or y_lo >= y_hi:
                continue

            px = np.random.randint(x_lo, x_hi)
            py = np.random.randint(y_lo, y_hi)
            if np.all(self.ground_truth[px - 1:px + 2, py - 1:py + 2] == 0):
                return px, py

        room = self.rooms[0]
        x, y, w, h = room
        return x + 1 + w // 2, y + 1 + h // 2

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
