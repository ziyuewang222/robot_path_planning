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
        area_reward_scale=AREA_REWARD_SCALE,
        penalty_step=STEP_PENALTY,
        penalty_collision=COLLISION_PENALTY,
    ):
        robot_pos_before = tuple(self.robot_pos)
        known_before = self.explored_map != -1
        area_before = np.sum(known_before)
        room_before = self._locate_room(robot_pos_before)
        target_room = self._locate_room(tuple(target_coord))
        room_completion_before = self._room_completion_ratio(room_before)
        room_frontiers_before = self._room_frontier_count(room_before)

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

        known_after = self.explored_map != -1
        area_after = np.sum(known_after)
        newly_known = known_after & ~known_before
        delta_area = float(area_after - area_before)

        room_after = self._locate_room(tuple(self.robot_pos))
        room_completion_after = self._room_completion_ratio(room_after)
        same_room_completion_after = self._room_completion_ratio(room_before)
        room_completion_gain = max(0.0, same_room_completion_after - room_completion_before)
        delta_area_current_room = float(self._room_known_delta(newly_known, room_before))
        delta_area_outside_current_room = max(0.0, delta_area - delta_area_current_room)

        room_completion_bonus = 0.0
        completion_progress_reward = 0.0
        early_exit_penalty = 0.0
        unfinished_frontier_penalty = 0.0
        frontier_jump_penalty = 0.0
        revisit_penalty = 0.0
        completed_room_revisit_penalty = 0.0
        abandoned_return_penalty = 0.0
        recent_repeat_penalty = self._recent_repeat_penalty(tuple(self.robot_pos))
        area_reward = area_reward_scale * delta_area

        near_frontier_score, near_frontier_dist = self._near_frontier_gain_score(robot_pos_before, radius=max(20, self.sensor_range * 2), top_k=5)
        target_frontier_score, target_frontier_dist = self._near_frontier_gain_score(tuple(target_coord), radius=self.sensor_range, top_k=3)

        if room_before is not None:
            before_ratio = room_completion_before
            after_ratio = same_room_completion_after
            current_room_unfinished = before_ratio < 0.995
            target_left_current_room = target_room != room_before
            path_left_current_room = room_after != room_before

            if target_left_current_room or path_left_current_room:
                if current_room_unfinished:
                    early_exit_penalty += 28.0 + 90.0 * (1.0 - before_ratio)
                else:
                    early_exit_penalty += 8.0 + 24.0 * max(0.0, 1.0 - before_ratio)

                if room_frontiers_before > 0:
                    frontier_pressure = min(float(room_frontiers_before), 20.0)
                    frontier_completion_gate = 0.5 + 0.5 * before_ratio
                    unfinished_frontier_penalty = (10.0 + 2.5 * frontier_pressure) * frontier_completion_gate
                    if before_ratio < 0.995:
                        self.abandoned_rooms.add(room_before)

            if near_frontier_dist is not None and target_frontier_dist is not None:
                if target_frontier_dist > near_frontier_dist + max(6.0, self.sensor_range * 0.5):
                    frontier_jump_penalty = 8.0 + 0.4 * (target_frontier_dist - near_frontier_dist)

            if room_after == room_before:
                completion_progress_reward = 80.0 * max(0.0, after_ratio ** 2 - before_ratio ** 2)

            if before_ratio < 0.995 and after_ratio >= 0.995:
                room_completion_bonus = 60.0 + 40.0 * max(0.0, after_ratio - 0.95)
                self.completed_rooms.add(room_before)
                self.abandoned_rooms.discard(room_before)

        if room_after in self.completed_rooms:
            completed_room_revisit_penalty = 8.0

        if room_after in self.abandoned_rooms and room_after != room_before:
            abandoned_return_penalty = 12.0

        room_progress_bonus = completion_progress_reward + room_completion_bonus
        macro_reward = (
            area_reward
            + 3.0 * (delta_area / (n_steps + 1.0))
            + 150.0 * room_completion_gain
            + 0.60 * room_progress_bonus
            - penalty_step * n_steps
            - early_exit_penalty
            - unfinished_frontier_penalty
            - revisit_penalty
            - completed_room_revisit_penalty
            - abandoned_return_penalty
            - frontier_jump_penalty
            - recent_repeat_penalty
        )
        if collision_occurred or path_blocked:
            macro_reward -= penalty_collision

        total_free_space = np.sum(self.ground_truth == 0)
        explored_free_space = np.sum(self.explored_map == 0)
        done = (explored_free_space >= total_free_space * 0.995)

        self.current_room_idx = room_after

        info = {
            'delta_area': delta_area,
            'n_steps': n_steps,
            'collision': collision_occurred,
            'blocked': path_blocked,
            'room_before': room_before,
            'room_after': room_after,
            'room_completion_before': room_completion_before,
            'room_completion_after': room_completion_after,
            'room_completion_gain': room_completion_gain,
            'room_completion_bonus': room_completion_bonus,
            'room_frontiers_before': room_frontiers_before,
            'delta_area_current_room': delta_area_current_room,
            'delta_area_outside_current_room': delta_area_outside_current_room,
            'area_reward': area_reward,
            'completion_progress_reward': completion_progress_reward,
            'early_exit_penalty': early_exit_penalty,
            'unfinished_frontier_penalty': unfinished_frontier_penalty,
            'revisit_penalty': revisit_penalty,
            'completed_room_revisit_penalty': completed_room_revisit_penalty,
            'abandoned_return_penalty': abandoned_return_penalty,
            'frontier_jump_penalty': frontier_jump_penalty,
            'recent_repeat_penalty': recent_repeat_penalty,
            'near_frontier_score': near_frontier_score,
            'near_frontier_dist': near_frontier_dist,
            'target_frontier_score': target_frontier_score,
            'target_frontier_dist': target_frontier_dist,
        }

        return self.explored_map.copy(), self.robot_pos.copy(), macro_reward, n_steps, done, info

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

    def _room_completion_ratio(self, room_idx):
        if room_idx is None or not self.rooms:
            return 0.0
        room = self.rooms[room_idx]
        x, y, w, h = room['bbox']
        patch = self.explored_map[x + 1:x + 1 + w, y + 1:y + 1 + h]
        if patch.size == 0:
            return 0.0
        known_free = np.sum(patch == 0)
        known_obstacle = np.sum(patch == 1)
        known = known_free + known_obstacle
        return float(known) / float(patch.size)

    def _room_known_delta(self, newly_known, room_idx):
        if room_idx is None or not self.rooms:
            return 0
        room = self.rooms[room_idx]
        x, y, w, h = room['bbox']
        patch = newly_known[x + 1:x + 1 + w, y + 1:y + 1 + h]
        if patch.size == 0:
            return 0
        return int(np.sum(patch))

    def _room_frontier_count(self, room_idx):
        if room_idx is None or not self.rooms:
            return 0
        room = self.rooms[room_idx]
        x, y, w, h = room['bbox']
        count = 0
        for rx in range(x + 1, x + 1 + w):
            for ry in range(y + 1, y + 1 + h):
                if self.explored_map[rx, ry] != 0:
                    continue
                for dx, dy in self.action_space.values():
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if self.explored_map[nx, ny] == -1:
                            count += 1
                            break
        return count

    def _is_frontier_cell(self, pos):
        x, y = pos
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return False
        if self.explored_map[x, y] != 0:
            return False
        for dx, dy in self.action_space.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                if self.explored_map[nx, ny] == -1:
                    return True
        return False

    def _line_of_sight_blocked(self, start, end):
        for lx, ly in self._bresenham_line(start[0], start[1], end[0], end[1])[1:-1]:
            if self.explored_map[lx, ly] == 1:
                return True
        return False

    def _frontier_lightweight_gain(self, pos, radius=None):
        if not self._is_frontier_cell(pos):
            return 0.0
        radius = self.sensor_range if radius is None else radius
        px, py = pos
        x_min = max(0, px - radius)
        x_max = min(self.map_size - 1, px + radius)
        y_min = max(0, py - radius)
        y_max = min(self.map_size - 1, py + radius)
        gain = 0
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if self.explored_map[x, y] != -1:
                    continue
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy > radius * radius:
                    continue
                if self._line_of_sight_blocked(pos, (x, y)):
                    continue
                gain += 1
        return float(gain)

    def _near_frontier_gain_score(self, start, radius=25, top_k=5):
        sx, sy = start
        candidates = []
        for x in range(max(0, sx - radius), min(self.map_size, sx + radius + 1)):
            for y in range(max(0, sy - radius), min(self.map_size, sy + radius + 1)):
                dist = abs(x - sx) + abs(y - sy)
                if dist > radius:
                    continue
                if not self._is_frontier_cell((x, y)):
                    continue
                gain = self._frontier_lightweight_gain((x, y))
                if gain <= 0.0:
                    continue
                candidates.append((gain, dist))
        if not candidates:
            return 0.0, None
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:top_k]
        score = min(80.0, float(sum(gain for gain, _ in selected)))
        best_dist = min(dist for _, dist in selected)
        return score, float(best_dist)

    def _nearby_mask_sum(self, mask, center, radius):
        cx, cy = center
        count = 0
        x_min = max(0, cx - radius)
        x_max = min(self.map_size - 1, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(self.map_size - 1, cy + radius)
        radius_sq = radius * radius
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy <= radius_sq and mask[x, y]:
                    count += 1
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
