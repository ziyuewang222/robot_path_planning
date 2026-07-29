
# from collections import deque

# import numpy as np
# import torch
# import torch.nn.functional as F

# from config import SENSOR_RANGE, TRAIN_MAP_SIZE


# class StateTensorPipeline:
#     UNKNOWN_FRONTIER_CHANNEL = 0
#     FRONTIER_CHANNEL = 4

#     def __init__(self, map_size=TRAIN_MAP_SIZE, sigma=None, device='cpu'):
#         self.map_size = map_size
#         self.device = device
#         self.reachable_floor = 0.05
#         self.sensor_range = SENSOR_RANGE
        
#         kernel = np.array([[0, 1, 0],
#                            [1, 0, 1],
#                            [0, 1, 0]], dtype=np.float32)
#         self.conv_kernel = torch.tensor(kernel, device=device).unsqueeze(0).unsqueeze(0)

#     def process(self, explored_map, robot_pos):
#         rx, ry = robot_pos
#         map_tensor = torch.tensor(explored_map, dtype=torch.float32, device=self.device)
        
#         c_unknown = (map_tensor == -1.0).float()
#         c_free = (map_tensor == 0.0).float()
#         c_obstacle = (map_tensor == 1.0).float()
        
#         c_frontier = self._extract_frontier_mask(c_unknown, c_free)
        
#         # --- Channel 0: gain map，已重构为局部前沿密度算法 ---
#         c_gain_map = self._build_visibility_aware_gain_map(explored_map, c_free, c_frontier, (rx, ry))
        
#         c_bfs_reachable_distance = torch.tensor(
#             self._build_bfs_reachable_distance_field(explored_map, (rx, ry)),
#             dtype=torch.float32,
#             device=self.device,
#         )
        
#         state_tensor = torch.stack([c_gain_map, c_free, c_obstacle, c_bfs_reachable_distance, c_frontier], dim=0)
        
#         return state_tensor

#     def _build_bfs_reachable_distance_field(self, explored_map, robot_pos):
#         h, w = explored_map.shape
#         ry, rx = int(robot_pos[0]), int(robot_pos[1])
#         dist = np.full((h, w), -1, dtype=np.int32)

#         if not (0 <= ry < h and 0 <= rx < w) or explored_map[ry, rx] != 0:
#             return np.zeros((h, w), dtype=np.float32)

#         queue = deque([(ry, rx)])
#         dist[ry, rx] = 0
#         while queue:
#             y, x = queue.popleft()
#             next_dist = dist[y, x] + 1
#             for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
#                 if ny < 0 or ny >= h or nx < 0 or nx >= w:
#                     continue
#                 if explored_map[ny, nx] != 0 or dist[ny, nx] >= 0:
#                     continue
#                 dist[ny, nx] = next_dist
#                 queue.append((ny, nx))

#         reachable = dist >= 0
#         field = np.zeros((h, w), dtype=np.float32)
#         max_dist = int(dist[reachable].max()) if np.any(reachable) else 0
#         if max_dist <= 0:
#             field[reachable] = 1.0
#             return field

#         normalized = 1.0 - dist[reachable].astype(np.float32) / float(max_dist)
#         field[reachable] = self.reachable_floor + (1.0 - self.reachable_floor) * normalized
#         return field

#     def _build_visibility_aware_gain_map(self, explored_map, c_free, c_frontier, _robot_pos):
#         h, w = explored_map.shape
#         gain_map = np.zeros((h, w), dtype=np.float32)
        
#         candidate_mask = (c_frontier > 0.5).cpu().numpy()
#         if not np.any(candidate_mask):
#             return torch.tensor(gain_map, dtype=torch.float32, device=self.device)

#         free_mask = (c_free > 0.5).cpu().numpy()
        
#         # 采用“局部前沿密度/跨度”替代存在盲区的射线检测
#         # 窗口半径设为传感器半径的一半
#         local_radius = max(3, int(self.sensor_range * 0.5))
#         frontier_coords = np.argwhere(candidate_mask)
        
#         for y, x in frontier_coords:
#             y0 = max(0, y - local_radius)
#             y1 = min(h - 1, y + local_radius)
#             x0 = max(0, x - local_radius)
#             x1 = min(w - 1, x + local_radius)
            
#             # 核心改变：直接统计局部窗口内的前沿点数量
#             # 空旷大厅前沿线极长，密度极大；狭窄走廊前沿线短，密度极小
#             local_frontier_count = np.sum(candidate_mask[y0:y1+1, x0:x1+1])
#             gain_map[y, x] = float(local_frontier_count)

#         # 保留原始计数值，由 runtime 统一按启发项自身规则归一化
#         gain_map[~free_mask] = 0.0
#         gain_map[~candidate_mask] = 0.0
        
#         return torch.tensor(gain_map, dtype=torch.float32, device=self.device)

#     def _extract_frontier_mask(self, c_unknown, c_free):
#         is_unknown = c_unknown.unsqueeze(0).unsqueeze(0)
#         unknown_neighbors = F.conv2d(is_unknown, self.conv_kernel, padding=1).squeeze()
#         frontier_mask = ((c_free > 0.5) & (unknown_neighbors > 0.5)).float()
#         return frontier_mask
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import SENSOR_RANGE, TRAIN_MAP_SIZE


class StateTensorPipeline:
    UNKNOWN_FRONTIER_CHANNEL = 0
    FRONTIER_CHANNEL = 4

    def __init__(self, map_size=TRAIN_MAP_SIZE, sigma=None, device='cpu'):
        self.map_size = map_size
        self.device = device
        self.sensor_range = SENSOR_RANGE
        
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.float32)
        self.conv_kernel = torch.tensor(kernel, device=device).unsqueeze(0).unsqueeze(0)

    def process(self, explored_map, robot_pos):
        rx, ry = robot_pos
        map_tensor = torch.tensor(explored_map, dtype=torch.float32, device=self.device)
        
        c_unknown = (map_tensor == -1.0).float()
        c_free = (map_tensor == 0.0).float()
        c_obstacle = (map_tensor == 1.0).float()
        
        c_frontier = self._extract_frontier_mask(c_unknown, c_free)
        
        # --- Channel 0: 真实可见未知区域面积 (不进行归一化) ---
        c_gain_map = self._build_visibility_aware_gain_map(explored_map, c_free, c_frontier, (rx, ry))
        
        # --- Channel 3: 绝对 BFS 距离场 (不进行归一化，不可达区域设为地图尺寸上限) ---
        c_bfs_reachable_distance = torch.tensor(
            self._build_bfs_reachable_distance_field(explored_map, (rx, ry)),
            dtype=torch.float32,
            device=self.device,
        )
        
        state_tensor = torch.stack([c_gain_map, c_free, c_obstacle, c_bfs_reachable_distance, c_frontier], dim=0)
        
        return state_tensor

    def _build_bfs_reachable_distance_field(self, explored_map, robot_pos):
        h, w = explored_map.shape
        ry, rx = int(robot_pos[0]), int(robot_pos[1])
        dist = np.full((h, w), -1, dtype=np.int32)

        if not (0 <= ry < h and 0 <= rx < w) or explored_map[ry, rx] != 0:
            return np.full((h, w), h * w, dtype=np.float32)

        queue = deque([(ry, rx)])
        dist[ry, rx] = 0
        while queue:
            y, x = queue.popleft()
            next_dist = dist[y, x] + 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if explored_map[ny, nx] != 0 or dist[ny, nx] >= 0:
                    continue
                dist[ny, nx] = next_dist
                queue.append((ny, nx))

        # 仅输出原始物理步数，无法到达的区域赋予极大值
        field = np.where(dist >= 0, dist.astype(np.float32), float(h * 2))
        return field

    def _build_visibility_aware_gain_map(self, explored_map, c_free, c_frontier, robot_pos):
        h, w = explored_map.shape
        gain_map = np.zeros((h, w), dtype=np.float32)
        if self.sensor_range <= 0:
            return torch.tensor(gain_map, dtype=torch.float32, device=self.device)

        reachable = self._build_bfs_reachable_mask(explored_map, robot_pos)
        candidate_mask = (c_frontier > 0.5).cpu().numpy() & reachable
        if not np.any(candidate_mask):
            return torch.tensor(gain_map, dtype=torch.float32, device=self.device)

        free_mask = (c_free > 0.5).cpu().numpy()
        obstacle_mask = explored_map == 1
        max_r = int(self.sensor_range)

        # 针对每一个前沿点，扫描传感器范围内的未知区域 (-1)
        for y, x in np.argwhere(candidate_mask):
            y0 = max(0, y - max_r)
            y1 = min(h - 1, y + max_r)
            x0 = max(0, x - max_r)
            x1 = min(w - 1, x + max_r)
            
            visible_unknown_count = 0
            
            for ny in range(y0, y1 + 1):
                for nx in range(x0, x1 + 1):
                    if explored_map[ny, nx] != -1:
                        continue
                    dy = ny - y
                    dx = nx - x
                    if dy * dy + dx * dx > max_r * max_r:
                        continue
                    if self._line_of_sight_blocked(explored_map, (y, x), (ny, nx), obstacle_mask):
                        continue
                    visible_unknown_count += 1
            
            # 直接赋值真实面积个数，不再除以 max_gain
            gain_map[y, x] = float(visible_unknown_count)

        gain_map[~free_mask] = 0.0
        gain_map[~candidate_mask] = 0.0
        return torch.tensor(gain_map, dtype=torch.float32, device=self.device)

    def _build_bfs_reachable_mask(self, explored_map, robot_pos):
        h, w = explored_map.shape
        ry, rx = int(robot_pos[0]), int(robot_pos[1])
        reachable = np.zeros((h, w), dtype=bool)
        if not (0 <= ry < h and 0 <= rx < w) or explored_map[ry, rx] != 0:
            return reachable

        queue = deque([(ry, rx)])
        reachable[ry, rx] = True
        while queue:
            y, x = queue.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if explored_map[ny, nx] != 0 or reachable[ny, nx]:
                    continue
                reachable[ny, nx] = True
                queue.append((ny, nx))
        return reachable

    @staticmethod
    def _line_of_sight_blocked(explored_map, start, end, obstacle_mask):
        sy, sx = start
        ey, ex = end
        dy = abs(ey - sy)
        dx = abs(ex - sx)
        y, x = sy, sx
        step_y = 1 if ey >= sy else -1
        step_x = 1 if ex >= sx else -1

        if dx > dy:
            err = dx / 2.0
            while x != ex:
                x += step_x
                err -= dy
                if err < 0:
                    y += step_y
                    err += dx
                if (y, x) == (ey, ex):
                    break
                if obstacle_mask[y, x]:
                    return True
        else:
            err = dy / 2.0
            while y != ey:
                y += step_y
                err -= dx
                if err < 0:
                    x += step_x
                    err += dy
                if (y, x) == (ey, ex):
                    break
                if obstacle_mask[y, x]:
                    return True
        return False

    def _extract_frontier_mask(self, c_unknown, c_free):
        is_unknown = c_unknown.unsqueeze(0).unsqueeze(0)
        unknown_neighbors = F.conv2d(is_unknown, self.conv_kernel, padding=1).squeeze()
        frontier_mask = ((c_free > 0.5) & (unknown_neighbors > 0.5)).float()
        return frontier_mask