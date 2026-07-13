from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import SENSOR_RANGE, TRAIN_MAP_SIZE


class StateTensorPipeline:
    UNKNOWN_FRONTIER_CHANNEL = 0
    FRONTIER_CHANNEL = 4

    def __init__(self, map_size=TRAIN_MAP_SIZE, sigma=None, device='cpu'):
        """
        状态张量处理流水线：生成 5xHxW 的 PyTorch Tensor
        :param map_size: 地图边长
        :param sigma: 兼容旧接口，当前 Channel 3 使用 BFS 可达距离场，不再使用 sigma
        :param device: 计算设备 ('cpu' 或 'cuda')
        """
        self.map_size = map_size
        self.device = device
        self.reachable_floor = 0.05
        self.sensor_range = SENSOR_RANGE
        
        # 定义用于提取 0/-1 边界的邻域检测卷积核（上下左右 4 邻域）。
        # free-side frontier: 自身是空地，四邻域存在未知。
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.float32)
        self.conv_kernel = torch.tensor(kernel, device=device).unsqueeze(0).unsqueeze(0)

    def process(self, explored_map, robot_pos):
        """
        将原始地图和机器人坐标转化为 5 个通道的特征张量。
        Channel 0 改为版本 C 的可见信息增益图：只统计“当前已知障碍”未遮挡的可见 unknown 数量。
        Channel 3 使用 BFS 可达距离场：机器人位置为 1，最远可达空地为 0.05，未知/障碍/不可达区域为 0。
        :param explored_map: np.ndarray, 形状 (H, W), 元素为 -1, 0, 1
        :param robot_pos: list or tuple [rx, ry]
        :return: torch.Tensor, 形状 (5, H, W)
        """
        rx, ry = robot_pos
        map_tensor = torch.tensor(explored_map, dtype=torch.float32, device=self.device)
        
        c_unknown = (map_tensor == -1.0).float()
        c_free = (map_tensor == 0.0).float()
        c_obstacle = (map_tensor == 1.0).float()
        
        # --- Channel 4: free-side frontier，作为合法动作掩码 ---
        c_frontier = self._extract_frontier_mask(c_unknown, c_free)
        
        # --- Channel 0: gain map，仅统计从候选站位出发“可见”的外边界数量 ---
        # 只对 reachable frontier 赋值，其余位置保持为 0，避免无效候选污染输入。
        c_gain_map = self._build_visibility_aware_gain_map(explored_map, c_free, c_frontier, (rx, ry))
        
        # --- Channel 3: BFS 可达距离场。仅在当前已知空地内扩展，不穿过未知或障碍。 ---
        c_bfs_reachable_distance = torch.tensor(
            self._build_bfs_reachable_distance_field(explored_map, (rx, ry)),
            dtype=torch.float32,
            device=self.device,
        )
        
        # 将 5 个通道堆叠拼接到一起 -> (5, H, W)
        state_tensor = torch.stack([c_gain_map, c_free, c_obstacle, c_bfs_reachable_distance, c_frontier], dim=0)
        
        return state_tensor

    def _build_bfs_reachable_distance_field(self, explored_map, robot_pos):
        """
        构建 BFS 可达距离场。BFS 只在 explored_map == 0 的已知空地内扩展，不穿过未知区或障碍。
        可达空地使用版本 C 归一化：robot=1，最远可达空地=reachable_floor，不可达/未知/障碍=0。
        """
        h, w = explored_map.shape
        ry, rx = int(robot_pos[0]), int(robot_pos[1])
        dist = np.full((h, w), -1, dtype=np.int32)

        if not (0 <= ry < h and 0 <= rx < w) or explored_map[ry, rx] != 0:
            return np.zeros((h, w), dtype=np.float32)

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

        reachable = dist >= 0
        field = np.zeros((h, w), dtype=np.float32)
        max_dist = int(dist[reachable].max()) if np.any(reachable) else 0
        if max_dist <= 0:
            field[reachable] = 1.0
            return field

        normalized = 1.0 - dist[reachable].astype(np.float32) / float(max_dist)
        field[reachable] = self.reachable_floor + (1.0 - self.reachable_floor) * normalized
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

        for y, x in np.argwhere(candidate_mask):
            y0 = max(0, y - max_r)
            y1 = min(h - 1, y + max_r)
            x0 = max(0, x - max_r)
            x1 = min(w - 1, x + max_r)
            visible_outer_boundary_count = 0
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
                    # 外边界：-1 与 0 的交界，或 -1 与 1 的交界。
                    is_outer_boundary = False
                    for by, bx in ((ny - 1, nx), (ny + 1, nx), (ny, nx - 1), (ny, nx + 1)):
                        if by < 0 or by >= h or bx < 0 or bx >= w:
                            continue
                        if explored_map[by, bx] in (0, 1):
                            is_outer_boundary = True
                            break
                    if is_outer_boundary:
                        visible_outer_boundary_count += 1
            gain_map[y, x] = float(visible_outer_boundary_count)

        max_gain = float(gain_map.max())
        if max_gain > 0:
            gain_map /= max_gain
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
        """
        提取 free-side frontier：自身为已知空地，且相邻元素包含未知迷雾。
        这个通道用于动作选择 mask。
        """
        is_unknown = c_unknown.unsqueeze(0).unsqueeze(0)
        unknown_neighbors = F.conv2d(is_unknown, self.conv_kernel, padding=1).squeeze()
        frontier_mask = ((c_free > 0.5) & (unknown_neighbors > 0.5)).float()
        
        return frontier_mask

    def _extract_unknown_frontier_mask(self, c_unknown, c_free):
        """
        提取 unknown-side frontier：自身为未知迷雾，且四邻域存在已知空地。
        只保留 0/-1 边界的 -1 侧；单纯 1/-1 边界不会被标记。
        """
        is_free = c_free.unsqueeze(0).unsqueeze(0)
        free_neighbors = F.conv2d(is_free, self.conv_kernel, padding=1).squeeze()
        unknown_frontier_mask = ((c_unknown > 0.5) & (free_neighbors > 0.5)).float()
        
        return unknown_frontier_mask