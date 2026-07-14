import heapq


class AStarPlanner:
    def __init__(self, fog_penalty=2.0):
        # 4方向移动：上下左右
        self.motions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # 进入迷雾区域的额外代价，越大越倾向走已知区域
        self.fog_penalty = fog_penalty

    def plan(self, explored_map, start, goal):
        """
        在认知地图上寻找起点到终点的最短路径
        :param explored_map: 当前探明的栅格 (-1:迷雾, 0:空地, 1:障碍)
        :param start: [x, y]
        :param goal: [x, y]
        :return: 路径列表 [[x0,y0], [x1,y1], ...], 若无路返回 None
        """
        sx, sy = start[0], start[1]
        gx, gy = goal[0], goal[1]

        # 校验：如果目标点是已知的墙，直接判定不可达
        if explored_map[gx, gy] == 1:
            return None
        if sx == gx and sy == gy:
            return [[sx, sy]]

        H, W = explored_map.shape
        open_set = []
        # (f_cost, g_cost, x, y)
        heapq.heappush(open_set, (self._heuristic(sx, sy, gx, gy), 0, sx, sy))

        came_from = {}
        g_costs = {(sx, sy): 0}
        closed_set = set()

        while open_set:
            _, g, curr_x, curr_y = heapq.heappop(open_set)

            if curr_x == gx and curr_y == gy:
                return self._reconstruct_path(came_from, curr_x, curr_y)

            if (curr_x, curr_y) in closed_set:
                continue
            closed_set.add((curr_x, curr_y))

            for dx, dy in self.motions:
                nx, ny = curr_x + dx, curr_y + dy

                # 越界检查
                if not (0 <= nx < H and 0 <= ny < W):
                    continue

                # 只有已知墙壁 (1) 绝对不能走；未知迷雾 (-1) 和已知空地 (0) 允许尝试前往
                if explored_map[nx, ny] == 1:
                    continue

                step_cost = 1.0
                if explored_map[nx, ny] == -1:
                    step_cost += self.fog_penalty

                new_g = g + step_cost
                if (nx, ny) not in g_costs or new_g < g_costs[(nx, ny)]:
                    g_costs[(nx, ny)] = new_g
                    f = new_g + self._heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (f, new_g, nx, ny))
                    came_from[(nx, ny)] = (curr_x, curr_y)

        return None  # 死路

    def _heuristic(self, x1, y1, x2, y2):
        """曼哈顿距离启发函数"""
        return abs(x1 - x2) + abs(y1 - y2)

    def _reconstruct_path(self, came_from, curr_x, curr_y):
        path = [[curr_x, curr_y]]
        while (curr_x, curr_y) in came_from:
            curr_x, curr_y = came_from[(curr_x, curr_y)]
            path.append([curr_x, curr_y])
        path.reverse()
        return path