import heapq


class AStarPlanner:
    def __init__(self, fog_penalty=0.8):
        # 4方向移动：上下左右
        self.motions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # 进入迷雾区域的额外代价，越大越倾向走已知区域
        self.fog_penalty = fog_penalty

    def plan(self, explored_map, start, goal):
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])

        if not (0 <= gx < explored_map.shape[0] and 0 <= gy < explored_map.shape[1]):
            return None
        if explored_map[gx, gy] == 1:
            return None
        if sx == gx and sy == gy:
            return [[sx, sy]]

        H, W = explored_map.shape
        open_set = []
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

                if not (0 <= nx < H and 0 <= ny < W):
                    continue
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

        return None 

    def _heuristic(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def _reconstruct_path(self, came_from, curr_x, curr_y):
        path = [[curr_x, curr_y]]
        while (curr_x, curr_y) in came_from:
            curr_x, curr_y = came_from[(curr_x, curr_y)]
            path.append([curr_x, curr_y])
        path.reverse()
        return path