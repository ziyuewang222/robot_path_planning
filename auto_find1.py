import numpy as np
from scipy.ndimage import label
import heapq

class PathPlannerNode:
    def __init__(self):
        # 定义状态常量，需与你的组员对齐
        self.FREE = 0
        self.WALL = 1
        self.UNKNOWN = 2

    def process_map_to_path(self, grid_map, start_pos):
        """
        你的核心黑盒接口
        输入: grid_map (二维numpy数组), start_pos (起始坐标 [x, y])
        输出: 路径点集 [(x1, y1), (x2, y2), ...]
        """
        # 1. 提取所有边界点
        frontiers = self._get_frontier_pixels(grid_map)
        if not frontiers:
            return [] # 没有边界，探索完成

        # 2. 对边界进行聚类，并计算最合理的目标点
        target_point = self._select_best_target(frontiers, grid_map, start_pos)
        if not target_point:
            return []

        # 3. 生成从起点到目标点的路径
        path = self._a_star_search(grid_map, tuple(start_pos), target_point)
        return path

    def _get_frontier_pixels(self, grid_map):
        """找出所有 (空闲区域 且 毗邻 未知区域) 的像素点"""
        frontiers = []
        rows, cols = grid_map.shape
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid_map[r, c] == self.FREE:
                    # 检查上下左右是否有未知区域
                    neighbors = [
                        grid_map[r-1, c], grid_map[r+1, c],
                        grid_map[r, c-1], grid_map[r, c+1]
                    ]
                    if self.UNKNOWN in neighbors:
                        frontiers.append((r, c))
        return frontiers

    def _select_best_target(self, frontiers, grid_map, start_pos):
        """
        对散落的边界点进行聚类(连成线)，并选择一条最大的边界线，取其中点作为目标。
        这解决了你提到的“如何生成一个更加便捷的目标点”的问题。
        """
        # 创建一个与地图等大的全0矩阵，将边界点标为1
        frontier_map = np.zeros_like(grid_map)
        for r, c in frontiers:
            frontier_map[r, c] = 1

        # 使用 scipy 的 label 进行连通域聚类 (把挨在一起的点算作同一个群体)
        labeled_array, num_features = label(frontier_map)
        
        if num_features == 0:
            return None

        # 找出包含像素最多的那个边界群体 (最大的一条未探索边界)
        largest_cluster_id = 0
        max_size = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            if size > max_size:
                max_size = size
                largest_cluster_id = i

        # 获取该最大边界群体的所有坐标
        cluster_coords = np.argwhere(labeled_array == largest_cluster_id)
        
        # 计算该群体的几何中心点 (均值)
        center_r = int(np.mean(cluster_coords[:, 0]))
        center_c = int(np.mean(cluster_coords[:, 1]))

        # 由于几何中心可能落在墙壁或未知区域，我们需要在群体中找一个离中心最近的边界点作为真实目标
        target = tuple(cluster_coords[np.argmin(np.sum((cluster_coords - [center_r, center_c])**2, axis=1))])
        return target

    def _a_star_search(self, grid_map, start, goal):
        """传统的 A* 算法，输入起点和终点，输出路径点集"""
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if 0 <= neighbor[0] < grid_map.shape[0] and 0 <= neighbor[1] < grid_map.shape[1]:
                    # 关键限制：寻路只能在确定安全的 FREE 区域内进行
                    if grid_map[neighbor[0]][neighbor[1]] != self.FREE:
                        continue
                else:
                    continue
                move_cost = 1 if i == 0 or j == 0 else 1.414
                tentative_g_score = gscore[current] + move_cost
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

# 测试你的模块接口
if __name__ == "__main__":
    planner = PathPlannerNode()
    
    # 模拟组员传给你的数据: 0空闲, 1墙壁, 2未知
    mock_map = np.full((20, 20), 2)  # 默认全未知
    mock_map[5:15, 5:15] = 0         # 中间扫出一片空地
    mock_map[5:15, 14] = 1           # 右边扫到一堵墙
    
    current_robot_pos = [10, 10]     # 小车当前在空地中央
    
    # 执行你的模块逻辑
    path_points = planner.process_map_to_path(mock_map, current_robot_pos)
    
    print(f"生成的路径点集: {path_points}")