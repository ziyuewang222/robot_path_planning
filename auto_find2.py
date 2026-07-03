import numpy as np
from scipy.ndimage import label
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors

class PathPlannerNode:
    def __init__(self):
        # 0空闲, 1墙壁, 2未知
        self.FREE = 0
        self.WALL = 1
        self.UNKNOWN = 2

    def process_map_to_path(self, grid_map, start_pos):
        """黑盒接口：输入地图和起点，输出路径"""
        frontiers = self._get_frontier_pixels(grid_map)
        if not frontiers:
            return []
        target_point = self._select_best_target(frontiers, grid_map, start_pos)
        if not target_point:
            return []
        path = self._a_star_search(grid_map, tuple(start_pos), target_point)
        return path

    def _get_frontier_pixels(self, grid_map):
        """找出所有 (空闲区域 且 毗邻 未知区域) 的像素点"""
        frontiers = []
        rows, cols = grid_map.shape
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid_map[r, c] == self.FREE:
                    neighbors = [
                        grid_map[r-1, c], grid_map[r+1, c],
                        grid_map[r, c-1], grid_map[r, c+1]
                    ]
                    if self.UNKNOWN in neighbors:
                        frontiers.append((r, c))
        return frontiers

    def _select_best_target(self, frontiers, grid_map, start_pos):
        """对边界进行聚类，选择最大边界的中点作为目标"""
        if not frontiers:
            return None
            
        frontier_map = np.zeros_like(grid_map)
        for r, c in frontiers:
            frontier_map[r, c] = 1

        labeled_array, num_features = label(frontier_map)
        if num_features == 0:
            return None

        largest_cluster_id = 0
        max_size = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            if size > max_size:
                max_size = size
                largest_cluster_id = i

        cluster_coords = np.argwhere(labeled_array == largest_cluster_id)
        center_r = int(np.mean(cluster_coords[:, 0]))
        center_c = int(np.mean(cluster_coords[:, 1]))

        # 在该边界群体中找一个离几何中心最近的真实边界点
        distances = np.sum((cluster_coords - [center_r, center_c])**2, axis=1)
        target = tuple(cluster_coords[np.argmin(distances)])
        return target

    def _a_star_search(self, grid_map, start, goal):
        """A* 路径规划"""
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

# ================= 测试与可视化模块 =================
if __name__ == "__main__":
    planner = PathPlannerNode()
    
    # 1. 模拟组员传给你的数据: 0空闲, 1墙壁, 2未知
    mock_map = np.full((20, 20), 2)  # 默认全未知 (外围全是迷雾)
    mock_map[5:15, 5:15] = 0         # 中间扫出一片空地
    mock_map[5:15, 14] = 1           # 空地右侧边缘扫到一堵墙
    
    current_robot_pos = [10, 10]     # 小车当前在空地中央
    
    # 为了绘图，我们把算法的三个步骤拆开调用
    print("正在计算边界...")
    frontiers = planner._get_frontier_pixels(mock_map)
    
    print("正在进行边界聚类并选择最优目标点...")
    target_point = planner._select_best_target(frontiers, mock_map, current_robot_pos)
    
    print(f"确定的目标点为: {target_point}")
    print("正在规划到达目标点的路径...")
    path_points = planner._a_star_search(mock_map, tuple(current_robot_pos), target_point)
    
    print(f"生成的路径点集: {path_points}")
    
    # 2. 开始使用 matplotlib 进行可视化渲染
    plt.figure(figsize=(8, 8))
    
    # 定义颜色映射：0白色(空地), 1黑色(墙壁), 2灰色(未知)
    cmap = colors.ListedColormap(['white', 'black', 'gray'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # 画出底层地图
    plt.imshow(mock_map, cmap=cmap, norm=norm, origin='upper')
    
    # 画出所有提取到的边界点 (绿色小方块)
    if frontiers:
        fy, fx = zip(*frontiers) # 提取所有边界点的行列坐标，绘图时x对应列，y对应行
        plt.plot(fx, fy, 'gs', markersize=4, label='Frontiers (Extracted)')
        
    # 画出小车当前位置 (红色圆点)
    plt.plot(current_robot_pos[1], current_robot_pos[0], 'ro', markersize=8, label='Robot Start')
    
    # 画出算法最终决定的目标点 (黄色五角星)
    if target_point:
        plt.plot(target_point[1], target_point[0], 'y*', markersize=14, markeredgecolor='black', label='Target Point')
        
    # 画出规划的路径 (蓝色实线)
    if path_points:
        py, px = zip(*path_points)
        plt.plot(px, py, 'b-', linewidth=2, label='A* Path')
        
    plt.title("Autonomous Exploration - Map Processing & Path Planning")
    plt.legend(loc='upper right')
    plt.grid(True, color='lightgray', linewidth=0.5)
    
    # 弹出显示窗口
    plt.show()