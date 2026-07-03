#第一版代码：bfs探索，非常贪心容易被一点噪声吸引忽略一大片未知区域。



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
import time

class AutoExplorer:
    def __init__(self, map_size=30, sensor_range=5):
        self.map_size = map_size
        self.sensor_range = sensor_range
        
        # 1. 真实地图 (上帝视角)
        self.ground_truth = np.zeros((map_size, map_size))
        self._build_walls()
        
        # 2. 机器人认知地图 (-1:未知, 0:空闲, 1:障碍)
        self.robot_map = np.full((map_size, map_size), -1)
        self.robot_pos = self._get_valid_pos()
        
        self.frontiers = [] # 用于存储当前所有的边界点
        self.scan_lidar()   # 出生第一眼先扫描

    # def _build_walls(self):
    #     """构建围墙和几个随机的房间块"""
    #     self.ground_truth[0, :] = 1
    #     self.ground_truth[-1, :] = 1
    #     self.ground_truth[:, 0] = 1
    #     self.ground_truth[:, -1] = 1
    #     np.random.seed(42) 
    #     for _ in range(7):
    #         x, y = np.random.randint(2, self.map_size-6, 2)
    #         self.ground_truth[x:x+5, y:y+5] = 1
    def _build_walls(self):
        """构建模拟真实室内环境的复杂地图"""
        # 1. 基础外围墙壁（地图边缘全设为 1）
        self.ground_truth[0, :] = 1
        self.ground_truth[-1, :] = 1
        self.ground_truth[:, 0] = 1
        self.ground_truth[:, -1] = 1
        
        # 2. 构建长墙隔断（模拟走廊和房间结构）
        # 横向墙壁，分隔出一个纵向走廊
        self.ground_truth[15:17, 0:45] = 1
        self.ground_truth[45:47, 15:60] = 1
        # 纵向墙壁，分隔出一个房间
        self.ground_truth[15:45, 30:32] = 1

        # 挖出门洞（将墙壁部分重新设为空闲 0，门宽可以设大一点）
        self.ground_truth[15:17, 10:15] = 0 # 房间1的门
        self.ground_truth[15:17, 35:40] = 0 # 房间2的门
        self.ground_truth[45:47, 45:50] = 0 # 门3
        self.ground_truth[25:30, 30:32] = 0 # 走廊连接门

        # 3. 随机散布较小的障碍物（模拟杂物、桌椅）
        np.random.seed(42) # 固定随机种子以便复现
        num_extra_obstacles = 25 # 增加障碍物数量
        for _ in range(num_extra_obstacles):
            # 在地图内部随机选择位置
            x = np.random.randint(2, self.map_size - 6)
            y = np.random.randint(2, self.map_size - 6)
            # 只有当该位置是空地时才放置障碍物，避免障碍物叠放在墙里
            if self.ground_truth[x, y] == 0:
                # 放置 2x2 到 4x4 大小的随机障碍物
                w = np.random.randint(2, 5) 
                h = np.random.randint(2, 5)
                # 检查边界，避免杂物超出外墙
                x_end = min(x + w, self.map_size - 1)
                y_end = min(y + h, self.map_size - 1)
                self.ground_truth[x:x_end, y:y_end] = 1
    def _get_valid_pos(self):
        """在空地上随机降生"""
        while True:
            x, y = np.random.randint(1, self.map_size-1, 2)
            if self.ground_truth[x, y] == 0:
                return [x, y]

    def scan_lidar(self):
        """模拟雷达扫描，将上帝视角的真实地形更新到机器人地图中"""
        rx, ry = self.robot_pos
        r = self.sensor_range
        x_min, x_max = max(0, rx - r), min(self.map_size, rx + r + 1)
        y_min, y_max = max(0, ry - r), min(self.map_size, ry + r + 1)
        self.robot_map[x_min:x_max, y_min:y_max] = self.ground_truth[x_min:x_max, y_min:y_max]

    def find_frontiers(self):
        """【算法模块1】：寻找已知空地与未知迷雾的交界处"""
        self.frontiers = []
        for x in range(1, self.map_size - 1):
            for y in range(1, self.map_size - 1):
                # 如果这个点是已知空地，检查它上下左右四个邻居
                if self.robot_map[x, y] == 0:
                    neighbors = [self.robot_map[x-1, y], self.robot_map[x+1, y],
                                 self.robot_map[x, y-1], self.robot_map[x, y+1]]
                    # 只要有一个邻居是 -1(未知)，它就是边界！
                    if -1 in neighbors:
                        self.frontiers.append((x, y))

    def plan_next_step(self):
        """【算法模块2】:使用广度优先搜索(BFS)寻找最近的边界，决定下一步走哪"""
        if not self.frontiers:
            return None # 没边界了，说明图建完了
        
        start = tuple(self.robot_pos)
        queue = deque([start])
        visited = set([start])
        came_from = {start: None} # 记录路径从哪里来
        
        target_frontier = None

        # 向四周像水波一样扩散寻找最近的边界点
        while queue:
            current = queue.popleft()
            
            # 找到了最近的边界
            if current in self.frontiers:
                target_frontier = current
                break
                
            # 遍历上下左右
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (current[0] + dx, current[1] + dy)
                # 只能在已知的空地(0)上规划路径
                if 0 <= nxt[0] < self.map_size and 0 <= nxt[1] < self.map_size:
                    if nxt not in visited and self.robot_map[nxt[0], nxt[1]] == 0:
                        visited.add(nxt)
                        queue.append(nxt)
                        came_from[nxt] = current

        if not target_frontier:
            return None # 可能被墙壁死死卡住了
        

        # 回溯路径：从目标点往回倒推，找到出发后的第一步
        curr = target_frontier
        path = []
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
            
        return list(path[-1]) # 返回下一步的坐标

    def render(self, step):
        """可视化渲染"""
        plt.clf()
        cmap = colors.ListedColormap(['gray', 'white', 'black'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(self.robot_map, cmap=cmap, norm=norm, origin='upper')

        # 画出绿色的边界点
        if self.frontiers:
            fx, fy = zip(*self.frontiers)
            # 注意 matplotlib 画图时，x,y 坐标轴与矩阵行、列是反过来的
            plt.plot(fy, fx, 'gs', markersize=3, label="Frontiers")

        # 画出小车位置
        plt.plot(self.robot_pos[1], self.robot_pos[0], 'ro', markersize=8, label="Robot")

        plt.title(f"Autonomous Exploration - Step {step}")
        plt.legend(loc='upper right')
        plt.grid(True, color='lightgray', linewidth=0.5)
        plt.pause(0.05)

# ================= 核心执行主循环 =================
if __name__ == '__main__':
    print("启动自主探索仿真...")
    robot = AutoExplorer(map_size=100, sensor_range=10) 

    plt.ion()
    plt.figure(figsize=(8, 8))

    step = 0
    while True:
        # 1. 寻找地图上所有的边界
        robot.find_frontiers()
        
        # 2. 渲染当前画面
        robot.render(step)
        
        # 3. 算法决策下一步往哪走
        next_pos = robot.plan_next_step()
        
        if next_pos is None:
            print("没有找到新的边界，地图探索完成！")
            break
            
        # 4. 执行移动并再次扫描雷达
        robot.robot_pos = next_pos
        robot.scan_lidar()
        
        step += 1

    plt.ioff()
    plt.show()