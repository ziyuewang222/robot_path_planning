#第二版代码：mdp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

class RLAutoExplorer:
    def __init__(self, map_size=30, sensor_range=4):
        self.map_size = map_size
        self.sensor_range = sensor_range
        
        # 地图初始化
        self.ground_truth = np.zeros((map_size, map_size))
        self._build_walls()
        self.robot_map = np.full((map_size, map_size), -1)
        self.robot_pos = self._get_valid_pos()
        self.frontiers = []
        
        # 记录 RL 的状态价值矩阵 V(s)
        self.value_map = np.zeros((map_size, map_size))
        self.scan_lidar()

    def _build_walls(self):
        """构建模拟真实室内环境的复杂地图"""
        # 1. 基础外围墙壁
        self.ground_truth[0, :] = 1
        self.ground_truth[-1, :] = 1
        self.ground_truth[:, 0] = 1
        self.ground_truth[:, -1] = 1
        
        # 2. 构建长墙隔断（模拟走廊和房间）
        # 横向墙壁
        self.ground_truth[15:17, 0:45] = 1
        self.ground_truth[45:47, 15:60] = 1
        # 纵向墙壁
        self.ground_truth[15:45, 30:32] = 1

        # 3. 挖出门洞 (将墙壁部分重新设为空闲 0)
        self.ground_truth[15:17, 10:15] = 0 # 房间1的门
        self.ground_truth[15:17, 35:40] = 0 # 房间2的门
        self.ground_truth[45:47, 45:50] = 0 # 房间3的门
        self.ground_truth[25:30, 30:32] = 0 # 走廊连接门

        # 4. 随机散布小障碍物 (模拟实验室的桌子、箱子、柱子)
        np.random.seed(42) # 固定随机种子，保证每次生成的障碍物位置一样便于调试
        for _ in range(35):
            x = np.random.randint(2, self.map_size-4)
            y = np.random.randint(2, self.map_size-4)
            # 只有当该位置是空地时才放置障碍物，避免把墙变厚
            if self.ground_truth[x, y] == 0:
                # 放置 2x2 到 3x3 大小的随机障碍物
                w, h = np.random.randint(2, 4, 2)
                self.ground_truth[x:x+w, y:y+h] = 1
    def _get_valid_pos(self):
        while True:
            x, y = np.random.randint(1, self.map_size-1, 2)
            if self.ground_truth[x, y] == 0:
                return [x, y]

    def scan_lidar(self):
        rx, ry = self.robot_pos
        r = self.sensor_range
        x_min, x_max = max(0, rx - r), min(self.map_size, rx + r + 1)
        y_min, y_max = max(0, ry - r), min(self.map_size, ry + r + 1)
        self.robot_map[x_min:x_max, y_min:y_max] = self.ground_truth[x_min:x_max, y_min:y_max]

    def find_frontiers(self):
        self.frontiers = []
        for x in range(1, self.map_size - 1):
            for y in range(1, self.map_size - 1):
                if self.robot_map[x, y] == 0:
                    neighbors = [self.robot_map[x-1, y], self.robot_map[x+1, y],
                                 self.robot_map[x, y-1], self.robot_map[x, y+1]]
                    if -1 in neighbors:
                        self.frontiers.append((x, y))

    def plan_next_step(self):
        """【算法核心】：使用 RL 价值迭代 (Value Iteration) 进行全局决策"""
        if not self.frontiers:
            return None
            
        # 1. 状态空间 (State Space)：当前的认知地图
        # 2. 奖励函数 (Reward Function)：边界点赋予高额奖励，障碍物赋予负奖励
        self.value_map = np.zeros((self.map_size, self.map_size))
        for fx, fy in self.frontiers:
            self.value_map[fx, fy] = 100.0  # 终点奖励
            
        # RL 超参数
        gamma = 0.90     # 折扣因子 (Discount Factor)：决定小车是注重当下还是长远规划
        step_cost = -1.0 # 动作代价 (Action Cost)：鼓励小车走最短路径
        
        # 3. 价值迭代推演 (模拟 Bellman Optimality Update)
        # 在真实环境中，这里可以用预训练的神经网络输出；这里我们在线迭代求解
        iterations = max(20, self.map_size) # 迭代次数需足以让价值波及全图
        for _ in range(iterations):
            v_new = np.copy(self.value_map)
            for x in range(1, self.map_size - 1):
                for y in range(1, self.map_size - 1):
                    # 仅在已知的安全空地上更新价值
                    if self.robot_map[x, y] == 0 and (x, y) not in self.frontiers:
                        max_v = -float('inf')
                        # 遍历动作空间 Action Space：上下左右
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if self.robot_map[nx, ny] == 0 or (nx, ny) in self.frontiers:
                                max_v = max(max_v, self.value_map[nx, ny])
                        
                        if max_v != -float('inf'):
                            # Bellman 方程核心
                            v_new[x, y] = step_cost + gamma * max_v
            self.value_map = v_new

        # 4. 策略提取 (Policy Extraction)：基于当前的 V(s) 选择价值最大的动作
        rx, ry = self.robot_pos
        best_action = None
        best_val = -float('inf')
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = rx + dx, ry + dy
            if self.robot_map[nx, ny] == 0 or (nx, ny) in self.frontiers:
                if self.value_map[nx, ny] > best_val:
                    best_val = self.value_map[nx, ny]
                    best_action = [nx, ny]
                    
        return best_action

    def render(self, step):
        plt.clf()
        
        # 底图渲染逻辑
        display_map = np.full((self.map_size, self.map_size), np.nan)
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.robot_map[x, y] == -1:
                    display_map[x, y] = -50  # 未知区域涂暗
                elif self.robot_map[x, y] == 1:
                    display_map[x, y] = -100 # 墙壁涂黑
                else:
                    # 空闲区域显示 RL 传递出的价值 V(s)
                    display_map[x, y] = self.value_map[x, y]

        # 绘制价值热力图
        plt.imshow(display_map, cmap='viridis', origin='upper')
        plt.colorbar(label='State Value $V(s)$')

        if self.frontiers:
            fx, fy = zip(*self.frontiers)
            plt.plot(fy, fx, 'r*', markersize=4, label="Rewards (Frontiers)")

        plt.plot(self.robot_pos[1], self.robot_pos[0], 'wo', markersize=8, markeredgecolor='black', label="Robot Agent")

        plt.title(f"RL Value Iteration Exploration - Step {step}")
        plt.legend(loc='upper right')
        plt.pause(0.05)

if __name__ == '__main__':
    print("启动强化学习价值迭代探索仿真...")
    robot = RLAutoExplorer(map_size=100, sensor_range=3)
    plt.ion()
    plt.figure(figsize=(9, 7))

    step = 0
    while True:
        robot.find_frontiers()
        robot.plan_next_step() # 先计算 V(s) 以便渲染
        robot.render(step)
        next_pos = robot.plan_next_step() # 提取策略
        
        if next_pos is None:
            print("探索完成！")
            break
            
        robot.robot_pos = next_pos
        robot.scan_lidar()
        step += 1

    plt.ioff()
    plt.show()