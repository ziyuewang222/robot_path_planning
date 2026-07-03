# 加入防卡死机制，可以探索完全图

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import time

class FrontierExplorerDemo:
    # def __init__(self, map_size=40, sensor_range=5):
    #     self.map_size = map_size
    #     self.sensor_range = sensor_range
        
    #     # 内部真实地图：0 是空地，1 是墙壁
    #     self.ground_truth = np.zeros((map_size, map_size), dtype=np.int8)
    #     self._build_walls()
        
    #     # 小车的认知地图：初始化全是 -1 (未知)
    #     self.robot_map = np.full((map_size, map_size), -1, dtype=np.int8)
    #     self.robot_pos = [2, 2] # 初始位置
        
    #     # 可视化设置
    #     plt.ion() # 开启交互模式
    #     self.fig, self.ax = plt.subplots(figsize=(8, 8))
    #     # 颜色映射: -1未知(灰色), 0空地(白色), 1墙壁(黑色)
    #     self.cmap = ListedColormap(['gray', 'white', 'black'])
    def __init__(self, map_size=40, sensor_range=5):
        self.map_size = map_size
        self.sensor_range = sensor_range
        
        self.ground_truth = np.zeros((map_size, map_size), dtype=np.int8)
        self._build_walls()
        
        self.robot_map = np.full((map_size, map_size), -1, dtype=np.int8)
        self.robot_pos = [2, 2] 
        
        # ========== 新增：防卡死（死锁检测）机制 ==========
        self.last_pos = [-1, -1]       # 记录上一步的位置
        self.stuck_patience = 0        # 失去耐心的计数器
        self.ignore_closest = False    # 触发降级策略（去第二小边界）的标志位
        # ===================================================

        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cmap = ListedColormap(['gray', 'white', 'black'])
        
    def _build_walls(self):
        """建一个简单的测试迷宫"""
        self.ground_truth[0, :] = 1
        self.ground_truth[-1, :] = 1
        self.ground_truth[:, 0] = 1
        self.ground_truth[:, -1] = 1
        self.ground_truth[10:12, 0:25] = 1
        self.ground_truth[25:27, 15:40] = 1
        self.ground_truth[15:30, 20:22] = 1

    def scan_lidar(self):
        """模拟雷达扫描"""
        rx, ry = self.robot_pos
        r = self.sensor_range
        x_min, x_max = max(0, rx - r), min(self.map_size, rx + r + 1)
        y_min, y_max = max(0, ry - r), min(self.map_size, ry + r + 1)
        self.robot_map[x_min:x_max, y_min:y_max] = self.ground_truth[x_min:x_max, y_min:y_max]

    def get_frontiers(self):
        """核心逻辑：寻找已知空地与未知区域的交界处"""
        frontiers = []
        for x in range(1, self.map_size - 1):
            for y in range(1, self.map_size - 1):
                # 如果当前格子是已知的空地
                if self.robot_map[x, y] == 0:
                    # 检查上下左右是否有未知区域 (-1)
                    neighbors = [
                        self.robot_map[x+1, y], self.robot_map[x-1, y],
                        self.robot_map[x, y+1], self.robot_map[x, y-1]
                    ]
                    if -1 in neighbors:
                        frontiers.append([x, y])
        return np.array(frontiers)

    # def select_target(self, frontiers):
    #     """伪 RL 大脑：聚类并选出最近的目标"""
    #     if len(frontiers) < 3:
    #         return None, frontiers
            
    #     # 使用 K-Means 将碎片的边界聚类成几个大目标 (候选前沿点)
    #     k = min(5, len(frontiers))
    #     kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(frontiers)
    #     candidates = kmeans.cluster_centers_
        
    #     # 贪心策略：计算离小车最近的候选点
    #     distances = [np.linalg.norm(c - np.array(self.robot_pos)) for c in candidates]
    #     best_target = candidates[np.argmin(distances)]
        
    #     return best_target, candidates
    def select_target(self, frontiers):
        """带有死锁规避机制的决策大脑"""
        if len(frontiers) < 3:
            return None, frontiers
            
        k = min(5, len(frontiers))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(frontiers)
        candidates = kmeans.cluster_centers_
        
        # 1. 计算所有候选点到小车的距离
        distances = [np.linalg.norm(c - np.array(self.robot_pos)) for c in candidates]
        
        # 2. 获取按距离从小到大排序的索引集 (例如: [2, 0, 1, 4, 3])
        sorted_indices = np.argsort(distances)
        
        # 3. ========== 核心避障/降级逻辑 ==========
        if self.ignore_closest and len(candidates) > 1:
            # 触发了防卡死机制，且当前地图上还有超过1个候选点
            print("  ⚠️ [系统警告] 检测到局部死锁，放弃最近目标，强制切换至第二顺位！")
            # 选取欧式距离第二小的目标点
            best_target = candidates[sorted_indices[1]]
            # 切换完毕后，重置标志位，恢复正常的贪心探索
            self.ignore_closest = False 
        else:
            # 正常情况：无脑选择距离最小的目标点
            best_target = candidates[sorted_indices[0]]
            
        return best_target, candidates

    def render(self, frontiers, candidates, target):
        """绘制当前状态"""
        self.ax.clear()
        
        # 1. 画基础地图
        self.ax.imshow(self.robot_map, cmap=self.cmap, vmin=-1, vmax=1)
        
        # 2. 画所有的边界点 (蓝色小点)
        if len(frontiers) > 0:
            self.ax.scatter(frontiers[:, 1], frontiers[:, 0], c='cyan', s=10, label='Frontier Pixels')
            
        # 3. 画聚类后的候选点 (黄色大点)
        if candidates is not None:
            self.ax.scatter(candidates[:, 1], candidates[:, 0], c='orange', s=50, marker='X', label='Candidate Targets')
            
        # 4. 画最终决定的目标点 (绿色五角星)
        if target is not None:
            self.ax.scatter(target[1], target[0], c='lime', s=150, marker='*', edgecolors='black', label='Selected Target')
            
        # 5. 画小车当前位置 (红色圆圈)
        self.ax.scatter(self.robot_pos[1], self.robot_pos[0], c='red', s=80, marker='o', label='Robot')
        
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.set_title("Hierarchical RL: High-level Decision Making Demo")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # def run_demo(self):
    #     """运行模拟循环"""
    #     print("启动探索测试...")
    #     for step in range(50):
    #         # 1. 扫描环境
    #         self.scan_lidar()
            
    #         # 2. 提取边界
    #         frontiers = self.get_frontiers()
            
    #         if len(frontiers) == 0:
    #             print("地图探索完毕！")
    #             break
                
    #         # 3. 决策系统选出目标点
    #         target, candidates = self.select_target(frontiers)
            
    #         # 4. 可视化当前决策
    #         self.render(frontiers, candidates, target)
    #         time.sleep(0.5) # 暂停一下，让大家看清决策过程
            
    #         # 5. 模拟底层同学写的 A* 算法，小车直接移动到目标点附近
    #         if target is not None:
    #             # 简单粗暴的模拟移动：直接把坐标设过去 (加个约束防止跳出地图)
    #             self.robot_pos = [int(np.clip(target[0], 1, self.map_size-2)), 
    #                               int(np.clip(target[1], 1, self.map_size-2))]
    def run_demo(self):
        print("启动探索测试 (带防卡死机制)...")
        for step in range(80): # 增加一点测试步数
            self.scan_lidar()
            frontiers = self.get_frontiers()
            
            if len(frontiers) == 0:
                print("地图探索完毕！")
                break
                
            # ========== 卡死检测逻辑 ==========
            # 如果当前位置和上一步位置完全一样（说明没动，或者撞墙滑步了）
            if self.robot_pos == self.last_pos:
                self.stuck_patience += 1
            else:
                self.stuck_patience = 0 # 只要动了，就重新恢复耐心
                
            self.last_pos = list(self.robot_pos)

            # 连续 2 步卡在原地，说明这个目标点是个“幽灵边界”或者物理不可达
            if self.stuck_patience >= 2:
                self.ignore_closest = True
                self.stuck_patience = 0 # 触发降级后，耐心清零
            # ==================================
            
            target, candidates = self.select_target(frontiers)
            
            self.render(frontiers, candidates, target)
            time.sleep(0.3) 
            
            if target is not None:
                self.robot_pos = [int(np.clip(target[0], 1, self.map_size-2)), 
                                  int(np.clip(target[1], 1, self.map_size-2))]

if __name__ == "__main__":
    demo = FrontierExplorerDemo()
    demo.run_demo()
    # 保持窗口打开
    plt.ioff()
    plt.show()