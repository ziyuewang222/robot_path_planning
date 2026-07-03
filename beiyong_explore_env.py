import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AutoExploreEnv(gym.Env):
    """自定义的机器人 64*64 大型复杂建图探索强化学习环境"""
    
    def __init__(self, map_size=64, sensor_range=5):
        super(AutoExploreEnv, self).__init__()
        
        self.map_size = map_size
        self.sensor_range = sensor_range
        
        # 定义动作空间 (Discrete: 0=上, 1=下, 2=左, 3=右)
        self.action_space = spaces.Discrete(4)
        
        # 定义观测空间 (Box): 严格符合 Stable-Baselines3 的 NatureCNN 图像标准
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(1, self.map_size, self.map_size), 
            dtype=np.uint8
        )
        
        self.ground_truth = None
        self.robot_map = None
        self.robot_pos = None
        
        # 【新增】：连续撞墙计数器
        self.consecutive_wall_hits = 0 

    def _get_obs(self):
        """将内部的物理逻辑矩阵转换为交给神经网络的标准灰度图"""
        obs = np.full((self.map_size, self.map_size), 127, dtype=np.uint8)
        obs[self.robot_map == 0] = 255
        obs[self.robot_map == 1] = 0
        return np.expand_dims(obs, axis=0)

    def _build_walls(self):
        """构建模拟真实室内环境的复杂拓扑地图"""
        self.ground_truth[0, :] = 1
        self.ground_truth[-1, :] = 1
        self.ground_truth[:, 0] = 1
        self.ground_truth[:, -1] = 1
        
        self.ground_truth[15:17, 0:45] = 1
        self.ground_truth[45:47, 15:60] = 1
        self.ground_truth[15:45, 30:32] = 1

        self.ground_truth[15:17, 10:15] = 0 
        self.ground_truth[15:17, 35:40] = 0 
        self.ground_truth[45:47, 45:50] = 0 
        self.ground_truth[25:30, 30:32] = 0 

        np.random.seed(42) 
        num_extra_obstacles = 25 
        for _ in range(num_extra_obstacles):
            x = np.random.randint(2, self.map_size - 6)
            y = np.random.randint(2, self.map_size - 6)
            if self.ground_truth[x, y] == 0:
                w = np.random.randint(2, 5) 
                h = np.random.randint(2, 5)
                x_end = min(x + w, self.map_size - 1)
                y_end = min(y + h, self.map_size - 1)
                self.ground_truth[x:x_end, y:y_end] = 1
       
    def _get_valid_pos(self):
        """在空地上随机生成出生点"""
        while True:
            x, y = np.random.randint(1, self.map_size-1, 2)
            if self.ground_truth[x, y] == 0:
                return [x, y]

    def scan_lidar(self):
        """模拟雷达扫描：更新机器人当前视野内的认知地图"""
        rx, ry = self.robot_pos
        r = self.sensor_range
        x_min, x_max = max(0, rx - r), min(self.map_size, rx + r + 1)
        y_min, y_max = max(0, ry - r), min(self.map_size, ry + r + 1)
        self.robot_map[x_min:x_max, y_min:y_max] = self.ground_truth[x_min:x_max, y_min:y_max]

    def reset(self, seed=None, options=None):
        """回合重置：初始化地图与小车位置"""
        super().reset(seed=seed)
        
        self.ground_truth = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self._build_walls()
        
        self.robot_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)
        self.robot_pos = self._get_valid_pos()
        
        # 【新增】：回合重置时，清零撞墙计数器
        self.consecutive_wall_hits = 0 
        
        self.scan_lidar()
        return self._get_obs(), {}

    def step(self, action):
        """核心物理与逻辑引擎：执行动作、计算合理奖惩"""
        dx, dy = 0, 0
        if action == 0: dx = -1   # 上
        elif action == 1: dx = 1  # 下
        elif action == 2: dy = -1 # 左
        elif action == 3: dy = 1  # 右
            
        new_x = self.robot_pos[0] + dx
        new_y = self.robot_pos[1] + dy
        
        reward = 0.0
        terminated = False 
        truncated = False  

        # --- 碰撞检测与惩罚机制 ---
        if self.ground_truth[new_x, new_y] == 1:
            # 撞墙后位置不变，并累加计数器
            self.consecutive_wall_hits += 1
            reward = -0.5 # 单次撞墙的基础惩罚
            
            # 【新增】：连续撞墙防卡死机制
            if self.consecutive_wall_hits >= 3:
                reward -= 50.0       # 附加毁灭性惩罚
                terminated = True    # 强制结束当前回合
                print(f"【环境通知】连续撞墙 {self.consecutive_wall_hits} 次，触发防卡死，回合结束！")
        else:
            # 正常移动，清零计数器
            self.consecutive_wall_hits = 0
            self.robot_pos = [new_x, new_y]
            reward = -0.1 
            
        # --- 计算信息增益奖励 ---
        old_explored = np.sum(self.robot_map != -1)
        self.scan_lidar() 
        new_explored = np.sum(self.robot_map != -1)
        
        new_cells = new_explored - old_explored
        if new_cells > 0:
            reward += new_cells * 1.0  
            
        # --- 终止条件：探索超过 80% 的全局安全区 ---
        total_safe_cells = np.sum(self.ground_truth == 0)
        current_safe_cells = np.sum(self.robot_map == 0)
        
        if current_safe_cells >= total_safe_cells * 0.8:
            reward += 100.0 
            terminated = True
            print("【环境通知】地图探索超过 80%，通关成功！")
            
        return self._get_obs(), reward, terminated, truncated, {}