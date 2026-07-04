import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import (
    TEST_MAP_SIZE,
    TEST_SEED,
    SENSOR_RANGE,
    SIGMA,
    MAP_GENERATOR,
    max_macro_steps_for,
    CHECKPOINTS_DIR,
    LATEST_CHECKPOINT_PATH,
)
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline
from models import SpatialFCN, SpatialActionDecision

class FullScreenVisualizer:
    def __init__(self, map_size):
        """
        初始化 2x3 全息网格可视化控制台
        [0,0] 主视窗：小车开荒轨迹与目标点    [0,1] Ch0: 占据栅格地图       [0,2] Ch2: 前线掩码 Mask
        [1,0] Q_raw: FCN 全图原始推测值       [1,1] Ch1: 小车高斯热力图     [1,2] Q_masked: 最终决策合法候选池
        """
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Spatial FCN RL - Autonomous Exploration Live Dashboard")
        
        # 子图句柄字典
        self.im_plots = {}
        self.target_plots = {}
        self.robot_plots = {}
        
        titles = [
            ["Main Exploration View", "Ch 0: Explored Grid Map", "Ch 2: Frontier Mask (Mask_t)"],
            ["Neural Prediction Q_raw", "Ch 1: Robot Gaussian Heatmap", "Final Decision Pool Q_masked"]
        ]
        cmaps = [
            ['viridis', 'viridis', 'gray'],
            ['plasma', 'magma', 'plasma']
        ]
        
        for r in range(2):
            for c in range(3):
                ax = self.axes[r, c]
                ax.set_title(titles[r][c], fontsize=11, fontweight='bold')
                # 初始化空图像，稍后在更新循环里更新数据
                im = ax.imshow(np.zeros((map_size, map_size)), cmap=cmaps[r][c], origin='upper')
                self.im_plots[(r, c)] = im
                
                # 为主图和最终决策池添加标注星号
                if (r, c) in [(0, 0), (1, 2)]:
                    star, = ax.plot([], [], 'r*', markersize=12, label='Target')
                    self.target_plots[(r, c)] = star
                if (r, c) == (0, 0):
                    robot, = ax.plot([], [], 'ro', markersize=8, label='Robot')
                    self.robot_plots[(r, c)] = robot
                    ax.legend(loc='upper right')
                    
        plt.tight_layout()

    def update(self, step_num, explored, pos, state_tensor, q_raw, q_masked, chosen_coord):
        """实时刷新所有 6 个窗口的张量数据"""
        c0 = state_tensor[0].cpu().numpy()
        c1 = state_tensor[1].cpu().numpy()
        c2 = state_tensor[2].cpu().numpy()
        
        # 1. 主渲染视窗数据转换
        display_grid = np.zeros_like(explored, dtype=np.float32)
        display_grid[explored == -1] = 0.5  # 灰色迷雾
        display_grid[explored == 0] = 1.0   # 黄色空地
        display_grid[explored == 1] = 0.0   # 深紫障碍
        
        # 2. 对 Q_masked 进行可视化处理（把负无穷替换为 NaN 便于清晰渲染边缘）
        q_masked_vis = q_masked.copy()
        q_masked_vis[q_masked_vis == -1e9] = np.nan
        
        # 3. 更新各个图像矩阵
        data_map = {
            (0, 0): display_grid,
            (0, 1): c0,
            (0, 2): c2,
            (1, 0): q_raw,
            (1, 1): c1,
            (1, 2): q_masked_vis
        }
        
        for key, im in self.im_plots.items():
            im.set_data(data_map[key])
            im.autoscale() # 动态缩放色彩范围
            
        # 4. 更新坐标标注
        if chosen_coord:
            for star in self.target_plots.values():
                star.set_data([chosen_coord[1]], [chosen_coord[0]])
        self.robot_plots[(0, 0)].set_data([pos[1]], [pos[0]])
        
        self.fig.suptitle(f"Autonomous Navigation Inference | Macro Step: {step_num:02d} | Robot Pos: {pos}", fontsize=14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)


def evaluate_model(weight_path):
    print(f"--> 正在准备载入模型策略权重: {weight_path}")
    if not os.path.exists(weight_path):
        print(f"错误：找不到权重文件 {weight_path}，请确认路径或训练是否成功生成文件！")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 实例化网络并加载权重
    policy_net = SpatialFCN().to(device)
    state_dict = torch.load(weight_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    
    # 极为关键：开启评估模式！
    policy_net.eval()
    print("--> 策略网络权重加载完毕，已切换至推理模式 (eval)！")
    
    # 2. 初始化测试环境与决策器 (epsilon设为0.0，关闭随机探索)
    map_size = TEST_MAP_SIZE
    env = GridWorldEnv(
        map_size=map_size,
        sensor_range=SENSOR_RANGE,
        seed=TEST_SEED,
        map_generator=MAP_GENERATOR,
    )
    pipeline = StateTensorPipeline(map_size=map_size, sigma=SIGMA, device=device)
    decision = SpatialActionDecision(policy_net, epsilon=0.0, device=device)

    vis = FullScreenVisualizer(map_size=map_size)
    
    explored, pos = env.reset()
    
    print("=== 启动实时推理演练演示 ===")
    total_steps = 0
    total_area = 0.0
    
    with torch.no_grad(): # 全程关闭梯度流，速度极快
        for step in range(1, max_macro_steps_for(map_size) + 1):
            # 转换当前状态为 3xHxW Tensor
            state_tensor = pipeline.process(explored, pos)
            
            # 模型选点 (纯贪心)
            chosen_coord, q_masked, q_raw = decision.select_action(state_tensor, explore=False)
            
            if chosen_coord is None:
                print("未发现有效探索边界，地图探索完成！")
                break
                
            # 刷新 6 宫格全息图
            vis.update(step, explored, pos, state_tensor, q_raw, q_masked, chosen_coord)
            
            # 执行底层 A* 闭环运动
            next_explored, next_pos, _reward, n_steps, done, info = env.step_macro(chosen_coord)
            
            total_steps += n_steps
            total_area += info['delta_area']
            print(f"执行宏观决策 {step:02d} | 奔赴目标: {chosen_coord} | 物理消耗步数: {n_steps:2d} | 获得面积收益: {info['delta_area']:4.0f}")
            
            explored = next_explored
            pos = next_pos
            
            if done:
                print("\n🎉 成功达成 95% 以上区域开荒！探索完结！")
                break
                
    print(f"评估总结 -> 累计物理步数: {total_steps} | 累计探明新区域: {total_area:.0f}")
    print("窗口将保留 10 秒供观察...")
    time.sleep(10)

if __name__ == "__main__":
    # 优先使用稳定的 latest 权重；如果它不存在，再退回到某个具体 checkpoint
    target_weights = str(LATEST_CHECKPOINT_PATH)
    if not os.path.exists(target_weights):
        target_weights = str(CHECKPOINTS_DIR / "fcn_rl_ep050.pth")
    evaluate_model(target_weights)