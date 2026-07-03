import time
import matplotlib.pyplot as plt
from config import TRAIN_MAP_SIZE, SENSOR_RANGE
from grid_env import GridWorldEnv

if __name__ == "__main__":
    env = GridWorldEnv(map_size=TRAIN_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=42)
    explored, pos = env.reset()
    
    print("正在启动 Matplotlib 图形界面...")
    env.render(step_num=0)
    
    # 让小车走一段折线，展示开荒效果
    actions = [3]*20 + [1]*25 + [2]*15 + [0]*10
    
    for step, act in enumerate(actions):
        explored, pos, collision = env.step(act)
        # 渲染可视界面
        env.render(step_num=step+1)
        
    print("测试完毕！保持窗口打开 3 秒...")
    time.sleep(3)
# import time
# import os
# from grid_env import GridWorldEnv

# def print_map(explored, pos):
#     """终端控制台轻量级打印地图"""
#     os.system('cls' if os.name == 'nt' else 'clear')
#     H, W = explored.shape
#     # 为了防止控制台换行，只打印部分区域或进行简单的字符映射
#     display_str = ""
#     for i in range(0, H, 2):  # 隔行采样以适应屏幕高宽比
#         for j in range(0, W, 2):
#             if [i, j] == pos or [i+1, j] == pos:
#                 display_str += "🤖"
#             elif explored[i, j] == -1:
#                 display_str += "░░" # 迷雾
#             elif explored[i, j] == 1:
#                 display_str += "██" # 墙壁
#             else:
#                 display_str += "  " # 已知空地
#         display_str += "\n"
#     print(display_str)
#     print(f"机器人坐标: {pos}")

# if __name__ == "__main__":
#     env = GridWorldEnv(map_size=100, sensor_range=12, seed=10)
#     explored, pos = env.reset()
    
#     print("=== 初始化成功，正在测试移动和透雾感知 ===")
#     print_map(explored, pos)
#     time.sleep(1)
    
#     # 模拟让小车连续向右跑 15 步 (动作 3)
#     for step in range(15):
#         explored, pos, collision = env.step(action=3)
#         print_map(explored, pos)
#         if collision:
#             print("⚠️ 撞到了障碍物！")
#         time.sleep(0.15)