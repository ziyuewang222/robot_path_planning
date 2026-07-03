import time
import matplotlib.pyplot as plt
import numpy as np
from explore_env import AutoExploreEnv
from stable_baselines3 import PPO

if __name__ == '__main__':
    print("正在加载 100x100 环境与 AI 模型...")
    
    # 1. 初始化与训练时完全一致的环境
    env = AutoExploreEnv(map_size=64, sensor_range=5)
    
    # 2. 加载训练好的“大脑”
    # 注意：这里的路径请根据你刚才实际生成的文件夹名字进行修改！
    # 如果你想看半成品，可以换成类似 "./models/cnn_100x100_v1/checkpoint_50000_steps"
    model_path = "./models/cnn_100x100_v7/checkpoint_100000_steps" 
    try:
        model = PPO.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"找不到模型文件，请检查路径是否正确: {model_path}")
        exit()
    
    # 3. 重置环境，获取第一眼雷达扫描的数据 (obs)
    obs, info = env.reset()

    # 4. 设置 Matplotlib 动态可视化
    plt.ion()
    plt.figure(figsize=(10, 10))

    step = 0
    while True:
        # 【核心推理】：把你现在的局部地图 obs 喂给模型
        # deterministic=False: 允许模型保留一点随机性（掷骰子），防止在墙角鬼畜振荡
        action, _states = model.predict(obs, deterministic=False)
        
        # 让环境执行这个动作，并返回新的地图状态
        obs, reward, terminated, truncated, info = env.step(action)
        
        # ================= 渲染可视化 =================
        plt.clf()
        
        # 处理一下矩阵颜色以便显示：未知为暗色，空地为亮色
        display_map = np.copy(env.robot_map).astype(float)
        display_map[env.robot_map == -1] = -50    # 迷雾 (深色)
        display_map[env.ground_truth == 1] = -100 # 真实的墙壁 (最深色)，方便你对比
        
        plt.imshow(display_map, cmap='viridis', origin='upper')
        
        # 画出小车当前位置
        plt.plot(env.robot_pos[1], env.robot_pos[0], 'ro', markersize=8, label="Robot (AI)")
        
        plt.title(f"AI Autonomous Exploration - Step {step} | Reward: {reward:.1f}")
        plt.legend(loc='upper right')
        
        # 暂停一小会儿，形成动画效果（如果嫌太慢可以改成 0.05）
        plt.pause(0.1)  
        # ==============================================
        
        # 检查是否通关（探索超 80%）或者被击毙（连续撞墙）
        if terminated or truncated:
            print(f"探索结束！总计步数: {step}")
            break
            
        step += 1

    # 保持最后一张图不关闭
    plt.ioff()
    plt.show()