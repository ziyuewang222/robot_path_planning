import os
from explore_env import AutoExploreEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

if __name__ == '__main__':
    # ================= 实验配置 =================
    EXP_NAME = "cnn_100x100_v7"  # 每次修改奖励公式或地图，请更改这个名字 (如 v2, v3)
    SAVE_DIR = f"./models/{EXP_NAME}/"
    LOG_DIR = f"./logs/{EXP_NAME}/"
    
    # 自动创建管理文件夹
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # ============================================

    print(f"1. 正在初始化 100x100 的大型复杂探索环境...")
    env = AutoExploreEnv(map_size=64, sensor_range=5) 

    print("2. 正在构建 PPO 深度神经网络 (带 TensorBoard 日志)...")
    # 开启 tensorboard_log 以便后续可视化训练曲线
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR) 

    print(f"3. 配置自动存档: 每 50000 步存档至 {SAVE_DIR}")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=SAVE_DIR, 
        name_prefix='checkpoint'
    )

    print("4. 开始脱机训练！(按 Ctrl+C 可随时安全中断)")
    # 增加训练步数至 300,000，大地图需要更多试错
    model.learn(total_timesteps=300000, callback=checkpoint_callback) 

    print("5. 训练完全结束，保存最终模型！")
    model.save(os.path.join(SAVE_DIR, "final_model"))