import time
import matplotlib.pyplot as plt
from config import TRAIN_MAP_SIZE, SENSOR_RANGE, MAP_GENERATOR
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline

def visualize_5_channels(state_tensor, step):
    """同时渲染 5 个通道，进行对账验证"""
    channels = [state_tensor[i].cpu().numpy() for i in range(5)]
    titles = [
        "Ch 0: Unknown-side Frontier",
        "Ch 1: Free Mask",
        "Ch 2: Obstacle Mask",
        "Ch 3: BFS Reachable Distance",
        "Ch 4: Free-side Frontier",
    ]
    cmaps = ["gray", "gray", "gray", "viridis", "gray"]
    
    plt.clf()
    for idx, (channel, title, cmap) in enumerate(zip(channels, titles, cmaps), start=1):
        ax = plt.subplot(1, 5, idx)
        ax.set_title(title)
        ax.imshow(channel, cmap=cmap, origin='upper')
    
    plt.suptitle(f"State Space Tensor Visualization - Step {step}", fontsize=14)
    plt.tight_layout()
    plt.pause(0.3)

if __name__ == "__main__":
    env = GridWorldEnv(
        map_size=TRAIN_MAP_SIZE,
        sensor_range=SENSOR_RANGE,
        seed=42,
        map_generator=MAP_GENERATOR,
    )
    pipeline = StateTensorPipeline(map_size=TRAIN_MAP_SIZE)
    
    explored, pos = env.reset()
    
    plt.ion()
    plt.figure(figsize=(20, 5))
    
    # 模拟移动 20 步并实时观察通道张量的动态变化
    actions = [3]*10 + [1]*10
    
    for step, act in enumerate(actions):
        explored, pos, _ = env.step(act)
        
        # 核心：调用流水线转换出 Tensor
        state_tensor = pipeline.process(explored, pos)
        
        # 打印张量规格确认无误
        if step == 0:
            print(f"生成的张量类型: {type(state_tensor)}, 维度形状: {state_tensor.shape}")
            
        visualize_5_channels(state_tensor, step+1)
        
    print("测试完毕！请检查 Ch0 是否只显示 0/-1 边界的 -1 侧，Ch3 是否只在已知可达空地中呈现 BFS 距离场，Ch4 是否只显示 0/-1 边界的 0 侧。窗口将保持 5 秒...")
    time.sleep(5)