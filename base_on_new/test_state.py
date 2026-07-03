import time
import matplotlib.pyplot as plt
from config import TRAIN_MAP_SIZE, SENSOR_RANGE, SIGMA
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline

def visualize_3_channels(state_tensor, step):
    """同时渲染 3 个通道，进行对账验证"""
    c0 = state_tensor[0].cpu().numpy()
    c1 = state_tensor[1].cpu().numpy()
    c2 = state_tensor[2].cpu().numpy()
    
    plt.clf()
    
    # 子图 1: Channel 0 占据栅格
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Ch 0: Explored Map")
    ax1.imshow(c0, cmap='viridis', origin='upper')
    
    # 子图 2: Channel 1 小车热力图
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Ch 1: Robot Gaussian")
    ax2.imshow(c1, cmap='magma', origin='upper')
    
    # 子图 3: Channel 2 前线掩码 Mask (这是动作掩码的核心!)
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Ch 2: Frontier Mask")
    ax3.imshow(c2, cmap='gray', origin='upper')
    
    plt.suptitle(f"State Space Tensor Visualization - Step {step}", fontsize=14)
    plt.tight_layout()
    plt.pause(0.3)

if __name__ == "__main__":
    env = GridWorldEnv(map_size=TRAIN_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=42)
    pipeline = StateTensorPipeline(map_size=TRAIN_MAP_SIZE, sigma=SIGMA)
    
    explored, pos = env.reset()
    
    plt.ion()
    plt.figure(figsize=(15, 5))
    
    # 模拟移动 20 步并实时观察通道张量的动态变化
    actions = [3]*10 + [1]*10
    
    for step, act in enumerate(actions):
        explored, pos, _ = env.step(act)
        
        # 核心：调用流水线转换出 Tensor
        state_tensor = pipeline.process(explored, pos)
        
        # 打印张量规格确认无误
        if step == 0:
            print(f"生成的张量类型: {type(state_tensor)}, 维度形状: {state_tensor.shape}")
            
        visualize_3_channels(state_tensor, step+1)
        
    print("测试完毕！请检查最右侧图 Ch2 是否精准高亮了白色与灰色的边缘交界线。窗口将保持 5 秒...")
    time.sleep(5)