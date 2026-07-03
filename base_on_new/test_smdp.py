import time
from config import TRAIN_MAP_SIZE, SENSOR_RANGE, SIGMA
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline
from models import SpatialFCN, SpatialActionDecision

if __name__ == "__main__":
    env = GridWorldEnv(map_size=TRAIN_MAP_SIZE, sensor_range=SENSOR_RANGE, seed=42)
    pipeline = StateTensorPipeline(map_size=TRAIN_MAP_SIZE, sigma=SIGMA)
    model = SpatialFCN()
    decision = SpatialActionDecision(model, epsilon=0.5) # 提高探索率以测试不同终点
    
    explored, pos = env.reset()
    
    print("\n--- 开始执行 SMDP 宏观动作闭环验证 ---")
    for macro_step in range(1, 11):
        # 1. 组装张量
        state = pipeline.process(explored, pos)
        
        # 2. 网络选点 (经过 Action Masking)
        target, _, _ = decision.select_action(state, explore=True)
        if target is None:
            print("没有可用前线，探索终止。")
            break
            
        # 3. 执行宏观动作，底层 A* 自动走到该点
        explored, pos, reward, n_steps, done, info = env.step_macro(target)
        
        print(f"高层决策 {macro_step:02d} | 目标坐标: {target} | 消耗步数(N_steps): {n_steps:2d} | 新增面积: {info['delta_area']:4.0f} | 物理回报(R_t): {reward:6.2f}")
        
        # 简单渲染看一下当前状态
        env.render(step_num=macro_step)
        
        if done:
            print("完成整个地图探索！")
            break
            
    time.sleep(2)