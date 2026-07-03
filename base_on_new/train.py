import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from config import TRAIN_MAP_SIZE, TRAIN_MAX_MACRO_STEPS, SENSOR_RANGE, SIGMA
from grid_env import GridWorldEnv
from state_pipeline import StateTensorPipeline
from models import SpatialFCN, SpatialActionDecision

class SMDPReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_coord, reward, next_state, next_mask, done, tau_steps):
        """
        存入 SMDP 宏观经验转换
        :param tau_steps: 底层控制消耗的物理时间步 N_steps
        """
        self.buffer.append((state, action_coord, reward, next_state, next_mask, done, tau_steps))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class SpatialQTrainer:
    def __init__(self, lr=1e-4, gamma=0.99, buffer_size=2000, device='cpu'):
        self.device = device
        self.gamma = gamma
        
        # 1. 初始化双网络架构
        self.policy_net = SpatialFCN().to(device)
        self.target_net = SpatialFCN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = SMDPReplayBuffer(capacity=buffer_size)

    def update_step(self, batch_size=16):
        if len(self.memory) < batch_size:
            return 0.0
            
        transitions = self.memory.sample(batch_size)
        loss_val = 0.0
        
        self.optimizer.zero_grad()
        
        # 由于输入特征图大小可能存在微调，且采用单点坐标稀疏监督，遍历处理 Batch
        for state, action_coord, reward, next_state, next_mask, done, tau in transitions:
            state = state.unsqueeze(0).to(self.device)
            next_state = next_state.unsqueeze(0).to(self.device)
            next_mask = next_mask.to(self.device)
            
            # 1. 前向传播：计算预测 Q 值
            q_raw = self.policy_net(state).squeeze(0).squeeze(0)
            pred_q = q_raw[action_coord[0], action_coord[1]]
            
            # 2. 计算 SMDP TD Target (引入时间折现修正 \gamma^\tau)
            with torch.no_grad():
                if done:
                    target_q = torch.tensor(reward, dtype=torch.float32, device=self.device)
                else:
                    q_next_raw = self.target_net(next_state).squeeze(0).squeeze(0)
                    # 动作掩码过滤
                    q_next_masked = q_next_raw.clone()
                    q_next_masked[next_mask < 0.5] = -1e9
                    
                    max_next_q = torch.max(q_next_masked)
                    # SMDP 时间折现公式：y_t = R_t + \gamma^\tau * max(Q_next)
                    gamma_tau = self.gamma ** tau
                    target_q = reward + gamma_tau * max_next_q
            
            # 3. 累加稀疏单点 MSE 损失
            loss = self.criterion(pred_q, target_q)
            loss.backward()
            loss_val += loss.item()

        # 梯度裁剪防止稀疏爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss_val / batch_size

    def soft_update_target(self, tau=0.01):
        """软更新目标网络权重"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


def train_main():
    print("=== 正在启动全全卷积空间强化学习 (Spatial FCN RL) 仿真训练引擎 ===")
    
    # 基础超参数配置
    map_size = TRAIN_MAP_SIZE
    max_episodes = 50
    max_macro_steps_per_epi = TRAIN_MAX_MACRO_STEPS
    batch_size = 16
    epsilon_start = 0.8
    epsilon_end = 0.1
    epsilon_decay = 0.95
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的底层计算加速设备: {device}")

    env = GridWorldEnv(map_size=map_size, sensor_range=SENSOR_RANGE)
    pipeline = StateTensorPipeline(map_size=map_size, sigma=SIGMA, device=device)


    trainer = SpatialQTrainer(lr=3e-4, gamma=0.98, device=device)
    
    # 创建模型权重保存目录
    os.makedirs("weights", exist_ok=True)
    
    epsilon = epsilon_start
    
    for episode in range(1, max_episodes + 1):
        explored, pos = env.reset()
        state = pipeline.process(explored, pos)
        
        decision = SpatialActionDecision(trainer.policy_net, epsilon=epsilon, device=device)
        
        total_reward = 0.0
        total_loss = 0.0
        update_cnt = 0
        
        for step in range(1, max_macro_steps_per_epi + 1):
            # 1. 结合当前动作掩码选择目标终点
            action_coord, _, _ = decision.select_action(state, explore=True)
            
            if action_coord is None:
                # 找不到合法前线点，提前退出回合
                break
                
            # 2. 执行宏观动作，底层 A* 闭环导航
            next_explored, next_pos, reward, tau_steps, done, _ = env.step_macro(action_coord)
            next_state = pipeline.process(next_explored, next_pos)
            
            # 提取下一帧的前线掩码 Ch2 作为下一帧的 Mask
            next_mask = next_state[2]
            
            # 3. 存入 SMDP 经验池
            trainer.memory.push(
                state.cpu(), 
                action_coord, 
                reward, 
                next_state.cpu(), 
                next_mask.cpu(), 
                done, 
                tau_steps
            )
            
            # 4. 执行反向传播与权重复用更新
            loss = trainer.update_step(batch_size=batch_size)
            trainer.soft_update_target(tau=0.02)
            
            state = next_state
            total_reward += reward
            if loss > 0:
                total_loss += loss
                update_cnt += 1
                
            # 每隔 10 步图形化渲染一次（或取消注释看实时展示）
            # env.render(step_num=step)
            
            if done:
                break
                
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        avg_loss = (total_loss / update_cnt) if update_cnt > 0 else 0.0
        
        print(f"Episode [{episode:02d}/{max_episodes:02d}] | 累积物理回报: {total_reward:7.2f} | 探索率(eps): {epsilon:.3f} | 平均 MSE 损失: {avg_loss:.4f} | 回合步数: {step:2d}")
        
        # 定期保存权重文件
        if episode % 10 == 0:
            save_path = f"weights/fcn_rl_ep{episode}.pth"
            torch.save(trainer.policy_net.state_dict(), save_path)
            print(f"--> 模型权重已成功归档至: {save_path}")

if __name__ == "__main__":
    train_main()