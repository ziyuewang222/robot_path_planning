import torch
import torch.nn as nn
import numpy as np

class SpatialFCN(nn.Module):
    def __init__(self, in_channels=3):
        """
        纯全卷积空间价值评估网络 (Spatial Fully Convolutional Network)
        输入: (Batch, 3, H, W) -> 输出: (Batch, 1, H, W)
        严禁使用 Linear 层或 Pooling 层（防止尺寸变化）
        """
        super().__init__()
        
        # 使用 padding='same' (即 padding=1 配合 kernel_size=3) 确保特图尺寸恒等
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 最后一层把多通道特征压缩为单通道 Q 值评估图 Q_raw
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x):
        """
        :param x: torch.Tensor, (B, 3, H, W)
        :return: torch.Tensor, (B, 1, H, W)
        """
        return self.net(x)


class SpatialActionDecision:
    def __init__(self, model, epsilon=0.1, device='cpu'):
        """
        动作掩码截断与决策执行类
        :param model: 训练好的 SpatialFCN 模型
        :param epsilon: 随机探索概率
        :param device: 运行设备
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state_tensor, explore=True):
        """
        执行动作掩码截断公式并选出最终导航终点 (i^*, j^*)
        :param state_tensor: (3, H, W) 单样本张量
        :param explore: 是否开启 epsilon-greedy 探索
        :return: action_coord (x, y), q_masked_map (H, W), raw_q_map (H, W)
        """
        self.model.eval()
        
        # 增加 Batch 维度 -> (1, 3, H, W)
        state_batch = state_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. 神经软推理：得到全图原始 Q 值
            q_raw = self.model(state_batch).squeeze(0).squeeze(0) # (H, W)
            
        # 2. 提取当前合法前线掩码 Ch2
        mask = state_tensor[2].to(self.device) # (H, W)
        
        # 3. 几何硬过滤：动作掩码截断 (Action Masking)
        # 将 Mask == 0 (非法区域/盲区) 强制覆盖为负无穷 (-1e9)
        q_masked = q_raw.clone()
        q_masked[mask < 0.5] = -1e9
        
        # 检查当前地图是否还有合法前线点
        valid_indices = torch.nonzero(mask > 0.5) # 形状: (N, 2)
        
        if len(valid_indices) == 0:
            # 极端情况：无前线点（探索完成或死锁），返回小车原地坐标作为兜底
            return None, q_masked.cpu().numpy(), q_raw.cpu().numpy()

        # 4. 决策机制：epsilon-贪心探索
        if explore and np.random.rand() < self.epsilon:
            # 必须且仅仅在当前合法前线候选池中随机抽取！绝不在盲区乱点！
            random_idx = np.random.randint(0, len(valid_indices))
            chosen_coord = valid_indices[random_idx].tolist()
        else:
            # 贪心选择 Q_masked 最大值坐标 argmax
            flat_idx = torch.argmax(q_masked).item()
            H, W = q_masked.shape
            chosen_coord = [flat_idx // W, flat_idx % W]
            
        return chosen_coord, q_masked.cpu().numpy(), q_raw.cpu().numpy()