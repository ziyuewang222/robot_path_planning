import numpy as np
import torch
import torch.nn.functional as F

from config import TRAIN_MAP_SIZE


class StateTensorPipeline:
    def __init__(self, map_size=TRAIN_MAP_SIZE, sigma=None, device='cpu'):
        """
        状态张量处理流水线：生成 3xHxW 的 PyTorch Tensor
        :param map_size: 地图边长
        :param sigma: 兼容旧接口，当前 Channel 1 使用归一化曼哈顿距离热力图，不再使用 sigma
        :param device: 计算设备 ('cpu' 或 'cuda')
        """
        self.map_size = map_size
        self.device = device
        
        # 提前生成坐标网格矩阵，避免每一次 step 都重复创建，大幅提高计算速度
        grid_y, grid_x = np.meshgrid(np.arange(map_size), np.arange(map_size))
        self.grid_x = torch.tensor(grid_x, dtype=torch.float32, device=device)
        self.grid_y = torch.tensor(grid_y, dtype=torch.float32, device=device)
        
        # 定义用于提取前置边界 (Frontier) 的拉普拉斯/邻域检测卷积核 (上下左右4邻域)
        # 当中心点为空地且周围有迷雾时，卷积结果能精准识别
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.float32)
        self.conv_kernel = torch.tensor(kernel, device=device).unsqueeze(0).unsqueeze(0)

    def process(self, explored_map, robot_pos):
        """
        将原始地图和机器人坐标转化为 3 个通道的特征张量
        :param explored_map: np.ndarray, 形状 (H, W), 元素为 -1, 0, 1
        :param robot_pos: list or tuple [rx, ry]
        :return: torch.Tensor, 形状 (3, H, W)
        """
        rx, ry = robot_pos
        
        # --- Channel 0: 探索认知地图 ---
        c0 = torch.tensor(explored_map, dtype=torch.float32, device=self.device)
        
        # --- Channel 1: 基于曼哈顿距离的归一化小车热力图 ---
        manhattan_dist = torch.abs(self.grid_x - rx) + torch.abs(self.grid_y - ry)
        max_manhattan_dist = 2 * (self.map_size - 1)
        c1 = 1.0 - torch.clamp(manhattan_dist / max_manhattan_dist, 0.0, 1.0)
        
        # --- Channel 2: 前线边界掩码 Mask (Frontier Detection) ---
        c2_mask = self._extract_frontier_mask(c0)
        
        # 将 3 个通道堆叠拼接到一起 -> (3, H, W)
        state_tensor = torch.stack([c0, c1, c2_mask], dim=0)
        
        return state_tensor

    def _extract_frontier_mask(self, c0):
        """
        提取合法探索前线：自身为已知空地(0)，且相邻元素包含未知迷雾(-1)
        """
        # 1. 找出所有的未知迷雾位置
        is_unknown = (c0 == -1.0).float().unsqueeze(0).unsqueeze(0)
        
        # 2. 使用卷积计算每个点周围 4 邻域中“迷雾”的数量
        # padding=1 保证输出尺寸依然是 HxW
        unknown_neighbors = F.conv2d(is_unknown, self.conv_kernel, padding=1).squeeze()
        
        # 3. 前线判定条件：自身当前是空地 (c0 == 0.0) 且 周围至少有 1 个迷雾格子 (unknown_neighbors > 0)
        is_free = (c0 == 0.0)
        frontier_mask = (is_free & (unknown_neighbors > 0.5)).float()
        
        return frontier_mask