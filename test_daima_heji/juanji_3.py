import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FogMapProcessor:
    def __init__(self, map_size=60):
        self.map_size = map_size
        # 初始化全为 -1 (全图迷雾)
        self.raw_map = np.full((self.map_size, self.map_size), -1, dtype=np.int32)
        
    def build_partial_map(self):
        """模拟机器人探索了一半的地图"""
        # 假设机器人只探索了左半边 (0 到 30 列)
        self.raw_map[:, 0:30] = 0  # 探索区域默认置为空地
        
        # 在已探索区域建墙
        self.raw_map[0, 0:30] = 1
        self.raw_map[-1, 0:30] = 1
        self.raw_map[:, 0] = 1
        self.raw_map[15:17, 0:25] = 1
        
        # 在已探索区域放置杂物
        self.raw_map[5:7, 5:7] = 1
        self.raw_map[20:23, 10:13] = 1

    def process_and_visualize(self):
        # === 1. 核心 SOP: 通道分离 (One-Hot Encoding) ===
        tensor_map = torch.tensor(self.raw_map, dtype=torch.long)
        
        # 将 -1, 0, 1 映射为索引: 0(迷雾), 1(空地), 2(墙壁)
        index_map = tensor_map + 1 
        
        # 生成 (1, 3, H, W) 的 One-Hot 张量，并将类型转为 float 以支持卷积
        one_hot_map = F.one_hot(index_map, num_classes=3).permute(2, 0, 1).unsqueeze(0).float()
        
        # 取出独立通道
        fog_channel = one_hot_map[:, 0:1, :, :]
        free_channel = one_hot_map[:, 1:2, :, :]
        wall_channel = one_hot_map[:, 2:3, :, :]

        # === 2. 独立池化 (保留降采样后的不确定性) ===
        wall_low = F.max_pool2d(wall_channel, kernel_size=4, stride=4)
        fog_low = F.max_pool2d(fog_channel, kernel_size=4, stride=4)

        # === 3. 提取不同物理意义的边界 ===
        edge_kernel = torch.tensor([[-1., -1., -1.],
                                    [-1.,  8., -1.],
                                    [-1., -1., -1.]]).view(1, 1, 3, 3)

        # 障碍物边界 (仅对墙壁通道卷积)
        wall_edges = F.conv2d(wall_channel, edge_kernel, padding=1)
        
        # 探索前线/边界 (仅对迷雾通道卷积，提取迷雾的轮廓，即下一步该去探索的地方)
        frontier_edges = F.conv2d(fog_channel, edge_kernel, padding=1)

        # 打印验证维度
        print(f"原图维度: {self.raw_map.shape}")
        print(f"分离后张量维度: {one_hot_map.shape}")
        
        return wall_edges, frontier_edges

if __name__ == "__main__":
    processor = FogMapProcessor(map_size=60)
    processor.build_partial_map()
    processor.process_and_visualize()