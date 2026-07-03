import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MapCNNProcessor:
    def __init__(self, map_size=60):
        self.map_size = map_size
        # 初始化全为迷雾 (-1)
        self.ground_truth = np.full((self.map_size, self.map_size), -1, dtype=np.int32)

    def build_and_visualize(self):
        """构建包含迷雾的半探索地图，并使用分组卷积输出 9 张独立特征图"""
        # 模拟机器人探索了左半边 (0 到 40 列)，默认置为空地 (0)
        self.ground_truth[:, 0:40] = 0

        # 1. 基础外围墙壁 (仅在已探索区域建墙)
        self.ground_truth[0, 0:40] = 1
        self.ground_truth[-1, 0:40] = 1
        self.ground_truth[:, 0] = 1

        # 2. 构建长墙隔断
        self.ground_truth[15:17, 0:35] = 1
        self.ground_truth[15:45, 30:32] = 1

        # 挖出门洞
        self.ground_truth[15:17, 10:15] = 0  
        self.ground_truth[25:30, 30:32] = 0  

        # 3. 随机散布较小的障碍物 (仅在已探索区域)
        np.random.seed(42)  
        num_extra_obstacles = 15  
        for _ in range(num_extra_obstacles):
            x = np.random.randint(2, self.map_size - 6)
            y = np.random.randint(2, 35)
            if self.ground_truth[x, y] == 0:
                w = np.random.randint(2, 5)
                h = np.random.randint(2, 5)
                self.ground_truth[x:x+w, y:y+h] = 1

        # === 三通道预处理 (分离迷雾、空地、墙壁) ===
        tensor_index = torch.tensor(self.ground_truth, dtype=torch.long)
        shifted_index = tensor_index + 1 
        tensor_map_3c = F.one_hot(shifted_index, num_classes=3).permute(2, 0, 1).unsqueeze(0).float()

        # === 核心修改: 适配分组卷积的 3 通道独立核 ===
        base_kernel = torch.tensor([[-1., -1., -1.],
                                    [-1.,  8., -1.],
                                    [-1., -1., -1.]])
        
        # 维度变为 (out_channels=3, in_channels/groups=1, H=3, W=3)
        edge_kernel_independent = base_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # === 逐级降采样 (MaxPool) 并执行分组卷积 (groups=3) ===
        
        # 1. 高分辨率特征 (输出维度: 1, 3, 60, 60)
        feat_high = F.conv2d(tensor_map_3c, edge_kernel_independent, padding=1, groups=3)

        # 2. 中分辨率特征 (输出维度: 1, 3, 30, 30)
        map_mid = F.max_pool2d(tensor_map_3c, kernel_size=2, stride=2)
        feat_mid = F.conv2d(map_mid, edge_kernel_independent, padding=1, groups=3)

        # 3. 低分辨率特征 (输出维度: 1, 3, 15, 15)
        map_low = F.max_pool2d(tensor_map_3c, kernel_size=4, stride=4)
        feat_low = F.conv2d(map_low, edge_kernel_independent, padding=1, groups=3)

        # === 执行全新可视化流程 ===
        self._plot_results(self.ground_truth, feat_high, feat_mid, feat_low)

    def _plot_results(self, original, high, mid, low):
        # 创建 3 行 4 列的画布 (第 1 列放原图，后 3 列放三种通道的特征图)
        fig, axes = plt.subplots(3, 4, figsize=(22, 16))
        
        # 绘制原始地图 (仅在左上角展示一次)
        axes[0, 0].imshow(original, cmap='viridis')
        axes[0, 0].set_title(f"Original Fog Map (-1,0,1)\n{original.shape}", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        features = [high, mid, low]
        row_names = ['High-Res (60x60)', 'Mid-Res (30x30)', 'Low-Res (15x15)']
        channel_names = ['Fog Channel (-1)', 'Free Channel (0)', 'Wall Channel (1)']

        # 遍历 3 个尺度
        for i, feat in enumerate(features):
            # 压缩 batch 维度，保留 (3, H, W)，并取绝对值
            feat_img = np.abs(feat.squeeze().numpy()) 
            
            # 遍历 3 个物理通道
            for j in range(3):
                ax = axes[i, j+1]
                ax.imshow(feat_img[j], cmap='hot')
                ax.set_title(f"[{row_names[i]}]\n{channel_names[j]} Edges", fontsize=12)
                ax.axis('off')

        # 添加全局大标题
        fig.suptitle("Independent Grouped Convolution on 3-Channel Fog Map", fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

# 运行测试
if __name__ == "__main__":
    processor = MapCNNProcessor(map_size=60)
    processor.build_and_visualize()