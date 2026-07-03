import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MapCNNProcessor:
    def __init__(self, map_size=60):
        self.map_size = map_size
        self.ground_truth = np.zeros((self.map_size, self.map_size))

    def build_and_visualize(self):
        """构建模拟真实室内环境的复杂地图并执行多尺度卷积与特征融合可视化"""
        # 1. 基础外围墙壁
        self.ground_truth[0, :] = 1
        self.ground_truth[-1, :] = 1
        self.ground_truth[:, 0] = 1
        self.ground_truth[:, -1] = 1

        # 2. 构建长墙隔断
        self.ground_truth[15:17, 0:45] = 1
        self.ground_truth[45:47, 15:60] = 1
        self.ground_truth[15:45, 30:32] = 1

        # 挖出门洞
        self.ground_truth[15:17, 10:15] = 0  
        self.ground_truth[15:17, 35:40] = 0  
        self.ground_truth[45:47, 45:50] = 0  
        self.ground_truth[25:30, 30:32] = 0  

        # 3. 随机散布较小的障碍物
        np.random.seed(42)  
        num_extra_obstacles = 25  
        for _ in range(num_extra_obstacles):
            x = np.random.randint(2, self.map_size - 6)
            y = np.random.randint(2, self.map_size - 6)
            if self.ground_truth[x, y] == 0:
                w = np.random.randint(2, 5)
                h = np.random.randint(2, 5)
                x_end = min(x + w, self.map_size - 1)
                y_end = min(y + h, self.map_size - 1)
                self.ground_truth[x:x_end, y:y_end] = 1

        # === 核心 SOP 1: 离散标签的语义通道分离 (双通道 One-Hot) ===
        tensor_index = torch.tensor(self.ground_truth, dtype=torch.long)
        # 生成维度: (1, 2, H, W). 通道0为空地掩码，通道1为障碍物掩码
        one_hot_map = F.one_hot(tensor_index, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
        
        # 提取目标障碍物通道作为特征输入
        wall_channel = one_hot_map[:, 1:2, :, :]

        # === 核心 SOP 2: 定义基础算子与并行空洞卷积 ===
        # 算子：经典拉普拉斯边缘检测算子
        laplacian_kernel = torch.tensor([[-1., -1., -1.],
                                         [-1.,  8., -1.],
                                         [-1., -1., -1.]]).view(1, 1, 3, 3)

        # 动态调配 Padding 确保输出长宽均与输入对齐 (保持 60x60)
        # 关系公式: Padding = Dilation * (Kernel_Size - 1) // 2
        
        # 分支 1: 标准局部核 (Dilation=1, Padding=1) -> 实际感受野 3x3
        feat_d1 = F.conv2d(wall_channel, laplacian_kernel, padding=1, dilation=1)

        # 分支 2: 中尺度空洞核 (Dilation=2, Padding=2) -> 实际感受野 5x5
        feat_d2 = F.conv2d(wall_channel, laplacian_kernel, padding=2, dilation=2)

        # 分支 3: 大尺度空洞核 (Dilation=4, Padding=4) -> 实际感受野 9x9
        feat_d4 = F.conv2d(wall_channel, laplacian_kernel, padding=4, dilation=4)

        # === 核心 SOP 3: 特征级联拼接与 1x1 卷积融合 ===
        # 在通道维度拼接，张量维度变为 (1, 3, 60, 60)
        multi_scale_tensor = torch.cat([feat_d1, feat_d2, feat_d4], dim=1)
        
        # 定义 1x1 卷积权重，形状为 (输出通道, 输入通道, H, W) -> (1, 3, 1, 1)
        # 这里初始化为均值加权，模拟无损特征整合
        fusion_kernel = torch.tensor([[[[1/3]], [[1/3]], [[1/3]]]]).float()
        fused_feature = F.conv2d(multi_scale_tensor, fusion_kernel, padding=0)

        # === 可视化流程 ===
        self._plot_results(feat_d1, feat_d2, feat_d4, fused_feature)

    def _plot_results(self, d1, d2, d4, fused):
        # 边缘响应取绝对值以增强可视化对比度
        d1_img = np.abs(d1.squeeze().numpy())
        d2_img = np.abs(d2.squeeze().numpy())
        d4_img = np.abs(d4.squeeze().numpy())
        fused_img = np.abs(fused.squeeze().numpy())

        # 创建 1x5 的子图画布
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        # 原始地图环境
        axes[0].imshow(self.ground_truth, cmap='gray_r')
        axes[0].set_title(f"Original Ground Truth\n{self.ground_truth.shape}")

        # 密集局部特征
        axes[1].imshow(d1_img, cmap='hot')
        axes[1].set_title(f"Dilation=1 (RF: 3x3)\n{d1_img.shape}")

        # 中量级环境特征
        axes[2].imshow(d2_img, cmap='hot')
        axes[2].set_title(f"Dilation=2 (RF: 5x5)\n{d2_img.shape}")

        # 宏观趋势特征（可以观察到大尺度下对门洞及连续障碍的整体响应）
        axes[3].imshow(d4_img, cmap='hot')
        axes[3].set_title(f"Dilation=4 (RF: 9x9)\n{d4_img.shape}")

        # 跨通道融合后的特征图
        axes[4].imshow(fused_img, cmap='jet')
        axes[4].set_title(f"1x1 Fused Feature\n{fused_img.shape}")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    processor = MapCNNProcessor(map_size=60)
    processor.build_and_visualize()