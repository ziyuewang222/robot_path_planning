import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MapCNNProcessor:
    def __init__(self, map_size=60):
        self.map_size = map_size
        self.ground_truth = np.zeros((self.map_size, self.map_size))

    def build_and_visualize(self):
        """构建模拟真实室内环境的复杂地图并执行多尺度卷积可视化"""
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

        # === 卷积操作 ===
        tensor_map = torch.tensor(self.ground_truth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        edge_kernel = torch.tensor([[-1., -1., -1.],
                                    [-1.,  8., -1.],
                                    [-1., -1., -1.]]).view(1, 1, 3, 3)

        feat_high = F.conv2d(tensor_map, edge_kernel, padding=1)

        map_mid = F.max_pool2d(tensor_map, kernel_size=2, stride=2)
        feat_mid = F.conv2d(map_mid, edge_kernel, padding=1)

        map_low = F.max_pool2d(tensor_map, kernel_size=4, stride=4)
        feat_low = F.conv2d(map_low, edge_kernel, padding=1)

        # === 可视化流程 ===
        self._plot_results(tensor_map, feat_high, feat_mid, feat_low)

    def _plot_results(self, original, high, mid, low):
        # 降维并取绝对值（边缘检测的拉普拉斯算子会产生负值，取绝对值便于观察边界强响应）
        orig_img = original.squeeze().numpy()
        high_img = np.abs(high.squeeze().numpy())
        mid_img = np.abs(mid.squeeze().numpy())
        low_img = np.abs(low.squeeze().numpy())

        # 创建 1x4 的子图画布
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 绘制原始地图
        axes[0].imshow(orig_img, cmap='gray_r') # gray_r 反转颜色，使墙壁(1)为深色，空地(0)为浅色
        axes[0].set_title(f"Original Map\n{orig_img.shape}")

        # 绘制高分辨率特征
        axes[1].imshow(high_img, cmap='hot')
        axes[1].set_title(f"High-Res Edges\n{high_img.shape}")

        # 绘制中分辨率特征
        axes[2].imshow(mid_img, cmap='hot')
        axes[2].set_title(f"Mid-Res Edges (Max Pool 2x2)\n{mid_img.shape}")

        # 绘制低分辨率特征
        axes[3].imshow(low_img, cmap='hot')
        axes[3].set_title(f"Low-Res Edges (Max Pool 4x4)\n{low_img.shape}")

        # 关闭坐标轴刻度线
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

# 运行测试
if __name__ == "__main__":
    processor = MapCNNProcessor(map_size=60)
    processor.build_and_visualize()

