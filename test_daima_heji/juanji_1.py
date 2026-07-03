import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 地图生成系统 (复用你的逻辑)
# ==========================================
class MapGenerator:
    def __init__(self, map_size=60):
        self.map_size = map_size
        self.ground_truth = np.zeros((map_size, map_size))
        self._build_map()

    def _build_map(self):
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

        # 3. 随机散布障碍物
        np.random.seed(42)
        num_extra_obstacles = 25
        for _ in range(num_extra_obstacles):
            x = np.random.randint(2, self.map_size - 6)
            y = np.random.randint(2, self.map_size - 6)
            if self.ground_truth[x, y] == 0:
                w, h = np.random.randint(2, 5), np.random.randint(2, 5)
                x_end = min(x + w, self.map_size - 1)
                y_end = min(y + h, self.map_size - 1)
                self.ground_truth[x:x_end, y:y_end] = 1

    def get_tensor(self):
        # 转换为 PyTorch 需要的格式：[Batch, Channel, Height, Width]
        # 这里 Batch=1, Channel=1
        tensor = torch.tensor(self.ground_truth, dtype=torch.float32)
        return tensor.unsqueeze(0).unsqueeze(0)

# ==========================================
# 2. 双分辨率卷积系统架构 (核心逻辑)
# ==========================================
class DualResolutionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42) # 固定随机权重以便观察
        
        # [高分辨率分支]：提取局部几何特征（如墙壁边缘、门洞）
        # 输入通道: 1, 输出通道: 4, 保持原地图尺寸不变
        self.conv_high = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        
        # [降采样单元]：将空间尺寸压缩一半，扩大感受野
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # [低分辨率分支]：提取全局拓扑特征（如房间结构、大片空地）
        # 输入通道: 1 (来自降采样后的原图), 输出通道: 4
        self.conv_low = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        
        # [上采样单元]：将低分辨率特征图放大回原尺寸
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # [特征融合与价值输出]：接收高低两种分辨率的特征，输出最终的一张热力图
        # 输入通道: 4(高) + 4(低) = 8， 输出通道: 1
        self.conv_fuse = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, x):
        print(f"=== 系统输入信号 ===\n输入原始地图 X: {x.shape}\n")
        
        # 1. 高分辨率处理流
        feat_high = self.conv_high(x)
        print(f"--- 状态变量 1: 高分辨率特征 ---")
        print(f"操作: 3x3卷积层")
        print(f"输出 feat_high: {feat_high.shape} -> (包含局部几何信息)\n")
        
        # 2. 降采样与低分辨率处理流
        x_down = self.pool(x)
        feat_low = self.conv_low(x_down)
        print(f"--- 状态变量 2: 低分辨率特征 ---")
        print(f"操作: 2x2最大池化 -> 输入尺寸减半: {x_down.shape}")
        print(f"操作: 3x3卷积层")
        print(f"输出 feat_low: {feat_low.shape} -> (包含宏观拓扑信息)\n")
        
        # 3. 特征对齐与融合流
        feat_low_up = self.upsample(feat_low)
        print(f"--- 状态变量 3: 特征对齐 ---")
        print(f"操作: 上采样放大两倍")
        print(f"输出 feat_low_up: {feat_low_up.shape}\n")
        
        # 在通道维度（dim=1）将高低分辨率特征拼接
        fused_features = torch.cat((feat_high, feat_low_up), dim=1)
        print(f"--- 状态变量 4: 特征拼接 ---")
        print(f"操作: 矩阵 Concatenate (4通道 + 4通道)")
        print(f"输出 fused_features: {fused_features.shape}\n")
        
        # 4. 价值输出层
        value_map = self.conv_fuse(fused_features)
        print(f"=== 系统输出信号 ===")
        print(f"操作: 1x1卷积降维")
        print(f"输出 value_map: {value_map.shape} -> (全局坐标价值热力图)\n")
        
        return feat_high, x_down, feat_low, value_map

# ==========================================
# 3. 运行与可视化子系统
# ==========================================
if __name__ == "__main__":
    # 生成地图并运行网络
    generator = MapGenerator(map_size=60)
    input_tensor = generator.get_tensor()
    
    model = DualResolutionCNN()
    # 关闭梯度计算，因为我们只做前向传播分析
    with torch.no_grad():
        feat_high, map_down, feat_low, value_map = model(input_tensor)

    # 提取 Tensor 的数据用于 matplotlib 绘图
    # [0, 0, :, :] 表示取 Batch=0, Channel=0 的二维矩阵
    img_input = input_tensor[0, 0].numpy()
    img_feat_high = feat_high[0, 0].numpy() # 仅展示 4 个通道中的第 0 个
    img_map_down = map_down[0, 0].numpy()
    img_feat_low = feat_low[0, 0].numpy()   # 仅展示 4 个通道中的第 0 个
    img_value = value_map[0, 0].numpy()

    # 绘制 I/O 对比图
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_input, cmap='gray_r')
    axes[0].set_title("Input Map\n(60x60)")
    
    axes[1].imshow(img_feat_high, cmap='viridis')
    axes[1].set_title("High-Res Feature (Ch 0)\n(60x60)")
    
    axes[2].imshow(img_map_down, cmap='gray_r')
    axes[2].set_title("Max Pooled Map\n(30x30)")
    
    axes[3].imshow(img_feat_low, cmap='viridis')
    axes[3].set_title("Low-Res Feature (Ch 0)\n(30x30)")
    
    axes[4].imshow(img_value, cmap='jet')
    axes[4].set_title("Final Value Heatmap\n(60x60)")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()