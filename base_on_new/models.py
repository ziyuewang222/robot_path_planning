import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpatialFCN(nn.Module):
    """多尺度空间价值网络。

    设计目标：
    1. 保留原始分辨率分支，负责精确选点；
    2. 让 1/2 分辨率分支提供中尺度空间上下文，并继续参与逐像素 Q 图生成；
    3. 让 1/4 分辨率分支提取更宏观的拓扑结构，上采样后直接参与逐像素 Q 图生成。

    说明：
    - full/half/quarter 三个尺度的融合权重是 nn.Parameter，会在强化学习训练中一起更新；
    - 1/2 与 1/4 输入下采样时，边界/可行动作通道继续使用最大池化，稠密通道使用平均池化；
    - 最终输出仍然是 (B, 1, H, W)，便于执行单像素级动作选择。
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 32):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2

        # 原始分辨率分支：保留局部精细信息
        self.full_res_branch = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 1/2 分辨率分支：扩大感受野，观察更大局部结构
        self.half_res_branch = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 1/4 分辨率分支：进一步扩大感受野，提取更全局的结构信息
        self.quarter_res_branch = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 三个尺度统一投影到同一通道数后，在原始分辨率上融合生成逐像素 Q 图。
        self.fuse_proj_full = nn.Conv2d(c1, c2, kernel_size=1)
        self.fuse_proj_half = nn.Conv2d(c2, c2, kernel_size=1)
        self.fuse_proj_quarter = nn.Conv2d(c2, c2, kernel_size=1)

        self.scale_fusion_logits = nn.Parameter(torch.zeros(3))

        self.head = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

        self._last_fusion_weights = None

        self.boundary_channels = (4,)
        self.dense_channels = (0, 1, 2, 3)

    def _mixed_pool2d(self, x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        """ch4 边界掩码用最大池化，ch0 gain 与其他连续/稠密通道用平均池化，并恢复原通道顺序。"""
        pooled = x.new_empty(
            x.shape[0],
            x.shape[1],
            x.shape[2] // stride,
            x.shape[3] // stride,
        )
        pooled[:, self.boundary_channels] = F.max_pool2d(
            x[:, self.boundary_channels],
            kernel_size=kernel_size,
            stride=stride,
        )
        pooled[:, self.dense_channels] = F.avg_pool2d(
            x[:, self.dense_channels],
            kernel_size=kernel_size,
            stride=stride,
        )
        return pooled

    def get_fusion_weights(self) -> torch.Tensor:
        """返回 full/half/quarter 三个空间分支的 softmax 融合权重。"""
        return torch.softmax(self.scale_fusion_logits, dim=0)

    def get_display_fusion_weights(self) -> torch.Tensor:
        """返回用于训练监控/可视化展示的三尺度融合权重。"""
        return self.get_fusion_weights()

    def set_fusion_logits(self, logits: torch.Tensor) -> None:
        """外部接口：手动更换 full/half/quarter 三尺度融合权重参数的初始化值。"""
        if logits.shape == (2,):
            logits = torch.cat([logits, logits.new_zeros(1)], dim=0)
        if logits.shape != (3,):
            raise ValueError("fusion logits 必须是形状为 (3,) 的张量，或兼容旧权重的 (2,) 张量")
        with torch.no_grad():
            self.scale_fusion_logits.copy_(logits.to(self.scale_fusion_logits.device))

    def set_fusion_weights(self, weights: torch.Tensor) -> None:
        """按概率权重直接设置 full/half/quarter 三尺度融合系数。"""
        weights = torch.as_tensor(weights, dtype=self.scale_fusion_logits.dtype, device=self.scale_fusion_logits.device)
        if weights.shape == (2,):
            weights = torch.cat([weights, weights.new_zeros(1)], dim=0)
        if weights.shape != (3,):
            raise ValueError("fusion weights 必须是形状为 (3,) 的张量，或兼容旧权重的 (2,) 张量")
        weights = torch.clamp(weights, min=1e-8)
        weights = weights / weights.sum()
        logits = torch.log(weights)
        self.set_fusion_logits(logits)

    def save_fusion_weights(self, path: str) -> None:
        """保存 full/half/quarter 三尺度空间融合权重。"""
        torch.save({"scale_fusion_logits": self.scale_fusion_logits.detach().cpu()}, path)

    def load_fusion_weights(self, path: str, strict: bool = True) -> None:
        """加载 full/half/quarter 三尺度空间融合权重，兼容旧版二尺度融合权重。"""
        payload = torch.load(path, map_location=self.scale_fusion_logits.device)
        if isinstance(payload, dict) and "scale_fusion_logits" in payload:
            self.set_fusion_logits(payload["scale_fusion_logits"])
        elif not strict:
            return
        else:
            raise KeyError("权重文件中未找到 scale_fusion_logits")

    def extract_features(self, x: torch.Tensor):
        _, _, h, w = x.shape

        full_feat = self.full_res_branch(x)
        full_feat = self.fuse_proj_full(full_feat)

        half_x = self._mixed_pool2d(x, kernel_size=2, stride=2)
        half_feat = self.half_res_branch(half_x)
        half_feat = self.fuse_proj_half(half_feat)
        half_feat = F.interpolate(half_feat, size=(h, w), mode="bilinear", align_corners=False)

        quarter_x = self._mixed_pool2d(x, kernel_size=4, stride=4)
        quarter_feat = self.quarter_res_branch(quarter_x)
        quarter_feat = self.fuse_proj_quarter(quarter_feat)
        quarter_feat = F.interpolate(quarter_feat, size=(h, w), mode="bilinear", align_corners=False)

        fusion_weights = self.get_fusion_weights()
        fused_feat = (
            fusion_weights[0] * full_feat
            + fusion_weights[1] * half_feat
            + fusion_weights[2] * quarter_feat
        )

        self._last_fusion_weights = self.get_display_fusion_weights().detach().cpu()

        return full_feat, half_feat, quarter_feat, fused_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, fused_feat = self.extract_features(x)
        return self.head(fused_feat)


class SpatialFCN(MultiScaleSpatialFCN):
    """兼容旧接口的模型名。

    训练/测试代码仍可直接使用 SpatialFCN；当前实现已经升级为：
    原分辨率 + 1/2 + 1/4 多尺度结构。
    """

    pass


class SpatialActionDecision:
    def __init__(self, model, epsilon=0.1, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state_tensor, explore=True, return_maps=True):
        self.model.eval()
        mask = state_tensor[4].to(self.device)
        valid_indices = torch.nonzero(mask > 0.5)
        if len(valid_indices) == 0:
            return None, None, None

        if explore and np.random.rand() < self.epsilon:
            random_idx = np.random.randint(0, len(valid_indices))
            chosen_coord = valid_indices[random_idx].tolist()
            return chosen_coord, None, None

        state_batch = state_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_raw = self.model(state_batch).squeeze(0).squeeze(0)

        q_masked = q_raw.clone()
        q_masked[mask < 0.5] = -1e9

        flat_idx = torch.argmax(q_masked).item()
        _, W = q_masked.shape
        chosen_coord = [flat_idx // W, flat_idx % W]

        if not return_maps:
            return chosen_coord, None, None
        return chosen_coord, q_masked.cpu().numpy(), q_raw.cpu().numpy()
