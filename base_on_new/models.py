import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpatialFCN(nn.Module):
    """多尺度空间价值网络。

    设计目标：
    1. 保留原始分辨率分支，负责精确选点；
    2. 引入 1/2 与 1/4 两个池化尺度，负责扩大感受野、观察走廊/房间等大结构；
    3. 通过可学习的融合权重把三个尺度融合成单通道价值图。

    说明：
    - 这里的融合权重是 nn.Parameter，会在强化学习训练中被一起更新；
    - 为了便于单独替换“参考权重”，本模型提供独立的融合权重保存/加载接口；
    - 最终输出仍然是 (B, 1, H, W)，便于执行单像素级动作选择。
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
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

        # 尺度融合前的通道对齐，便于三个分支做加权融合
        self.fuse_proj_full = nn.Conv2d(c1, c2, kernel_size=1)
        self.fuse_proj_half = nn.Conv2d(c2, c2, kernel_size=1)
        self.fuse_proj_quarter = nn.Conv2d(c2, c2, kernel_size=1)

        # 参考权重：这三个参数由强化学习一起训练出来，表示三个尺度的融合偏好。
        # softmax 后分别对应原分辨率、1/2 分辨率、1/4 分辨率的融合系数。
        self.scale_fusion_logits = nn.Parameter(torch.zeros(3))

        self.head = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

        self._last_fusion_weights = None

    def get_fusion_weights(self) -> torch.Tensor:
        """返回当前三个尺度的 softmax 融合权重。"""
        return torch.softmax(self.scale_fusion_logits, dim=0)

    def set_fusion_logits(self, logits: torch.Tensor) -> None:
        """外部接口：手动更换融合权重参数的初始化值。"""
        if logits.shape != (3,):
            raise ValueError("fusion logits 必须是形状为 (3,) 的张量")
        with torch.no_grad():
            self.scale_fusion_logits.copy_(logits.to(self.scale_fusion_logits.device))

    def save_fusion_weights(self, path: str) -> None:
        """只保存参考权重，便于和主干网络参数分开管理。"""
        torch.save({"scale_fusion_logits": self.scale_fusion_logits.detach().cpu()}, path)

    def load_fusion_weights(self, path: str, strict: bool = True) -> None:
        """只加载参考权重，不影响卷积主干参数。"""
        payload = torch.load(path, map_location=self.scale_fusion_logits.device)
        if isinstance(payload, dict) and "scale_fusion_logits" in payload:
            with torch.no_grad():
                self.scale_fusion_logits.copy_(payload["scale_fusion_logits"])
        elif not strict:
            return
        else:
            raise KeyError("权重文件中未找到 scale_fusion_logits")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # 原始分辨率路径
        full_feat = self.full_res_branch(x)
        full_feat = self.fuse_proj_full(full_feat)

        # 1/2 分辨率路径
        half_x = F.avg_pool2d(x, kernel_size=2, stride=2)
        half_feat = self.half_res_branch(half_x)
        half_feat = self.fuse_proj_half(half_feat)
        half_feat = F.interpolate(half_feat, size=(h, w), mode="bilinear", align_corners=False)

        # 1/4 分辨率路径
        quarter_x = F.avg_pool2d(x, kernel_size=4, stride=4)
        quarter_feat = self.quarter_res_branch(quarter_x)
        quarter_feat = self.fuse_proj_quarter(quarter_feat)
        quarter_feat = F.interpolate(quarter_feat, size=(h, w), mode="bilinear", align_corners=False)

        # 三个尺度的参考权重由强化学习学习得到
        fusion_weights = self.get_fusion_weights()
        self._last_fusion_weights = fusion_weights.detach().cpu()

        fused = (
            fusion_weights[0] * full_feat
            + fusion_weights[1] * half_feat
            + fusion_weights[2] * quarter_feat
        )
        return self.head(fused)


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

    def select_action(self, state_tensor, explore=True):
        self.model.eval()
        state_batch = state_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_raw = self.model(state_batch).squeeze(0).squeeze(0)

        mask = state_tensor[2].to(self.device)
        q_masked = q_raw.clone()
        q_masked[mask < 0.5] = -1e9

        valid_indices = torch.nonzero(mask > 0.5)
        if len(valid_indices) == 0:
            return None, q_masked.cpu().numpy(), q_raw.cpu().numpy()

        if explore and np.random.rand() < self.epsilon:
            random_idx = np.random.randint(0, len(valid_indices))
            chosen_coord = valid_indices[random_idx].tolist()
        else:
            flat_idx = torch.argmax(q_masked).item()
            _, W = q_masked.shape
            chosen_coord = [flat_idx // W, flat_idx % W]

        return chosen_coord, q_masked.cpu().numpy(), q_raw.cpu().numpy()
