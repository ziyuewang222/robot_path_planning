import torch.nn as nn
import torch.nn.functional as F

class SemanticUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # 编码器 (Encoder)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        
        # 解码器 (Decoder)
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        
        # 输出层 (类别0:障碍/迷雾, 类别1:走廊, 类别2:房间)
        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2[0](e3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2[1:](d2)
        # 引入特征融合（残差连接）
        d1 = self.dec1[0](d2 + e2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1[1:](d1)
        out = self.final(d1 + e1)
        return out