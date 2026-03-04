import torch
import torch.nn as nn
import torch.nn.functional as F


class ShearletFrequencyExtractor(nn.Module):
    """
    纯 PyTorch 实现的 Shearlet 风格多方向高频提取器 (GPU 加速版)
    模拟 Shearlet 的各向异性（Anisotropic）特性，提取 0°, 45°, 90°, 135° 四个方向的边缘。
    """

    def __init__(self, in_channels=3):
        super(ShearletFrequencyExtractor, self).__init__()

        # 1. 构建多方向剪切波/边缘高频滤波器 (3x3 紧支撑)
        # 0° (水平高频，抓取上下边缘)
        kernel_0 = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # 90° (垂直高频，抓取左右边缘)
        kernel_90 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        # 45° (对角线高频)
        kernel_45 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        # 135° (反对角线高频)
        kernel_135 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)

        # 2. 打包成卷积核: 形状为 [4, 1, 3, 3]
        kernels = torch.stack([kernel_0, kernel_90, kernel_45, kernel_135]).unsqueeze(1)

        # 3. 复制到每一个 RGB 通道 (使用分组卷积/深度卷积，避免通道串味)
        # 形状变为 [12, 1, 3, 3] (3个通道 * 4个方向)
        self.weight = nn.Parameter(kernels.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.in_channels = in_channels

    def forward(self, x):
        """
        x: 输入图像 Tensor, 形状为 [Batch, 3, H, W]
        """
        B, C, H, W = x.shape

        # 1. 极速 GPU 滤波：同时提取 3 个通道的 4 个方向特征
        # 输出 out 形状: [Batch, 12, H, W]
        out = F.conv2d(x, self.weight.to(x.device), padding=1, groups=C)

        # 2. 将张量重塑为 [Batch, 3通道, 4方向, H, W]
        out = out.view(B, C, 4, H, W)

        # 3. 频域能量聚合：计算各个方向的高频能量总和 (L2 范数)
        # 这样可以将复杂的 Shearlet 响应重新压缩回 3 个通道，无缝对接原模型！
        # 输出 high_freq_magnitude 形状: [Batch, 3, H, W]
        high_freq_magnitude = torch.sqrt(torch.sum(out ** 2, dim=2) + 1e-8)

        # 4. 能量归一化 (防止数值爆炸，对齐原 DWT 的分布范围)
        high_freq_magnitude = high_freq_magnitude / 2.0

        return high_freq_magnitude