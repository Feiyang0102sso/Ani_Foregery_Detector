import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ShearletFrequencyExtractor(nn.Module):
    """
    纯 PyTorch 实现的 Shearlet 风格多方向高频提取器 (GPU 加速版)
    模拟 Shearlet 的各向异性（Anisotropic）特性，提取多方向边缘响应，并用能量聚合压回 3 通道。
    """

    def __init__(self, in_channels=3, num_directions=8):
        super(ShearletFrequencyExtractor, self).__init__()
        if num_directions != 8:
            raise ValueError("当前实现仅支持 num_directions=8（8 个方向）")

        self.in_channels = in_channels
        self.num_directions = num_directions

        # 1. 构建 8 方向边缘高频滤波器 (3x3 紧支撑，compass/prewitt 风格)
        # 方向顺序：N, NE, E, SE, S, SW, W, NW
        kernel_n = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        kernel_ne = torch.tensor([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=torch.float32)
        kernel_e = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32)
        kernel_se = torch.tensor([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=torch.float32)
        kernel_s = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32)
        kernel_sw = torch.tensor([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=torch.float32)
        kernel_w = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        kernel_nw = torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=torch.float32)

        # 2. 打包成卷积核: 形状为 [8, 1, 3, 3]
        kernels = torch.stack(
            [kernel_n, kernel_ne, kernel_e, kernel_se, kernel_s, kernel_sw, kernel_w, kernel_nw]
        ).unsqueeze(1)

        # 3. 复制到每一个输入通道 (使用分组卷积/深度卷积，避免通道串味)
        # 形状变为 [in_channels * 8, 1, 3, 3]
        self.weight = nn.Parameter(kernels.repeat(in_channels, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        """
        x: 输入图像 Tensor, 形状为 [Batch, 3, H, W]
        """
        B, C, H, W = x.shape

        # 1. 极速 GPU 滤波：同时提取 3 个通道的 4 个方向特征
        # 输出 out 形状: [Batch, in_channels * num_directions, H, W]
        out = F.conv2d(x, self.weight.to(x.device), padding=1, groups=C)

        # 2. 将张量重塑为 [Batch, in_channels, num_directions, H, W]
        out = out.view(B, C, self.num_directions, H, W)

        # 3. 频域能量聚合：计算各个方向的高频能量总和 (L2 范数)
        # 这样可以将复杂的 Shearlet 响应重新压缩回 3 个通道，无缝对接原模型！
        # 输出 high_freq_magnitude 形状: [Batch, 3, H, W]
        high_freq_magnitude = torch.sqrt(torch.sum(out ** 2, dim=2) + 1e-8)

        # 4. 能量归一化：幅值随方向数近似按 sqrt(K) 增长，这里做 sqrt(K) 归一化对齐尺度
        high_freq_magnitude = high_freq_magnitude / math.sqrt(float(self.num_directions))

        return high_freq_magnitude