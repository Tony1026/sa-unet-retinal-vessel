import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sa_unet import DropBlock2D, SpatialAttention


def _group_count(channels, preferred=8):
    upper = min(preferred, channels)
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class CSAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip, upsampled):
        avg_skip = torch.mean(skip, dim=1, keepdim=True)
        avg_upsampled = torch.mean(upsampled, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_skip, avg_upsampled], dim=1)))
        return skip * attention


class StructuredConvBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.15, block_size=7):
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop1 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop2 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.norm1(self.drop1(self.conv1(x))))
        x = self.act2(self.norm2(self.drop2(self.conv2(x))))
        return x


class UpBlockV2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, drop_prob=0.15, block_size=7):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.csa = CSAttention()
        self.conv = StructuredConvBlockV2(
            out_channels + skip_channels,
            out_channels,
            drop_prob=drop_prob,
            block_size=block_size,
        )

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        skip = self.csa(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))


class SAUNetV2(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, drop_prob=0.15, block_size=7):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 3, base_channels * 4]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = StructuredConvBlockV2(in_channels, channels[0], drop_prob=drop_prob, block_size=block_size)
        self.enc2 = StructuredConvBlockV2(channels[0], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.enc3 = StructuredConvBlockV2(channels[1], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.bottleneck = StructuredConvBlockV2(channels[2], channels[3], drop_prob=drop_prob, block_size=block_size)
        self.attention = SpatialAttention()
        self.dec3 = UpBlockV2(channels[3], channels[2], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.dec2 = UpBlockV2(channels[2], channels[1], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.dec1 = UpBlockV2(channels[1], channels[0], channels[0], drop_prob=drop_prob, block_size=block_size)
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.attention(self.bottleneck(self.pool(enc3)))

        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        logits = self.head(dec1)

        if pad_h or pad_w:
            logits = logits[..., :orig_h, :orig_w]
        return logits
