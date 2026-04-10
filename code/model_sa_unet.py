import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=3):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x

        block_size = min(self.block_size, x.shape[-2], x.shape[-1])
        if block_size < 1:
            return x

        gamma = self.drop_prob * x.shape[-2] * x.shape[-1]
        gamma /= float(block_size ** 2 * x.shape[-2] * x.shape[-1])

        seed_mask = torch.full_like(x, gamma)
        seed_mask = torch.bernoulli(seed_mask)
        block_mask = F.max_pool2d(seed_mask, kernel_size=block_size, stride=1, padding=block_size // 2)
        block_mask = 1 - block_mask

        scale = block_mask.numel() / block_mask.sum().clamp(min=1.0)
        return x * block_mask * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class StructuredConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, block_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        x = self.dropblock(self.relu(self.bn1(self.conv1(x))))
        x = self.dropblock(self.relu(self.bn2(self.conv2(x))))
        return x


class OfficialDropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.18, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def _compute_valid_seed_region(self, height, width, device):
        positions_y = torch.arange(height, device=device).view(height, 1).expand(height, width)
        positions_x = torch.arange(width, device=device).view(1, width).expand(height, width)
        half_block = self.block_size // 2
        valid = (
            (positions_y >= half_block)
            & (positions_x >= half_block)
            & (positions_y < height - half_block)
            & (positions_x < width - half_block)
        )
        return valid.to(dtype=torch.float32).view(1, 1, height, width)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x

        height, width = x.shape[-2:]
        block_size = min(self.block_size, height, width)
        if block_size < 1:
            return x

        block_size = int(block_size)
        gamma = self.drop_prob / float(block_size ** 2)
        gamma *= (height * width) / float((height - block_size + 1) * (width - block_size + 1))

        seed_mask = torch.bernoulli(torch.full_like(x, gamma))
        valid_seed = self._compute_valid_seed_region(height, width, x.device)
        seed_mask = seed_mask * valid_seed
        block_mask = 1.0 - F.max_pool2d(
            seed_mask,
            kernel_size=block_size,
            stride=1,
            padding=block_size // 2,
        )
        scale = block_mask.numel() / block_mask.sum().clamp(min=1.0)
        return x * block_mask * scale


class OfficialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.18, block_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop1 = OfficialDropBlock2D(drop_prob=drop_prob, block_size=block_size)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop2 = OfficialDropBlock2D(drop_prob=drop_prob, block_size=block_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x, attention=None):
        x = self.act1(self.bn1(self.drop1(self.conv1(x))))
        if attention is not None:
            x = attention(x)
        x = self.act2(self.bn2(self.drop2(self.conv2(x))))
        return x


class OfficialUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, drop_prob=0.18, block_size=7):
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
        self.conv = OfficialConvBlock(out_channels + skip_channels, out_channels, drop_prob=drop_prob, block_size=block_size)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x, skip], dim=1))


class SAUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, drop_prob=0.18, block_size=7):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = OfficialConvBlock(in_channels, channels[0], drop_prob=drop_prob, block_size=block_size)
        self.enc2 = OfficialConvBlock(channels[0], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.enc3 = OfficialConvBlock(channels[1], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.attention = SpatialAttention()
        self.bottleneck = OfficialConvBlock(channels[2], channels[3], drop_prob=drop_prob, block_size=block_size)
        self.dec3 = OfficialUpBlock(channels[3], channels[2], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.dec2 = OfficialUpBlock(channels[2], channels[1], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.dec1 = OfficialUpBlock(channels[1], channels[0], channels[0], drop_prob=drop_prob, block_size=block_size)
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3), attention=self.attention)

        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        logits = self.head(dec1)

        if pad_h or pad_w:
            logits = logits[..., :orig_h, :orig_w]
        return logits
