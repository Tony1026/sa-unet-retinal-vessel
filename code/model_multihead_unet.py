import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sa_unet import SpatialAttention
from model_sa_unetv2 import StructuredConvBlockV2, UpBlockV2
from model_unet import ConvBlock, UpBlock


def _zero_flow_loss(*tensors):
    base = tensors[0]
    for tensor in tensors[1:]:
        base = base + tensor.sum() * 0.0
    return base.sum() * 0.0


class MultiHeadUNet(nn.Module):
    """Two independent terminal heads on a standard U-Net decoder.

    This is the no-flow multi-head baseline: it has the same supervised sink
    interface as GFlow-UNet, but no source-to-sink feature-flow graph.
    """

    def __init__(self, in_channels=1, base_channels=16, drop_prob=0.0, block_size=7):
        super().__init__()
        del drop_prob, block_size
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(in_channels, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.bottleneck = ConvBlock(channels[2], channels[3])
        self.dec3 = UpBlock(channels[3], channels[2], channels[2])
        self.dec2 = UpBlock(channels[2], channels[1], channels[1])
        self.dec1 = UpBlock(channels[1], channels[0], channels[0])
        self.head1 = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.head2 = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        sink1_logits = self.head1(dec1)
        sink2_logits = self.head2(dec1)

        if pad_h or pad_w:
            sink1_logits = sink1_logits[..., :orig_h, :orig_w]
            sink2_logits = sink2_logits[..., :orig_h, :orig_w]

        zero = _zero_flow_loss(sink1_logits, sink2_logits)
        return {
            'logits': sink1_logits,
            'sink1': sink1_logits,
            'sink2': sink2_logits,
            'flow_conservation_loss': zero,
            'flow_sparse_loss': zero,
            'flow_edges': {},
        }


class MultiHeadSAUNetV2(nn.Module):
    """Two-head SA-UNetV2 baseline without learned feature-flow edges."""

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
        self.head1 = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.head2 = nn.Conv2d(channels[0], 1, kernel_size=1)

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
        sink1_logits = self.head1(dec1)
        sink2_logits = self.head2(dec1)

        if pad_h or pad_w:
            sink1_logits = sink1_logits[..., :orig_h, :orig_w]
            sink2_logits = sink2_logits[..., :orig_h, :orig_w]

        zero = _zero_flow_loss(sink1_logits, sink2_logits)
        return {
            'logits': sink1_logits,
            'sink1': sink1_logits,
            'sink2': sink2_logits,
            'flow_conservation_loss': zero,
            'flow_sparse_loss': zero,
            'flow_edges': {},
        }
