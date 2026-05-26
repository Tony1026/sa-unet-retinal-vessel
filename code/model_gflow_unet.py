import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sa_unet import SpatialAttention
from model_sa_unetv2 import StructuredConvBlockV2, UpBlockV2


def _resize_like(x, ref):
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


def _initialize_primary_consensus(module):
    with torch.no_grad():
        module.weight.zero_()
        module.weight[:, 0:1].fill_(1.0)
        module.bias.zero_()


class FlowMixer(nn.Module):
    def __init__(self, source_channels, out_channels, target_name, conditional=False):
        super().__init__()
        self.target_name = target_name
        self.source_names = [name for name, _ in source_channels]
        self.conditional = conditional
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, out_channels),
                )
                for _, channels in source_channels
            ]
        )
        self.logits = nn.Parameter(torch.zeros(len(source_channels)))
        if conditional:
            hidden_channels = max(out_channels // 2, len(source_channels))
            self.conditioner = nn.Sequential(
                nn.Conv2d(out_channels, hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, len(source_channels), kernel_size=1),
            )
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.conditioner = None

    def weights(self, ref=None):
        logits = self.logits
        if self.conditioner is not None and ref is not None:
            logits = logits.view(1, -1, 1, 1) + self.conditioner(ref)
            return torch.softmax(logits, dim=1)
        return torch.softmax(logits, dim=0)

    def forward(self, features, ref):
        weights = self.weights(ref)
        mixed = None
        edge_weights = {}
        for index, (name, transform) in enumerate(zip(self.source_names, self.transforms)):
            weight = weights[:, index:index + 1] if weights.dim() == 4 else weights[index]
            value = _resize_like(transform(features[name]), ref)
            mixed = value * weight if mixed is None else mixed + value * weight
            edge_weights[f'{name}->{self.target_name}'] = (value.abs() * weight).mean()
        return mixed, edge_weights


def _configure_flow_modules(module, mode, freeze=True):
    if mode == 'learned':
        return
    generator = torch.Generator()
    generator.manual_seed(2026)
    for submodule in module.modules():
        if not isinstance(submodule, FlowMixer):
            continue
        with torch.no_grad():
            if mode == 'uniform':
                submodule.logits.zero_()
            elif mode == 'random':
                submodule.logits.copy_(torch.randn(submodule.logits.shape, generator=generator))
            else:
                raise ValueError(f'Unsupported flow mode: {mode}')
        if freeze:
            submodule.logits.requires_grad_(False)
            if submodule.conditioner is not None:
                for parameter in submodule.conditioner.parameters():
                    parameter.requires_grad_(False)


class FlowSink(nn.Module):
    def __init__(self, source_channels, hidden_channels, target_name, drop_prob=0.15, block_size=7, conditional=False):
        super().__init__()
        self.mixer = FlowMixer(source_channels, hidden_channels, target_name, conditional=conditional)
        self.refine = StructuredConvBlockV2(
            hidden_channels,
            hidden_channels,
            drop_prob=drop_prob,
            block_size=block_size,
        )
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, features, ref):
        x, edge_weights = self.mixer(features, ref)
        x = self.refine(x)
        return self.head(x), edge_weights


class GFlowUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, drop_prob=0.15, block_size=7):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 3, base_channels * 4]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.source = StructuredConvBlockV2(in_channels, channels[0], drop_prob=drop_prob, block_size=block_size)
        self.enc2 = StructuredConvBlockV2(channels[0], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.enc3 = StructuredConvBlockV2(channels[1], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.bottleneck = StructuredConvBlockV2(channels[2], channels[3], drop_prob=drop_prob, block_size=block_size)

        self.d3_mixer = FlowMixer(
            [('B', channels[3]), ('E3', channels[2]), ('E2', channels[1])],
            channels[2],
            'D3',
        )
        self.d3_refine = StructuredConvBlockV2(channels[2], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.d2_mixer = FlowMixer(
            [('D3', channels[2]), ('E2', channels[1]), ('E1', channels[0]), ('B', channels[3])],
            channels[1],
            'D2',
        )
        self.d2_refine = StructuredConvBlockV2(channels[1], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.d1_mixer = FlowMixer(
            [('D2', channels[1]), ('E1', channels[0]), ('D3', channels[2])],
            channels[0],
            'D1',
        )
        self.d1_refine = StructuredConvBlockV2(channels[0], channels[0], drop_prob=drop_prob, block_size=block_size)

        sink_sources = [
            ('D1', channels[0]),
            ('D2', channels[1]),
            ('D3', channels[2]),
            ('E1', channels[0]),
            ('B', channels[3]),
        ]
        self.sink1 = FlowSink(sink_sources, channels[0], 'Y1', drop_prob=drop_prob, block_size=block_size)
        self.sink2 = FlowSink(sink_sources, channels[0], 'Y2', drop_prob=drop_prob, block_size=block_size)
        self.consensus = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        _initialize_primary_consensus(self.consensus)

    def _flow_regularizers(self, edge_weights):
        incoming = {}
        outgoing = {}
        for edge, weight in edge_weights.items():
            source, target = edge.split('->')
            outgoing[source] = outgoing.get(source, weight.new_tensor(0.0)) + weight
            incoming[target] = incoming.get(target, weight.new_tensor(0.0)) + weight

        intermediate_nodes = ('E1', 'E2', 'E3', 'B', 'D3', 'D2', 'D1')
        conservation = edge_weights[next(iter(edge_weights))].new_tensor(0.0)
        for node in intermediate_nodes:
            if node in incoming and node in outgoing:
                conservation = conservation + (incoming[node] - outgoing[node]).pow(2)

        entropies = []
        for module in (self.d3_mixer, self.d2_mixer, self.d1_mixer, self.sink1.mixer, self.sink2.mixer):
            weights = module.weights().clamp(min=1e-8)
            entropies.append(-(weights * weights.log()).sum())
        sparse = torch.stack(entropies).mean()
        return conservation, sparse

    def flow_report(self):
        report = {}
        for module in (self.d3_mixer, self.d2_mixer, self.d1_mixer, self.sink1.mixer, self.sink2.mixer):
            weights = module.weights().detach().cpu().tolist()
            for name, weight in zip(module.source_names, weights):
                report[f'{name}->{module.target_name}'] = float(weight)
        return report

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        features = {}
        features['E1'] = self.source(x)
        features['E2'] = self.enc2(self.pool(features['E1']))
        features['E3'] = self.enc3(self.pool(features['E2']))
        features['B'] = self.bottleneck(self.pool(features['E3']))

        edge_weights = {}
        mixed, weights = self.d3_mixer(features, features['E3'])
        edge_weights.update(weights)
        features['D3'] = self.d3_refine(mixed)

        mixed, weights = self.d2_mixer(features, features['E2'])
        edge_weights.update(weights)
        features['D2'] = self.d2_refine(mixed)

        mixed, weights = self.d1_mixer(features, features['E1'])
        edge_weights.update(weights)
        features['D1'] = self.d1_refine(mixed)

        sink1_logits, weights = self.sink1(features, features['D1'])
        edge_weights.update(weights)
        sink2_logits, weights = self.sink2(features, features['D1'])
        edge_weights.update(weights)
        primary_logits = self.consensus(torch.cat([sink1_logits, sink2_logits], dim=1))
        conservation, sparse = self._flow_regularizers(edge_weights)

        if pad_h or pad_w:
            sink1_logits = sink1_logits[..., :orig_h, :orig_w]
            sink2_logits = sink2_logits[..., :orig_h, :orig_w]
            primary_logits = primary_logits[..., :orig_h, :orig_w]

        return {
            'logits': primary_logits,
            'sink1': sink1_logits,
            'sink2': sink2_logits,
            'flow_conservation_loss': conservation,
            'flow_sparse_loss': sparse,
            'flow_edges': edge_weights,
        }


class GFlowUNetNoConservation(GFlowUNet):
    def _flow_regularizers(self, edge_weights):
        conservation, sparse = super()._flow_regularizers(edge_weights)
        return conservation.detach() * 0.0, sparse


class GFlowUNetUniformFlow(GFlowUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'uniform')


class GFlowUNetRandomFlow(GFlowUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'random')


class GFlowUNetRandomInit(GFlowUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'random', freeze=False)


class GFlowUNetDirectSink(GFlowUNet):
    def forward(self, x):
        output = super().forward(x)
        output['logits'] = output['sink1']
        return output


class GFlowUNetNoConservationDirectSink(GFlowUNetNoConservation):
    def forward(self, x):
        output = super().forward(x)
        output['logits'] = output['sink1']
        return output


class GFlowUNetUniformFlowDirectSink(GFlowUNetUniformFlow):
    def forward(self, x):
        output = super().forward(x)
        output['logits'] = output['sink1']
        return output


class GFlowUNetRandomFlowDirectSink(GFlowUNetRandomFlow):
    def forward(self, x):
        output = super().forward(x)
        output['logits'] = output['sink1']
        return output


class GFlowUNetRandomInitDirectSink(GFlowUNetRandomInit):
    def forward(self, x):
        output = super().forward(x)
        output['logits'] = output['sink1']
        return output


class GFlowSAUNetV2(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, drop_prob=0.15, block_size=7, conditional_flow=False):
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

        self.d3_mixer = FlowMixer(
            [('B', channels[3]), ('E3', channels[2]), ('E2', channels[1])],
            channels[2],
            'D3',
            conditional=conditional_flow,
        )
        self.d2_mixer = FlowMixer(
            [('D3', channels[2]), ('E2', channels[1]), ('E1', channels[0]), ('B', channels[3])],
            channels[1],
            'D2',
            conditional=conditional_flow,
        )
        self.d1_mixer = FlowMixer(
            [('D2', channels[1]), ('E1', channels[0]), ('D3', channels[2])],
            channels[0],
            'D1',
            conditional=conditional_flow,
        )
        self.d3_refine = StructuredConvBlockV2(channels[2], channels[2], drop_prob=drop_prob, block_size=block_size)
        self.d2_refine = StructuredConvBlockV2(channels[1], channels[1], drop_prob=drop_prob, block_size=block_size)
        self.d1_refine = StructuredConvBlockV2(channels[0], channels[0], drop_prob=drop_prob, block_size=block_size)
        self.flow_scales = nn.ParameterDict(
            {
                'D3': nn.Parameter(torch.tensor(0.25)),
                'D2': nn.Parameter(torch.tensor(0.25)),
                'D1': nn.Parameter(torch.tensor(0.25)),
            }
        )

        sink_sources = [
            ('D1', channels[0]),
            ('D2', channels[1]),
            ('D3', channels[2]),
            ('E1', channels[0]),
            ('B', channels[3]),
        ]
        self.sink1 = FlowSink(
            sink_sources,
            channels[0],
            'Y1',
            drop_prob=drop_prob,
            block_size=block_size,
            conditional=conditional_flow,
        )
        self.sink2 = FlowSink(
            sink_sources,
            channels[0],
            'Y2',
            drop_prob=drop_prob,
            block_size=block_size,
            conditional=conditional_flow,
        )
        self.consensus = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        _initialize_primary_consensus(self.consensus)

    def _flow_regularizers(self, edge_weights):
        return GFlowUNet._flow_regularizers(self, edge_weights)

    def flow_report(self):
        report = {}
        for module in (self.d3_mixer, self.d2_mixer, self.d1_mixer, self.sink1.mixer, self.sink2.mixer):
            weights = module.weights().detach().cpu().tolist()
            for name, weight in zip(module.source_names, weights):
                report[f'{name}->{module.target_name}'] = float(weight)
        for name, value in self.flow_scales.items():
            report[f'flow_scale_{name}'] = float(value.detach().cpu())
        return report

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        features = {}
        features['E1'] = self.enc1(x)
        features['E2'] = self.enc2(self.pool(features['E1']))
        features['E3'] = self.enc3(self.pool(features['E2']))
        features['B'] = self.attention(self.bottleneck(self.pool(features['E3'])))

        edge_weights = {}
        dec3 = self.dec3(features['B'], features['E3'])
        flow, weights = self.d3_mixer(features, dec3)
        edge_weights.update(weights)
        features['D3'] = self.d3_refine(dec3 + self.flow_scales['D3'] * flow)

        dec2 = self.dec2(features['D3'], features['E2'])
        flow, weights = self.d2_mixer(features, dec2)
        edge_weights.update(weights)
        features['D2'] = self.d2_refine(dec2 + self.flow_scales['D2'] * flow)

        dec1 = self.dec1(features['D2'], features['E1'])
        flow, weights = self.d1_mixer(features, dec1)
        edge_weights.update(weights)
        features['D1'] = self.d1_refine(dec1 + self.flow_scales['D1'] * flow)

        sink1_logits, weights = self.sink1(features, features['D1'])
        edge_weights.update(weights)
        sink2_logits, weights = self.sink2(features, features['D1'])
        edge_weights.update(weights)
        primary_logits = self.consensus(torch.cat([sink1_logits, sink2_logits], dim=1))
        conservation, sparse = self._flow_regularizers(edge_weights)

        if pad_h or pad_w:
            sink1_logits = sink1_logits[..., :orig_h, :orig_w]
            sink2_logits = sink2_logits[..., :orig_h, :orig_w]
            primary_logits = primary_logits[..., :orig_h, :orig_w]

        return {
            'logits': primary_logits,
            'sink1': sink1_logits,
            'sink2': sink2_logits,
            'flow_conservation_loss': conservation,
            'flow_sparse_loss': sparse,
            'flow_edges': edge_weights,
        }


class GFlowSAUNetV2NoConservation(GFlowSAUNetV2):
    def _flow_regularizers(self, edge_weights):
        conservation, sparse = super()._flow_regularizers(edge_weights)
        return conservation.detach() * 0.0, sparse


class GFlowSAUNetV2UniformFlow(GFlowSAUNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'uniform')


class GFlowSAUNetV2RandomFlow(GFlowSAUNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'random')


class GFlowSAUNetV2RandomInit(GFlowSAUNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'random', freeze=False)


class GFlowSAUNetV2Conditional(GFlowSAUNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, conditional_flow=True, **kwargs)


class GFlowSAUNetV2ConditionalNoConservation(GFlowSAUNetV2Conditional):
    def _flow_regularizers(self, edge_weights):
        conservation, sparse = super()._flow_regularizers(edge_weights)
        return conservation.detach() * 0.0, sparse


class GFlowSAUNetV2ConditionalUniformFlow(GFlowSAUNetV2Conditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'uniform')


class GFlowSAUNetV2ConditionalRandomFlow(GFlowSAUNetV2Conditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'random')


class GFlowSAUNetV2ConditionalRandomInit(GFlowSAUNetV2Conditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_flow_modules(self, 'random', freeze=False)
