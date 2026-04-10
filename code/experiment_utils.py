import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


MODEL_HPARAM_DEFAULTS = {
    'sa_unet': {
        'lr': 1e-3,
        'weight_decay': 0.0,
        'drop_prob': 0.18,
        'block_size': 7,
    },
    'sa_unetv2': {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'drop_prob': 0.15,
        'block_size': 7,
    },
}

TRAINING_HPARAM_DEFAULTS = {
    'epochs': 300,
    'plateau_patience': 12,
    'early_stop_patience': 24,
    'min_lr': 1e-6,
    'lr_factor': 0.5,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred='auto'):
    preferred = preferred.lower()
    if preferred == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('当前环境不可用 CUDA。')
        return torch.device('cuda'), 'cuda'
    if preferred == 'mps':
        if not getattr(torch.backends, 'mps', None) or not torch.backends.mps.is_available():
            raise RuntimeError('当前环境不可用 MPS。')
        return torch.device('mps'), 'mps'
    if preferred == 'cpu':
        return torch.device('cpu'), 'cpu'

    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps'), 'mps'
    return torch.device('cpu'), 'cpu'


def default_batch_size(device_type):
    return {'cuda': 8, 'mps': 4, 'cpu': 2}[device_type]


def apply_training_defaults(args):
    arg_values = vars(args)
    model_name = arg_values['model_name']
    for defaults in (MODEL_HPARAM_DEFAULTS[model_name], TRAINING_HPARAM_DEFAULTS):
        for key, value in defaults.items():
            if arg_values[key] is None:
                arg_values[key] = value
    return args


def masked_bce_dice_loss(logits, targets, masks, bce_weight=0.5, smooth=1e-6):
    masks = masks.float()
    targets = targets.float() * masks
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    bce = (bce * masks).sum() / masks.sum().clamp(min=1.0)

    probs = torch.sigmoid(logits) * masks
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
    return bce_weight * bce + (1.0 - bce_weight) * dice_loss.mean()


def masked_bce_loss(logits, targets, masks):
    masks = masks.float()
    targets = targets.float() * masks
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (bce * masks).sum() / masks.sum().clamp(min=1.0)


def compute_loss(logits, targets, masks, loss_mode='bce_dice'):
    if loss_mode == 'bce':
        return masked_bce_loss(logits, targets, masks)
    if loss_mode == 'bce_dice':
        return masked_bce_dice_loss(logits, targets, masks)
    raise ValueError(f'Unsupported loss mode: {loss_mode}')


def _safe_divide(numerator, denominator, default=0.0):
    return numerator / denominator if denominator else default


def compute_binary_metrics_from_probs(probs, targets, masks, threshold=0.5, smooth=1e-6):
    preds = (probs >= threshold).float() * masks
    targets = targets.float() * masks
    masks = masks.float()

    tp = (preds * targets).sum().item()
    tn = ((1 - preds) * (1 - targets) * masks).sum().item()
    fp = (preds * (1 - targets) * masks).sum().item()
    fn = (((1 - preds) * targets) * masks).sum().item()

    dice = _safe_divide(2 * tp + smooth, 2 * tp + fp + fn + smooth)
    iou = _safe_divide(tp + smooth, tp + fp + fn + smooth)
    sensitivity = _safe_divide(tp + smooth, tp + fn + smooth)
    specificity = _safe_divide(tn + smooth, tn + fp + smooth)
    accuracy = _safe_divide(tp + tn + smooth, tp + tn + fp + fn + smooth)
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
    }


def compute_binary_metrics(logits, targets, masks, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    return compute_binary_metrics_from_probs(probs, targets, masks, threshold=threshold, smooth=smooth)


def compute_mask_pair_metrics(preds, targets, masks, smooth=1e-6):
    preds = preds.float() * masks.float()
    return compute_binary_metrics_from_probs(preds, targets, masks, threshold=0.5, smooth=smooth)


def aggregate_metric_dicts(metrics_list):
    if not metrics_list:
        return {}
    aggregated = defaultdict(list)
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated[key].append(float(value))
    return {key: float(np.mean(values)) for key, values in aggregated.items()}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_prediction_image(prob_map, mask, path, threshold=0.5):
    ensure_dir(os.path.dirname(path))
    prob = np.clip(prob_map, 0.0, 1.0)
    binary = ((prob >= threshold).astype(np.uint8) * 255)
    if mask is not None:
        binary = binary * mask.astype(np.uint8)
    cv2.imwrite(path, binary)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_thresholds(start=0.30, end=0.70, step=0.05):
    values = np.arange(start, end + 1e-8, step)
    return [round(float(value), 2) for value in values]


@torch.no_grad()
def predict_probs(model, images, use_tta=False):
    if not use_tta:
        return torch.sigmoid(model(images))

    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
        (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
        (
            lambda x: torch.flip(torch.flip(x, dims=[-1]), dims=[-2]),
            lambda x: torch.flip(torch.flip(x, dims=[-1]), dims=[-2]),
        ),
    ]
    outputs = []
    for forward_transform, inverse_transform in transforms:
        probs = torch.sigmoid(model(forward_transform(images)))
        outputs.append(inverse_transform(probs))
    return torch.mean(torch.stack(outputs, dim=0), dim=0)
