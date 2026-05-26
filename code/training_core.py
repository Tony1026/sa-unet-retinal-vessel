import os

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from experiment_utils import (
    aggregate_metric_dicts,
    build_thresholds,
    compute_binary_metrics,
    compute_binary_metrics_from_probs,
    compute_loss,
    compute_mask_pair_metrics,
    ensure_dir,
    save_prediction_image,
)


THRESHOLD_CANDIDATES = build_thresholds()
GFLOW_ALIGN_WEIGHT = float(os.environ.get('GFLOW_ALIGN_WEIGHT', '0.15'))
GFLOW_CONS_WEIGHT = float(os.environ.get('GFLOW_CONS_WEIGHT', '0.01'))
GFLOW_SPARSE_WEIGHT = float(os.environ.get('GFLOW_SPARSE_WEIGHT', '0.02'))
GFLOW_MISSING_SECOND_WEIGHT = float(os.environ.get('GFLOW_MISSING_SECOND_WEIGHT', '0.35'))
GFLOW_SECOND_WEIGHT = float(os.environ.get('GFLOW_SECOND_WEIGHT', '1.0'))


class EarlyStopping:
    def __init__(self, patience):
        self.patience = max(0, int(patience))
        self.best_value = None
        self.bad_epoch_count = 0

    def step(self, value):
        current_value = float(value)
        if self.best_value is None or current_value < self.best_value:
            self.best_value = current_value
            self.bad_epoch_count = 0
            return False

        self.bad_epoch_count += 1
        return self.bad_epoch_count >= self.patience


def move_batch(batch, device):
    return {
        'image': batch['image'].to(device),
        'label': batch['label'].to(device),
        'label_first': batch['label_first'].to(device),
        'label_second': batch['label_second'].to(device),
        'mask': batch['mask'].to(device),
        'valid_region': batch['valid_region'].to(device),
        'dataset': batch['dataset'],
        'name': batch['name'],
        'has_label': batch['has_label'].to(device),
        'has_second_label': batch['has_second_label'].to(device),
    }


def primary_logits(output):
    if isinstance(output, dict):
        return output['logits']
    return output


def _select_output_logits(output, output_key='logits'):
    if isinstance(output, dict):
        return output[output_key]
    if output_key != 'logits':
        return None
    return output


def _masked_soft_dice_loss(probs, targets, masks, smooth=1e-6):
    masks = masks.float()
    probs = probs.float() * masks
    targets = targets.float() * masks
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def compute_model_loss(output, batch, loss_mode='bce_dice'):
    if not isinstance(output, dict):
        return compute_loss(output, batch['label_first'], batch['mask'], loss_mode=loss_mode)

    sink1 = output['sink1']
    sink2 = output['sink2']
    loss = compute_loss(output['logits'], batch['label_first'], batch['mask'], loss_mode=loss_mode)
    loss = loss + compute_loss(sink1, batch['label_first'], batch['mask'], loss_mode=loss_mode)

    second_flags = batch['has_second_label'].bool()
    if second_flags.any():
        loss = loss + GFLOW_SECOND_WEIGHT * compute_loss(
            sink2[second_flags],
            batch['label_second'][second_flags],
            batch['mask'][second_flags],
            loss_mode=loss_mode,
        )
        pred_diff = torch.abs(torch.sigmoid(sink1[second_flags]) - torch.sigmoid(sink2[second_flags]))
        target_diff = torch.abs(batch['label_first'][second_flags] - batch['label_second'][second_flags])
        loss = loss + GFLOW_ALIGN_WEIGHT * _masked_soft_dice_loss(pred_diff, target_diff, batch['mask'][second_flags])
    else:
        loss = loss + GFLOW_MISSING_SECOND_WEIGHT * compute_loss(sink2, batch['label_first'], batch['mask'], loss_mode=loss_mode)
        pred_diff = torch.abs(torch.sigmoid(sink1) - torch.sigmoid(sink2))
        loss = loss + GFLOW_ALIGN_WEIGHT * ((pred_diff * batch['mask']).pow(2).sum() / batch['mask'].sum().clamp(min=1.0))

    loss = loss + GFLOW_CONS_WEIGHT * output['flow_conservation_loss']
    loss = loss + GFLOW_SPARSE_WEIGHT * output['flow_sparse_loss']
    return loss


def run_epoch(model, loader, optimizer, device, amp_enabled, loss_mode='bce_dice'):
    model.train()
    scaler = GradScaler(device='cuda', enabled=amp_enabled)
    loss_sum = 0.0
    metrics_buffer = []

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            output = model(batch['image'])
            logits = primary_logits(output)
            loss = compute_model_loss(output, batch, loss_mode=loss_mode)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        metrics_buffer.append(compute_binary_metrics(logits.detach(), batch['label_first'], batch['mask']))

    return {
        'loss': loss_sum / max(1, len(loader)),
        **aggregate_metric_dicts(metrics_buffer),
    }


def _window_positions(length, window_size, stride):
    if length <= window_size:
        return [0]
    positions = list(range(0, length - window_size + 1, stride))
    if positions[-1] != length - window_size:
        positions.append(length - window_size)
    return positions


@torch.no_grad()
def sliding_window_predict_probs(model, images, window_size=256, stride=128, output_key='logits'):
    batch_size, _, height, width = images.shape
    pad_h = max(window_size - height, 0)
    pad_w = max(window_size - width, 0)
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')

    padded_height, padded_width = images.shape[-2:]
    probs_sum = torch.zeros((batch_size, 1, padded_height, padded_width), device=images.device)
    counts = torch.zeros_like(probs_sum)

    y_positions = _window_positions(padded_height, window_size, stride)
    x_positions = _window_positions(padded_width, window_size, stride)

    for top in y_positions:
        for left in x_positions:
            patch = images[:, :, top:top + window_size, left:left + window_size]
            output = model(patch)
            logits = _select_output_logits(output, output_key=output_key)
            if logits is None:
                return None
            probs = torch.sigmoid(logits)
            probs_sum[:, :, top:top + window_size, left:left + window_size] += probs
            counts[:, :, top:top + window_size, left:left + window_size] += 1

    probs = probs_sum / counts.clamp(min=1.0)
    return probs[:, :, :height, :width]


def _crop_to_valid_region(tensor, valid_region):
    top, left, height, width = [int(value) for value in valid_region]
    return tensor[..., top:top + height, left:left + width]


@torch.no_grad()
def collect_records(
    model,
    loader,
    device,
    inference_mode,
    threshold=0.5,
    output_dir=None,
    window_size=256,
    stride=128,
    loss_mode='bce_dice',
):
    model.eval()
    records = []
    loss_sum = 0.0
    labeled_batches = 0

    for batch in loader:
        batch = move_batch(batch, device)
        if inference_mode == 'sliding_window':
            probs = sliding_window_predict_probs(model, batch['image'], window_size=window_size, stride=stride)
            logits = torch.logit(probs.clamp(min=1e-6, max=1 - 1e-6))
            sink1_probs = sliding_window_predict_probs(
                model,
                batch['image'],
                window_size=window_size,
                stride=stride,
                output_key='sink1',
            )
            sink2_probs = sliding_window_predict_probs(
                model,
                batch['image'],
                window_size=window_size,
                stride=stride,
                output_key='sink2',
            )
        else:
            output = model(batch['image'])
            logits = primary_logits(output)
            probs = torch.sigmoid(logits)
            sink1_logits = _select_output_logits(output, output_key='sink1')
            sink2_logits = _select_output_logits(output, output_key='sink2')
            sink1_probs = torch.sigmoid(sink1_logits) if sink1_logits is not None else None
            sink2_probs = torch.sigmoid(sink2_logits) if sink2_logits is not None else None

        labeled_flags = batch['has_label'].bool()
        if labeled_flags.any():
            loss = compute_loss(
                logits[labeled_flags],
                batch['label_first'][labeled_flags],
                batch['mask'][labeled_flags],
                loss_mode=loss_mode,
            )
            loss_sum += loss.item()
            labeled_batches += 1

        for index in range(batch['image'].shape[0]):
            valid_region = batch['valid_region'][index].detach().cpu().tolist()
            record = {
                'dataset': batch['dataset'][index],
                'name': batch['name'][index],
                'probs': _crop_to_valid_region(probs[index:index + 1].detach().cpu(), valid_region),
                'sink1_probs': (
                    _crop_to_valid_region(sink1_probs[index:index + 1].detach().cpu(), valid_region)
                    if sink1_probs is not None
                    else None
                ),
                'sink2_probs': (
                    _crop_to_valid_region(sink2_probs[index:index + 1].detach().cpu(), valid_region)
                    if sink2_probs is not None
                    else None
                ),
                'label_first': _crop_to_valid_region(batch['label_first'][index:index + 1].detach().cpu(), valid_region),
                'label_second': _crop_to_valid_region(batch['label_second'][index:index + 1].detach().cpu(), valid_region),
                'mask': _crop_to_valid_region(batch['mask'][index:index + 1].detach().cpu(), valid_region),
                'valid_region': valid_region,
                'has_label': bool(batch['has_label'][index].item()),
                'has_second_label': bool(batch['has_second_label'][index].item()),
            }
            records.append(record)

            if output_dir is not None:
                prob_map = record['probs'][0, 0].numpy()
                fov_mask = record['mask'][0, 0].numpy() > 0.5
                save_prediction_image(
                    prob_map,
                    fov_mask,
                    os.path.join(output_dir, record['dataset'], f"{record['name']}.png"),
                    threshold=threshold,
                )

    meta = {}
    if labeled_batches > 0:
        meta['loss'] = loss_sum / labeled_batches
    return records, meta


def summarize_records(records, observer_key, threshold, loss=None, prob_key='probs'):
    metrics_buffer = []
    dataset_metrics = {}

    for record in records:
        if record.get(prob_key) is None:
            continue
        if observer_key == 'label_first' and not record['has_label']:
            continue
        if observer_key == 'label_second' and not record['has_second_label']:
            continue
        sample_metrics = compute_binary_metrics_from_probs(record[prob_key], record[observer_key], record['mask'], threshold=threshold)
        dataset_metrics.setdefault(record['dataset'], []).append(sample_metrics)
        metrics_buffer.append(sample_metrics)

    summary = {'labeled_samples': len(metrics_buffer)}
    if loss is not None:
        summary['loss'] = loss
    summary.update(aggregate_metric_dicts(metrics_buffer))
    if dataset_metrics:
        summary['per_dataset'] = {
            dataset_name: aggregate_metric_dicts(dataset_values)
            for dataset_name, dataset_values in dataset_metrics.items()
        }
    return summary


def summarize_sink_disagreement(records, smooth=1e-6):
    values = []
    mae_values = []
    corr_values = []
    dataset_values = {}

    for record in records:
        if record.get('sink1_probs') is None or record.get('sink2_probs') is None or not record['has_second_label']:
            continue

        mask = record['mask'].float()
        pred_diff = torch.abs(record['sink1_probs'] - record['sink2_probs']).float() * mask
        target_diff = torch.abs(record['label_first'] - record['label_second']).float() * mask

        intersection = (pred_diff * target_diff).sum().item()
        denominator = (pred_diff.sum() + target_diff.sum()).item()
        soft_dice = (2.0 * intersection + smooth) / (denominator + smooth)
        mae = (torch.abs(pred_diff - target_diff).sum() / mask.sum().clamp(min=1.0)).item()

        pred_flat = pred_diff[mask > 0.5].reshape(-1)
        target_flat = target_diff[mask > 0.5].reshape(-1)
        if pred_flat.numel() > 1 and float(pred_flat.std()) > 1e-8 and float(target_flat.std()) > 1e-8:
            corr = float(torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item())
        else:
            corr = 0.0

        item = {'soft_dice': soft_dice, 'mae': mae, 'pearson': corr}
        dataset_values.setdefault(record['dataset'], []).append(item)
        values.append(soft_dice)
        mae_values.append(mae)
        corr_values.append(corr)

    summary = {'labeled_samples': len(values)}
    if values:
        summary.update(
            {
                'soft_dice': float(sum(values) / len(values)),
                'mae': float(sum(mae_values) / len(mae_values)),
                'pearson': float(sum(corr_values) / len(corr_values)),
            }
        )
        summary['per_dataset'] = {
            dataset_name: {
                'soft_dice': float(sum(item['soft_dice'] for item in items) / len(items)),
                'mae': float(sum(item['mae'] for item in items) / len(items)),
                'pearson': float(sum(item['pearson'] for item in items) / len(items)),
            }
            for dataset_name, items in dataset_values.items()
        }
    return summary


def summarize_inter_observer(records):
    metrics_buffer = []
    dataset_metrics = {}

    for record in records:
        if not record['has_second_label']:
            continue
        sample_metrics = compute_mask_pair_metrics(record['label_first'], record['label_second'], record['mask'])
        dataset_metrics.setdefault(record['dataset'], []).append(sample_metrics)
        metrics_buffer.append(sample_metrics)

    summary = {'labeled_samples': len(metrics_buffer)}
    summary.update(aggregate_metric_dicts(metrics_buffer))
    if dataset_metrics:
        summary['per_dataset'] = {
            dataset_name: aggregate_metric_dicts(dataset_values)
            for dataset_name, dataset_values in dataset_metrics.items()
        }
    return summary


def select_threshold(records, loss=None):
    best_threshold = THRESHOLD_CANDIDATES[0]
    best_metrics = None
    for threshold in THRESHOLD_CANDIDATES:
        metrics = summarize_records(records, 'label_first', threshold, loss=loss)
        if best_metrics is None or metrics.get('dice', 0.0) > best_metrics.get('dice', 0.0):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_split(
    model,
    loader,
    device,
    inference_mode,
    threshold,
    output_dir=None,
    window_size=256,
    stride=128,
    loss_mode='bce_dice',
):
    ensure_dir(output_dir) if output_dir is not None else None
    records, meta = collect_records(
        model,
        loader,
        device,
        inference_mode,
        threshold=threshold,
        output_dir=output_dir,
        window_size=window_size,
        stride=stride,
        loss_mode=loss_mode,
    )
    return {
        'primary': summarize_records(records, 'label_first', threshold, loss=meta.get('loss')),
        'secondary': summarize_records(records, 'label_second', threshold),
        'sink1_primary': summarize_records(records, 'label_first', threshold, prob_key='sink1_probs'),
        'sink1_secondary': summarize_records(records, 'label_second', threshold, prob_key='sink1_probs'),
        'sink2_primary': summarize_records(records, 'label_first', threshold, prob_key='sink2_probs'),
        'sink2_secondary': summarize_records(records, 'label_second', threshold, prob_key='sink2_probs'),
        'sink_disagreement': summarize_sink_disagreement(records),
        'inter_observer': summarize_inter_observer(records),
        'records': records,
        'meta': meta,
    }
