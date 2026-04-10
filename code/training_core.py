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
        'dataset': batch['dataset'],
        'name': batch['name'],
        'has_label': batch['has_label'].to(device),
        'has_second_label': batch['has_second_label'].to(device),
    }


def run_epoch(model, loader, optimizer, device, amp_enabled):
    model.train()
    scaler = GradScaler(device='cuda', enabled=amp_enabled)
    loss_sum = 0.0
    metrics_buffer = []

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch['image'])
            loss = compute_loss(logits, batch['label_first'], batch['mask'], loss_mode='bce_dice')

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
def sliding_window_predict_probs(model, images, window_size=256, stride=128):
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
            probs = torch.sigmoid(model(patch))
            probs_sum[:, :, top:top + window_size, left:left + window_size] += probs
            counts[:, :, top:top + window_size, left:left + window_size] += 1

    probs = probs_sum / counts.clamp(min=1.0)
    return probs[:, :, :height, :width]


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
        else:
            logits = model(batch['image'])
            probs = torch.sigmoid(logits)

        labeled_flags = batch['has_label'].bool()
        if labeled_flags.any():
            loss = compute_loss(logits[labeled_flags], batch['label_first'][labeled_flags], batch['mask'][labeled_flags], loss_mode='bce_dice')
            loss_sum += loss.item()
            labeled_batches += 1

        for index in range(batch['image'].shape[0]):
            record = {
                'dataset': batch['dataset'][index],
                'name': batch['name'][index],
                'probs': probs[index:index + 1].detach().cpu(),
                'label_first': batch['label_first'][index:index + 1].detach().cpu(),
                'label_second': batch['label_second'][index:index + 1].detach().cpu(),
                'mask': batch['mask'][index:index + 1].detach().cpu(),
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


def summarize_records(records, observer_key, threshold, loss=None):
    metrics_buffer = []
    dataset_metrics = {}

    for record in records:
        if observer_key == 'label_first' and not record['has_label']:
            continue
        if observer_key == 'label_second' and not record['has_second_label']:
            continue
        sample_metrics = compute_binary_metrics_from_probs(record['probs'], record[observer_key], record['mask'], threshold=threshold)
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
    )
    return {
        'primary': summarize_records(records, 'label_first', threshold, loss=meta.get('loss')),
        'secondary': summarize_records(records, 'label_second', threshold),
        'inter_observer': summarize_inter_observer(records),
        'records': records,
        'meta': meta,
    }
