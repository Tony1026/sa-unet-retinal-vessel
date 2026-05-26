import argparse
import os
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import ChaseFullImageDataset, ChasePatchDataset, build_train_transform, input_channels_for_mode
from experiment_utils import apply_training_defaults, count_parameters, default_batch_size, save_json, seed_everything, select_device
from model_factory import available_models, create_model
from training_core import EarlyStopping, THRESHOLD_CANDIDATES, evaluate_split, run_epoch, select_threshold


def parse_args():
    parser = argparse.ArgumentParser(description='Train retinal vessel models on CHASEDB1')
    parser.add_argument('--model-name', type=str, default='sa_unet', choices=available_models())
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chase-train-ratio', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--input-mode', type=str, default='green_clahe', choices=['green_clahe', 'green', 'rgb'])
    parser.add_argument('--loss-mode', type=str, default='bce_dice', choices=['bce', 'bce_dice'])
    parser.add_argument('--label-bias-mode', type=str, default='none', choices=['none', 'second', 'random_primary'])
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--drop-prob', type=float, default=None)
    parser.add_argument('--block-size', type=int, default=None)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--sliding-window-stride', type=int, default=128)
    parser.add_argument('--patches-per-image', type=int, default=8)
    parser.add_argument('--plateau-patience', type=int, default=None)
    parser.add_argument('--early-stop-patience', type=int, default=None)
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--lr-factor', type=float, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    return apply_training_defaults(args)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device, device_type = select_device(args.device)
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size(device_type)

    train_loader = DataLoader(
        ChasePatchDataset(
            split='train',
            patch_size=args.patch_size,
            transform=build_train_transform(),
            patches_per_image=args.patches_per_image,
            train_ratio=args.chase_train_ratio,
            seed=args.seed,
            input_mode=args.input_mode,
            data_root=args.data_root,
            label_bias_mode=args.label_bias_mode,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ChaseFullImageDataset(
            split='val',
            train_ratio=args.chase_train_ratio,
            seed=args.seed,
            input_mode=args.input_mode,
            data_root=args.data_root,
            label_bias_mode=args.label_bias_mode,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_eval_loader = DataLoader(
        ChaseFullImageDataset(
            split='train_eval',
            train_ratio=args.chase_train_ratio,
            seed=args.seed,
            input_mode=args.input_mode,
            data_root=args.data_root,
            label_bias_mode=args.label_bias_mode,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        ChaseFullImageDataset(
            split='test',
            train_ratio=args.chase_train_ratio,
            seed=args.seed,
            input_mode=args.input_mode,
            data_root=args.data_root,
            label_bias_mode=args.label_bias_mode,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir or f'../output/chase/{args.model_name}'))
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    prediction_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = create_model(
        args.model_name,
        in_channels=input_channels_for_mode(args.input_mode),
        base_channels=args.base_channels,
        drop_prob=args.drop_prob,
        block_size=args.block_size,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.plateau_patience,
        min_lr=args.min_lr,
    )
    early_stopper = EarlyStopping(args.early_stop_patience)
    amp_enabled = device_type == 'cuda'

    history = []
    best_val_dice = -1.0
    best_epoch = None
    best_path = os.path.join(checkpoint_dir, 'best.pt')
    stop_epoch = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, amp_enabled, loss_mode=args.loss_mode)
        val_summary = evaluate_split(
            model,
            val_loader,
            device,
            'sliding_window',
            threshold=0.5,
            window_size=args.patch_size,
            stride=args.sliding_window_stride,
            loss_mode=args.loss_mode,
        )
        selected_threshold, val_metrics = select_threshold(val_summary['records'], loss=val_summary['meta'].get('loss'))
        val_loss = float(val_metrics['loss'])
        scheduler.step(val_loss)
        should_stop = early_stopper.step(val_loss)
        history.append(
            {
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'selected_threshold': selected_threshold,
                'train': train_metrics,
                'val_primary': val_metrics,
            }
        )
        print(
            f"[CHASE/{args.model_name}] Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_loss:.4f} | val_dice={val_metrics.get('dice', 0.0):.4f} | "
            f"threshold={selected_threshold:.2f}"
        )

        if val_metrics.get('dice', 0.0) > best_val_dice:
            best_val_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'args': {
                        'model_name': args.model_name,
                        'dataset_name': 'chase',
                        'seed': args.seed,
                        'input_mode': args.input_mode,
                        'loss_mode': args.loss_mode,
                        'label_bias_mode': args.label_bias_mode,
                        'patch_size': args.patch_size,
                        'sliding_window_stride': args.sliding_window_stride,
                        'base_channels': args.base_channels,
                        'drop_prob': args.drop_prob,
                        'block_size': args.block_size,
                    },
                    'epoch': epoch,
                    'best_val_dice': best_val_dice,
                    'selected_threshold': selected_threshold,
                },
                best_path,
            )

        if should_stop:
            stop_epoch = epoch
            print(
                f"[CHASE/{args.model_name}] Early stopping triggered at epoch {epoch:03d} "
                f"after {args.early_stop_patience} epochs without val_loss improvement."
            )
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs_ran = len(history)

    train_eval_summary = evaluate_split(
        model,
        train_eval_loader,
        device,
        'sliding_window',
        threshold=0.5,
        window_size=args.patch_size,
        stride=args.sliding_window_stride,
        loss_mode=args.loss_mode,
    )
    selected_threshold = checkpoint.get('selected_threshold')
    if selected_threshold is None:
        val_summary = evaluate_split(
            model,
            val_loader,
            device,
            'sliding_window',
            threshold=0.5,
            window_size=args.patch_size,
            stride=args.sliding_window_stride,
            loss_mode=args.loss_mode,
        )
        selected_threshold, _ = select_threshold(val_summary['records'], loss=val_summary['meta'].get('loss'))
    train_eval_primary = train_eval_summary['primary']
    val_summary = evaluate_split(
        model,
        val_loader,
        device,
        'sliding_window',
        threshold=selected_threshold,
        window_size=args.patch_size,
        stride=args.sliding_window_stride,
        loss_mode=args.loss_mode,
    )
    chase_test_summary = evaluate_split(
        model,
        test_loader,
        device,
        'sliding_window',
        threshold=selected_threshold,
        output_dir=os.path.join(prediction_dir, 'chase_test'),
        window_size=args.patch_size,
        stride=args.sliding_window_stride,
        loss_mode=args.loss_mode,
    )
    train_eval_metrics_summary = {key: value for key, value in train_eval_summary.items() if key not in ('records', 'meta')}
    val_metrics_summary = {key: value for key, value in val_summary.items() if key not in ('records', 'meta')}
    chase_test_metrics_summary = {key: value for key, value in chase_test_summary.items() if key not in ('records', 'meta')}

    metrics = {
        'model_name': args.model_name,
        'dataset_name': 'chase',
        'seed': args.seed,
        'device': device_type,
        'batch_size': batch_size,
        'data_root': args.data_root,
        'train_ratio': args.chase_train_ratio,
        'train_samples': len(train_loader.dataset.items),
        'val_samples': len(val_loader.dataset.items),
        'train_eval_samples': len(train_eval_loader.dataset.items),
        'test_samples': len(test_loader.dataset.items),
        'train_names': [item['name'] for item in train_loader.dataset.items],
        'val_names': [item['name'] for item in val_loader.dataset.items],
        'train_eval_names': [item['name'] for item in train_eval_loader.dataset.items],
        'test_names': [item['name'] for item in test_loader.dataset.items],
        'parameter_count': count_parameters(model),
        'input_mode': args.input_mode,
        'loss_mode': args.loss_mode,
        'label_bias_mode': args.label_bias_mode,
        'patch_size': args.patch_size,
        'sliding_window_stride': args.sliding_window_stride,
        'epochs': epochs_ran,
        'max_epochs': args.epochs,
        'epochs_ran': epochs_ran,
        'elapsed_seconds': time.time() - start_time,
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
        'early_stopped': stop_epoch is not None,
        'selected_threshold': selected_threshold,
        'threshold_candidates': THRESHOLD_CANDIDATES,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': args.lr_factor,
        'scheduler_patience': args.plateau_patience,
        'min_lr': args.min_lr,
        'early_stop_patience': args.early_stop_patience,
        'history': history,
        'train_eval_primary': train_eval_primary,
        'train_eval': train_eval_metrics_summary,
        'val_primary': val_summary['primary'],
        'val': val_metrics_summary,
        'chase_test': chase_test_metrics_summary,
    }
    if hasattr(model, 'flow_report'):
        metrics['flow_report'] = model.flow_report()
    save_json(metrics, os.path.join(output_dir, 'metrics.json'))
    print(f"Best checkpoint: {best_path}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'metrics.json')}")


if __name__ == '__main__':
    main()
