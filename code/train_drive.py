import argparse
import os
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import DriveDataset, build_train_transform
from experiment_utils import apply_training_defaults, count_parameters, default_batch_size, save_json, seed_everything, select_device
from model_factory import create_model
from training_core import EarlyStopping, THRESHOLD_CANDIDATES, collect_records, evaluate_split, run_epoch, select_threshold


def parse_args():
    parser = argparse.ArgumentParser(description='Train retinal vessel models on DRIVE')
    parser.add_argument('--model-name', type=str, default='sa_unet', choices=['sa_unet', 'sa_unetv2'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--drive-train-ratio', type=float, default=0.8)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--drop-prob', type=float, default=None)
    parser.add_argument('--block-size', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=592)
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
        DriveDataset(split='train', image_size=args.image_size, train_ratio=args.drive_train_ratio, seed=args.seed, transform=build_train_transform()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        DriveDataset(split='val', image_size=args.image_size, train_ratio=args.drive_train_ratio, seed=args.seed),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        DriveDataset(split='drive_test', image_size=args.image_size),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir or f'../output/drive/{args.model_name}'))
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    prediction_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = create_model(
        args.model_name,
        in_channels=1,
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
        train_metrics = run_epoch(model, train_loader, optimizer, device, amp_enabled)
        val_records, val_meta = collect_records(model, val_loader, device, inference_mode='full_image')
        selected_threshold, val_metrics = select_threshold(val_records, loss=val_meta.get('loss'))
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
            f"[DRIVE/{args.model_name}] Epoch {epoch:03d} | "
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
                        'dataset_name': 'drive',
                        'input_mode': 'green_clahe',
                        'loss_mode': 'bce_dice',
                        'resize_to': args.image_size,
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
                f"[DRIVE/{args.model_name}] Early stopping triggered at epoch {epoch:03d} "
                f"after {args.early_stop_patience} epochs without val_loss improvement."
            )
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    selected_threshold = float(checkpoint['selected_threshold'])
    epochs_ran = len(history)

    val_summary = evaluate_split(model, val_loader, device, 'full_image', selected_threshold)
    drive_test_summary = evaluate_split(
        model,
        test_loader,
        device,
        'full_image',
        selected_threshold,
        output_dir=os.path.join(prediction_dir, 'drive_test'),
    )

    metrics = {
        'model_name': args.model_name,
        'dataset_name': 'drive',
        'device': device_type,
        'batch_size': batch_size,
        'parameter_count': count_parameters(model),
        'input_mode': 'green_clahe',
        'loss_mode': 'bce_dice',
        'resize_to': args.image_size,
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
        'val_primary': val_summary['primary'],
        'drive_test': {
            'primary': drive_test_summary['primary'],
            'secondary': drive_test_summary['secondary'],
            'inter_observer': drive_test_summary['inter_observer'],
        },
    }
    save_json(metrics, os.path.join(output_dir, 'metrics.json'))
    print(f"Best checkpoint: {best_path}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'metrics.json')}")


if __name__ == '__main__':
    main()
