import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import ChaseFullImageDataset, DriveDataset, input_channels_for_mode
from experiment_utils import ensure_dir, save_prediction_image, select_device
from model_factory import create_model
from training_core import _crop_to_valid_region, sliding_window_predict_probs


def parse_args():
    parser = argparse.ArgumentParser(description='Export predictions from a trained retinal vessel checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    return parser.parse_args()


def build_dataset(model_args, split, data_root=None):
    dataset_name = model_args.get('dataset_name')
    input_mode = model_args.get('input_mode', 'green_clahe')
    if dataset_name == 'drive':
        resolved_split = split or 'drive_test'
        return (
            DriveDataset(
                split=resolved_split,
                image_size=model_args.get('resize_to', 592),
                input_mode=input_mode,
                data_root=data_root,
            ),
            'full_image',
            resolved_split,
        )
    if dataset_name == 'chase':
        resolved_split = split or 'test'
        return (
            ChaseFullImageDataset(split=resolved_split, input_mode=input_mode, data_root=data_root),
            'sliding_window',
            resolved_split,
        )
    raise ValueError(f"Unsupported checkpoint dataset: {dataset_name}")


@torch.no_grad()
def main():
    args = parse_args()
    device, _ = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get('args', {})
    model = create_model(
        model_args['model_name'],
        in_channels=input_channels_for_mode(model_args.get('input_mode', 'green_clahe')),
        base_channels=model_args.get('base_channels', 16),
        drop_prob=model_args.get('drop_prob', 0.18),
        block_size=model_args.get('block_size', 7),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset, inference_mode, resolved_split = build_dataset(model_args, args.split, data_root=args.data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    threshold = args.threshold if args.threshold is not None else checkpoint.get('selected_threshold', 0.5)
    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            args.output_dir or f"../output/{model_args['dataset_name']}/{model_args['model_name']}/manual_predictions/{resolved_split}",
        )
    )

    for batch in loader:
        image = batch['image'].to(device)
        if inference_mode == 'sliding_window':
            probs = sliding_window_predict_probs(
                model,
                image,
                window_size=model_args.get('patch_size', 256),
                stride=model_args.get('sliding_window_stride', 128),
            )
        else:
            probs = torch.sigmoid(model(image))
        valid_region = batch['valid_region'][0].detach().cpu().tolist()
        probs = _crop_to_valid_region(probs[0:1].detach().cpu(), valid_region)
        mask = _crop_to_valid_region(batch['mask'][0:1].detach().cpu(), valid_region)[0, 0].numpy() > 0.5
        prob_map = probs[0, 0].numpy()
        dataset_name = batch['dataset'][0]
        name = batch['name'][0]
        target_path = os.path.join(output_dir, dataset_name)
        ensure_dir(target_path)
        save_prediction_image(prob_map, mask, os.path.join(target_path, f'{name}.png'), threshold=threshold)

    print(f"Predictions saved to: {output_dir}")
    print(f"Model: {model_args['model_name']} | Dataset: {model_args['dataset_name']} | Threshold used: {threshold:.2f}")


if __name__ == '__main__':
    main()
