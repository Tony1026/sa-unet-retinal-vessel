import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np

from dataset import resolve_data_root


def _binary(path):
    return cv2.imread(str(path), 0) > 127


def _drive_items(data_root):
    for image_path in sorted(Path(data_root, 'DRIVE', 'test', 'images').glob('*.tif')):
        image_id = image_path.name[:2]
        yield {
            'name': image_path.stem,
            'image': image_path,
            'label': Path(glob.glob(os.path.join(data_root, 'DRIVE', 'test', '1st_manual', f'*{image_id}*.*'))[0]),
            'mask': Path(glob.glob(os.path.join(data_root, 'DRIVE', 'test', 'mask', f'*{image_id}*.*'))[0]),
        }


def _chase_items(data_root):
    for image_path in sorted(Path(data_root, 'CHASEDB1').glob('*.jpg'))[20:]:
        yield {
            'name': image_path.stem,
            'image': image_path,
            'label': image_path.with_name(f'{image_path.stem}_1stHO.png'),
            'mask': image_path.with_name(f'{image_path.stem}_mask.png'),
        }


def _overlay(image_path, pred, label, mask):
    image = cv2.imread(str(image_path))
    if image.shape[:2] != pred.shape:
        image = cv2.resize(image, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image * 0.35).astype(np.uint8)

    pred = pred & mask
    label = label & mask
    tp = pred & label
    fp = pred & ~label
    fn = ~pred & label

    overlay = image.copy()
    overlay[tp] = np.array([40, 180, 90], dtype=np.uint8)
    overlay[fp] = np.array([230, 70, 70], dtype=np.uint8)
    overlay[fn] = np.array([60, 120, 240], dtype=np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description='Create TP/FP/FN overlays for test predictions')
    parser.add_argument('--dataset', choices=['drive', 'chase'], required=True)
    parser.add_argument('--prediction-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = _drive_items(data_root) if args.dataset == 'drive' else _chase_items(data_root)

    for item in items:
        pred_path = Path(args.prediction_dir, f"{item['name']}.png")
        if not pred_path.exists():
            nested = list(Path(args.prediction_dir).rglob(f"{item['name']}.png"))
            if not nested:
                raise FileNotFoundError(f'Missing prediction for {item["name"]}: {pred_path}')
            pred_path = nested[0]

        pred = _binary(pred_path)
        label = _binary(item['label'])
        mask = _binary(item['mask'])
        if label.shape != pred.shape:
            label = cv2.resize(label.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mask = cv2.resize(mask.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        cv2.imwrite(str(output_dir / f"{item['name']}_error.png"), _overlay(item['image'], pred, label, mask))


if __name__ == '__main__':
    main()
