import argparse
import csv
import json
import sys
from pathlib import Path


SUMMARY_FIELDS = [
    'dataset_name',
    'model_name',
    'input_mode',
    'loss_mode',
    'parameter_count',
    'epochs_ran',
    'best_epoch',
    'best_val_dice',
    'selected_threshold',
    'test_labeled_samples',
    'test_dice',
    'test_iou',
    'test_sensitivity',
    'test_specificity',
    'test_accuracy',
    'metrics_path',
]


def _test_key(dataset_name):
    if dataset_name == 'drive':
        return 'drive_test'
    if dataset_name == 'chase':
        return 'chase_test'
    return None


def collect_row(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    dataset_name = data.get('dataset_name')
    test_section = data.get(_test_key(dataset_name) or '', {})
    primary = test_section.get('primary', {})
    return {
        'dataset_name': dataset_name,
        'model_name': data.get('model_name'),
        'input_mode': data.get('input_mode'),
        'loss_mode': data.get('loss_mode'),
        'parameter_count': data.get('parameter_count'),
        'epochs_ran': data.get('epochs_ran'),
        'best_epoch': data.get('best_epoch'),
        'best_val_dice': data.get('best_val_dice'),
        'selected_threshold': data.get('selected_threshold'),
        'test_labeled_samples': primary.get('labeled_samples'),
        'test_dice': primary.get('dice'),
        'test_iou': primary.get('iou'),
        'test_sensitivity': primary.get('sensitivity'),
        'test_specificity': primary.get('specificity'),
        'test_accuracy': primary.get('accuracy'),
        'metrics_path': str(path),
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize retinal vessel experiment metrics')
    parser.add_argument('--root', type=str, default='output')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = [collect_row(path) for path in sorted(root.rglob('metrics.json'))]
    rows.sort(key=lambda row: (row.get('dataset_name') or '', row.get('model_name') or '', row.get('input_mode') or '', row.get('loss_mode') or ''))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
