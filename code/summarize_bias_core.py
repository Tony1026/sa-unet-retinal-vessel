import argparse
import csv
import json
import os
from collections import defaultdict


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _metric(metrics, split, group, key):
    return metrics.get(split, {}).get(group, {}).get(key, '')


def _score(row):
    values = []
    for key, weight in (
        ('primary_dice', 0.25),
        ('secondary_dice', 0.35),
        ('sink2_secondary_dice', 0.30),
        ('sink_disagreement_soft_dice', 0.10),
    ):
        value = row.get(key, '')
        if value == '':
            continue
        values.append(float(value) * weight)
    return sum(values) if values else ''


def collect_rows(root_dir):
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'metrics.json' not in filenames:
            continue
        metrics_path = os.path.join(dirpath, 'metrics.json')
        metrics = _load_json(metrics_path)
        dataset = metrics.get('dataset_name', '')
        split = 'drive_test' if dataset == 'drive' else 'chase_test'
        row = {
            'dataset': dataset.upper(),
            'model_name': metrics.get('model_name', ''),
            'label_bias_mode': metrics.get('label_bias_mode', ''),
            'seed': metrics.get('seed', ''),
            'primary_dice': _metric(metrics, split, 'primary', 'dice'),
            'secondary_dice': _metric(metrics, split, 'secondary', 'dice'),
            'sink1_primary_dice': _metric(metrics, split, 'sink1_primary', 'dice'),
            'sink2_secondary_dice': _metric(metrics, split, 'sink2_secondary', 'dice'),
            'sink_cross_gap': '',
            'sink_disagreement_soft_dice': _metric(metrics, split, 'sink_disagreement', 'soft_dice'),
            'sink_disagreement_mae': _metric(metrics, split, 'sink_disagreement', 'mae'),
            'sink_disagreement_pearson': _metric(metrics, split, 'sink_disagreement', 'pearson'),
            'inter_observer_dice': _metric(metrics, split, 'inter_observer', 'dice'),
            'best_val_dice': metrics.get('best_val_dice', ''),
            'epochs_ran': metrics.get('epochs_ran', ''),
            'selected_threshold': metrics.get('selected_threshold', ''),
            'parameter_count': metrics.get('parameter_count', ''),
            'metrics_path': os.path.relpath(metrics_path, root_dir),
        }
        if row['sink1_primary_dice'] != '' and row['sink2_secondary_dice'] != '':
            row['sink_cross_gap'] = float(row['sink2_secondary_dice']) - float(row['sink1_primary_dice'])
        row['bias_core_score'] = _score(row)
        rows.append(row)
    return sorted(rows, key=lambda item: (item['dataset'], item['model_name'], item['label_bias_mode'], str(item['seed'])))


def main():
    parser = argparse.ArgumentParser(description='Summarize bias-core GFlow ablations.')
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--output-csv', required=True)
    args = parser.parse_args()

    rows = collect_rows(args.root_dir)
    fieldnames = [
        'dataset',
        'model_name',
        'label_bias_mode',
        'seed',
        'bias_core_score',
        'primary_dice',
        'secondary_dice',
        'sink1_primary_dice',
        'sink2_secondary_dice',
        'sink_cross_gap',
        'sink_disagreement_soft_dice',
        'sink_disagreement_mae',
        'sink_disagreement_pearson',
        'inter_observer_dice',
        'best_val_dice',
        'epochs_ran',
        'selected_threshold',
        'parameter_count',
        'metrics_path',
    ]
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output_csv}')

    aggregate_path = os.path.join(os.path.dirname(args.output_csv), 'aggregate.csv')
    groups = defaultdict(list)
    numeric_keys = [
        'bias_core_score',
        'primary_dice',
        'secondary_dice',
        'sink1_primary_dice',
        'sink2_secondary_dice',
        'sink_cross_gap',
        'sink_disagreement_soft_dice',
        'sink_disagreement_mae',
        'sink_disagreement_pearson',
        'inter_observer_dice',
        'best_val_dice',
        'epochs_ran',
        'parameter_count',
    ]
    for row in rows:
        groups[(row['dataset'], row['model_name'], row['label_bias_mode'])].append(row)

    aggregate_rows = []
    for (dataset, model_name, label_bias_mode), group_rows in sorted(groups.items()):
        aggregate = {
            'dataset': dataset,
            'model_name': model_name,
            'label_bias_mode': label_bias_mode,
            'runs': len(group_rows),
            'seeds': ';'.join(str(row.get('seed', '')) for row in group_rows),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows if row.get(key, '') != '']
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                aggregate[f'{key}_mean'] = mean
                aggregate[f'{key}_std'] = variance ** 0.5
            else:
                aggregate[f'{key}_mean'] = ''
                aggregate[f'{key}_std'] = ''
        aggregate_rows.append(aggregate)

    aggregate_fieldnames = ['dataset', 'model_name', 'label_bias_mode', 'runs', 'seeds']
    for key in numeric_keys:
        aggregate_fieldnames.extend([f'{key}_mean', f'{key}_std'])
    with open(aggregate_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f'Wrote {len(aggregate_rows)} rows to {aggregate_path}')


if __name__ == '__main__':
    main()
