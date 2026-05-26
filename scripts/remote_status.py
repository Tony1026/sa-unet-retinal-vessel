import csv
import re
import subprocess
from pathlib import Path


ROOT = Path('/data1/gushengda/retina-sa-unet-codex')
LOG_ROOT = ROOT / 'logs' / 'priority'
OUT_ROOT = ROOT / 'output_priority'


def latest_progress(path):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    interesting = [
        line for line in lines
        if 'Epoch' in line or 'Early stopping' in line or 'Metrics saved' in line or 'Traceback' in line or 'RuntimeError' in line or 'Error' in line
    ]
    return interesting[-3:]


print('processes:')
try:
    out = subprocess.check_output(['pgrep', '-af', 'run_priority_parallel|train_drive|train_chase'], text=True)
    print(out.strip())
except subprocess.CalledProcessError:
    print('none')

print('\nmetrics_json_count:', len(list(OUT_ROOT.rglob('metrics.json'))) if OUT_ROOT.exists() else 0)
summary = OUT_ROOT / 'summary.csv'
if summary.exists():
    with summary.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    print('summary_rows:', len(rows))
    for row in rows:
        print(
            f"{row.get('dataset_name')}/{row.get('model_name')}/"
            f"{row.get('input_mode')}/{row.get('loss_mode')}: "
            f"dice={row.get('test_dice')} acc={row.get('test_accuracy')}"
        )

print('\nlatest logs:')
for path in sorted(LOG_ROOT.glob('*.log')):
    print(f'==={path.name}===')
    progress = latest_progress(path)
    if not progress:
        print('(no progress lines yet)')
        continue
    for line in progress:
        print(line)
