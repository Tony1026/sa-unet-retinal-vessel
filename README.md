# SA-UNet Retinal Vessel Segmentation

基于 `SA-UNet` 和 `SA-UNetv2` 的眼底血管分割项目，当前面向两个公开数据集：

- `DRIVE`
- `CHASEDB1`

项目保留两条模型线：

- `sa_unet`：当前正式主线版本
- `sa_unetv2`：轻量化对照版本

## Overview

当前实现统一了两条模型线的训练协议，核心设置包括：

- 输入统一为 `green_clahe`
- 损失统一为 `BCE + Dice`
- 数据增强统一使用 `Albumentations`
- 优化器统一为 `Adam`
- 学习率调度统一为 `ReduceLROnPlateau`
- 提前停止统一为 `EarlyStopping`
- 训练、评估和预测导出全部严格使用 FOV mask

## Features

- 同时支持 `sa_unet` 与 `sa_unetv2`
- 统一 DRIVE 与 CHASEDB1 的训练治理参数
- 自动根据 checkpoint 中的 `dataset_name` 选择预测方式
- DRIVE 使用整图推理，CHASEDB1 使用滑窗推理
- 支持常见训练超参数覆盖，便于继续扫参与对照

当前训练增强包含：

- `HorizontalFlip`
- `VerticalFlip`
- `RandomRotate90`
- `ElasticTransform`
- `RandomGamma`
- `RandomBrightnessContrast`
- `GaussNoise`
- `GaussianBlur`

## Project Structure

```text
.
├── code/
│   ├── dataset.py
│   ├── experiment_utils.py
│   ├── model_factory.py
│   ├── model_sa_unet.py
│   ├── model_sa_unetv2.py
│   ├── predict.py
│   ├── train_chase.py
│   ├── train_drive.py
│   └── training_core.py
├── datasets/
├── output/
├── README.md
└── requirements.txt
```

## Installation

建议先创建并激活你自己的 Python 虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### DRIVE

- 使用官方 `training` / `test` 目录
- 输入统一为 `green_clahe`
- 整图会 pad / resize 到 `592x592`
- `training` 内部再切分 train / val
- `test` 作为独立评估集
- 始终使用官方 FOV mask

### CHASEDB1

- 前 20 张内部再切分 train / val，后 8 张作为测试集
- 默认按随机种子切分为 16 张训练、4 张验证
- 输入统一为 `green_clahe`
- 训练采用 `256x256` patch-based 裁剪
- 验证与测试采用 sliding window inference
- 默认滑窗参数为 `window_size=256`、`stride=128`
- 缺失 FOV mask 时会直接报错，不会自动生成

## Training

### DRIVE

```bash
python code/train_drive.py --model-name sa_unet
python code/train_drive.py --model-name sa_unetv2
```

### CHASEDB1

```bash
python code/train_chase.py --model-name sa_unet
python code/train_chase.py --model-name sa_unetv2
```

## Default Training Configuration

当前版本采用“训练策略统一、模型正则保留差异”的方案：

| 参数 | `sa_unet` | `sa_unetv2` |
| --- | --- | --- |
| `epochs` | `150` | `150` |
| `lr` | `1e-3` | `1e-3` |
| `weight_decay` | `0.0` | `1e-4` |
| `drop_prob` | `0.18` | `0.15` |
| `ReduceLROnPlateau factor` | `0.5` | `0.5` |
| `ReduceLROnPlateau patience` | `12` | `12` |
| `EarlyStopping patience` | `24` | `24` |
| `min_lr` | `1e-6` | `1e-6` |

训练行为说明：

- 学习率调度与提前停止都监控 `val_loss`
- 最佳 checkpoint 按 `val_dice` 保存
- 因为启用了 EarlyStopping，实际训练轮数可能小于 `150`

两个训练脚本都支持以下常用可覆盖参数：

- `--epochs`
- `--lr`
- `--weight-decay`
- `--drop-prob`
- `--plateau-patience`
- `--early-stop-patience`
- `--min-lr`
- `--lr-factor`

例如：

```bash
python code/train_drive.py \
  --model-name sa_unetv2 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --drop-prob 0.15 \
  --plateau-patience 8 \
  --early-stop-patience 16 \
  --min-lr 1e-6 \
  --lr-factor 0.5
```

## Inference

```bash
python code/predict.py --checkpoint path/to/best.pt
```

`predict.py` 会根据 checkpoint 中保存的 `dataset_name` 自动选择：

- DRIVE：整图推理
- CHASEDB1：滑窗推理

## Output

推荐输出目录结构：

```text
output/
├── drive/
│   ├── sa_unet/
│   └── sa_unetv2/
└── chase/
    ├── sa_unet/
    └── sa_unetv2/
```

每个训练目录下通常会生成：

- `checkpoints/best.pt`
- `metrics.json`
- `predictions/...`

`metrics.json` 会记录训练治理相关字段，例如：

- `max_epochs`
- `epochs_ran`
- `best_epoch`
- `early_stopped`
- `scheduler`
- `scheduler_factor`
- `scheduler_patience`
- `min_lr`
- `early_stop_patience`

## References

- [clguo/SA-UNet](https://github.com/clguo/SA-UNet)
- [clguo/SA-UNetv2](https://github.com/clguo/SA-UNetv2)
