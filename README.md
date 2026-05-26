# SA-UNet Retinal Vessel Segmentation

基于 `U-Net`、`SA-UNet` 和 `SA-UNetv2` 的眼底血管分割项目，当前面向两个公开数据集：

- `DRIVE`
- `CHASEDB1`

项目保留五条模型线：

- `unet`：标准 U-Net baseline
- `sa_unet`：当前正式主线版本
- `sa_unetv2`：轻量化对照版本
- `multihead_unet` / `multihead_sa_unetv2`：普通双头多标注 baseline
- `gflow_*`：带显式特征流节点、终端标注 sink 和守恒约束的 GFlow-UNet / GFlow-SAUNetV2

## Overview

当前实现统一了各模型线的训练协议，核心设置包括：

- 输入统一为 `green_clahe`
- 损失统一为 `BCE + Dice`
- 数据增强统一使用 `Albumentations`
- 优化器统一为 `Adam`
- 学习率调度统一为 `ReduceLROnPlateau`
- 提前停止统一为 `EarlyStopping`
- 训练、评估和预测导出全部严格使用 FOV mask
- 支持模拟人工标注边界偏差，用于测试多峰标注分布拟合能力
- GFlow 模型保留 source image、intermediate feature-flow nodes、terminal annotation sinks、forward flow participation 和 flow/conservation report

## Features

- 同时支持 `unet`、`sa_unet`、`sa_unetv2`、普通 multi-head 和 GFlow 系列模型
- 支持标准 `unet` baseline，便于判断提升来自模型结构还是训练/预处理
- 统一 DRIVE 与 CHASEDB1 的训练治理参数
- 支持 `green`、`green_clahe`、`rgb` 输入消融
- 支持 `BCE` 与 `BCE + Dice` 损失消融
- 支持 `second` 与 `random_primary` 标注偏差模式，验证/测试阶段可生成确定性的模拟 secondary label
- 支持通过 `--data-root` 或 `RETINA_DATA_ROOT` 在任意远程服务器上指定数据路径
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
│   ├── model_gflow_unet.py
│   ├── model_multihead_unet.py
│   ├── model_sa_unet.py
│   ├── model_sa_unetv2.py
│   ├── model_unet.py
│   ├── predict.py
│   ├── summarize_bias_core.py
│   ├── summarize_metrics.py
│   ├── train_chase.py
│   ├── train_drive.py
│   ├── training_core.py
│   └── visualize_errors.py
├── datasets/
├── output/
├── scripts/
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
- 整图会 pad / resize 到 `592x592` 进入模型
- 评估指标和预测导出会裁回原始有效区域，即官方 `584x565`
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
python code/train_drive.py --model-name unet
```

### CHASEDB1

```bash
python code/train_chase.py --model-name sa_unet
python code/train_chase.py --model-name sa_unetv2
python code/train_chase.py --model-name unet
```

常用消融参数：

```bash
python code/train_drive.py \
  --model-name unet \
  --data-root /path/to/datasets \
  --input-mode green_clahe \
  --loss-mode bce_dice \
  --output-dir /path/to/output/drive/unet_green_clahe_bce_dice
```

其中：

- `--data-root` 指向包含 `DRIVE/` 和 `CHASEDB1/` 的目录；也可使用环境变量 `RETINA_DATA_ROOT`
- `--input-mode` 可选 `green_clahe`、`green`、`rgb`
- `--loss-mode` 可选 `bce_dice`、`bce`
- `--drive-train-ratio` 和 `--chase-train-ratio` 默认均为 `0.8`，表示官方训练池内部按 16/4 划分 train/val

## Priority Experiment Matrix

远程服务器上推荐先跑统一 baseline 与关键消融：

```bash
export DATA_ROOT=/path/to/datasets
export OUT_ROOT=/path/to/output_priority
export DEVICE=cuda
export EPOCHS=300
export NUM_WORKERS=8
bash scripts/run_priority_experiments.sh
```

该脚本会运行：

- `unet`、`sa_unet`、`sa_unetv2` 在 `green_clahe + BCE/Dice` 下的同协议对比
- `unet` 的 `green`、`green_clahe`、`rgb` 输入消融
- `unet` 的 `BCE` 与 `BCE + Dice` 损失消融
- 最后生成 `summary.csv`

## GFlow Bias-Core Experiments

核心实验聚焦“模拟人工标注边界偏差 / 多峰标注分布”场景。训练和评估都使用确定性的 simulated secondary label，比较对象包括：

- learned conserved `gflow_sa_unetv2_cond`
- learned conserved random-init `gflow_sa_unetv2_cond_randinit`
- no-conservation / uniform-flow / fixed-random-flow ablations
- 普通 `multihead_unet` 与 `multihead_sa_unetv2`
- single-head `unet` random-primary augmentation

`fixed-random-flow` 指流权重由随机数固定生成、不参与学习；它用于区分“随便给一条固定信息流路径”与“学习到的守恒信息流”。

当前 focused one-seed 结果保存在：

- `output/evalsim_focused/comparison_focused.csv`
- `output/evalsim_focused/aggregate.csv`

关键 `bias_core_score` 如下：

| Dataset | Best learned conserved GFlow | Score | Best non-GFlow / fixed-flow comparator | Score | Delta |
| --- | --- | ---: | --- | ---: | ---: |
| DRIVE seed42 | `gflow_sa_unetv2_cond` | `0.714820` | `gflow_sa_unetv2_cond_random` | `0.710850` | `+0.003970` |
| CHASE seed42 | `gflow_sa_unetv2_cond_randinit` | `0.734882` | `gflow_sa_unetv2_cond_random` | `0.734299` | `+0.000582` |

同一结果中，learned conserved GFlow 也高于 no-conservation、uniform-flow、`multihead_unet` 与 `multihead_sa_unetv2`。CHASE 上优势很小，但方向为正；更完整的统计仍需要多 seed 扩展。

## Default Training Configuration

当前版本采用“训练策略统一、模型正则保留差异”的方案：

| 参数 | `unet` | `sa_unet` | `sa_unetv2` |
| --- | --- | --- | --- |
| `epochs` | `300` | `300` | `300` |
| `lr` | `1e-3` | `1e-3` | `1e-3` |
| `weight_decay` | `1e-4` | `0.0` | `1e-4` |
| `drop_prob` | `0.0` | `0.18` | `0.15` |
| `ReduceLROnPlateau factor` | `0.5` | `0.5` | `0.5` |
| `ReduceLROnPlateau patience` | `12` | `12` | `12` |
| `EarlyStopping patience` | `24` | `24` | `24` |
| `min_lr` | `1e-6` | `1e-6` | `1e-6` |

训练行为说明：

- 学习率调度与提前停止都监控 `val_loss`
- 最佳 checkpoint 按 `val_dice` 保存
- 因为启用了 EarlyStopping，实际训练轮数可能小于 `300`

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
python code/predict.py --checkpoint path/to/best.pt --data-root /path/to/datasets
```

`predict.py` 会根据 checkpoint 中保存的 `dataset_name` 自动选择：

- DRIVE：整图推理
- CHASEDB1：滑窗推理

DRIVE 预测导出会自动裁回官方原始有效区域。

## Metrics and Error Analysis

汇总多个实验：

```bash
python code/summarize_metrics.py --root /path/to/output_priority --output /path/to/output_priority/summary.csv
```

生成 TP/FP/FN 错误叠图：

```bash
python code/visualize_errors.py \
  --dataset drive \
  --data-root /path/to/datasets \
  --prediction-dir /path/to/output_priority/drive/sa_unetv2_green_clahe_bce_dice/predictions/drive_test/DRIVE \
  --output-dir /path/to/output_priority/drive_errors
```

颜色含义：

- 绿色：TP，预测正确的血管
- 红色：FP，误检血管
- 蓝色：FN，漏检血管

## Output

推荐输出目录结构：

```text
output/
├── drive/
│   ├── unet/
│   ├── sa_unet/
│   └── sa_unetv2/
└── chase/
    ├── unet/
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
- `train_names`
- `val_names`
- `test_names`
- `scheduler`
- `scheduler_factor`
- `scheduler_patience`
- `min_lr`
- `early_stop_patience`

## References

- [clguo/SA-UNet](https://github.com/clguo/SA-UNet)
- [clguo/SA-UNetv2](https://github.com/clguo/SA-UNetv2)
