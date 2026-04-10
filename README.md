# Retinal Vessel Segmentation

这个项目当前只保留 2 个模型：

- `sa_unet`：当前正式主线版本
- `sa_unetv2`：轻量化对照版本

两条模型线已经统一为：

- 输入：`green_clahe`
- 损失：`BCE + Dice`
- 训练增强：`Albumentations`
- 优化器：`Adam`
- 学习率调度：`ReduceLROnPlateau`
- 提前停止：`EarlyStopping`

其中训练增强同时用于 DRIVE 与 CHASEDB1，包含：

- `HorizontalFlip`
- `VerticalFlip`
- `RandomRotate90`
- `ElasticTransform`
- `RandomGamma`
- `RandomBrightnessContrast`
- `GaussNoise`
- `GaussianBlur`

所有训练、评估和预测导出都严格使用 FOV mask。

## 环境

当前目录自带 `.venv`，建议优先使用：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据规则

### DRIVE

- 使用官方 `training` / `test` 目录
- 输入统一为 `green_clahe`
- 整图 pad / resize 到 `592x592`
- `training` 内部按比例切分 train / val
- `test` 独立评估
- 始终使用官方 FOV mask

### CHASEDB1

- 前 20 张内部再切分 train / val，后 8 张测试
- 默认按随机种子切分为 16 张训练、4 张验证
- 输入统一为 `green_clahe`
- 训练采用 `256x256` patch-based 裁剪
- 验证与测试采用 sliding window inference
- 默认滑窗参数：
  - `window_size=256`
  - `stride=128`
- 缺失 FOV mask 时直接报错，不会自动生成

## 训练命令

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

## 默认训练策略

当前版本采用“训练策略统一、模型正则保留差异”的方案：

| 参数 | `sa_unet` | `sa_unetv2` | 是否跨 DRIVE / CHASE 统一 |
| --- | --- | --- | --- |
| `epochs` | `150` | `150` | 是 |
| `lr` | `1e-3` | `1e-3` | 是 |
| `weight_decay` | `0.0` | `1e-4` | 是 |
| `drop_prob` | `0.18` | `0.15` | 是 |
| `ReduceLROnPlateau factor` | `0.5` | `0.5` | 是 |
| `ReduceLROnPlateau patience` | `12` | `12` | 是 |
| `EarlyStopping patience` | `24` | `24` | 是 |
| `min_lr` | `1e-6` | `1e-6` | 是 |

训练行为说明：

- 学习率调度与提前停止都监控 `val_loss`
- 最佳 checkpoint 仍然按 `val_dice` 保存
- 因为启用了 EarlyStopping，实际训练轮数可能小于 `150`
- DRIVE 与 CHASE 共享同一套训练治理参数，不再分别维护不同的 `lr / patience / min_lr`

## 常用可覆盖参数

如果需要手动扫参，可以直接在训练命令后覆盖：

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

两个训练脚本都支持以下可覆盖参数：

- `--epochs`
- `--lr`
- `--weight-decay`
- `--drop-prob`
- `--plateau-patience`
- `--early-stop-patience`
- `--min-lr`
- `--lr-factor`

## 预测命令

```bash
python code/predict.py --checkpoint path/to/best.pt
```

`predict.py` 会根据 checkpoint 中保存的 `dataset_name` 自动选择：

- DRIVE：整图推理
- CHASEDB1：滑窗推理

## 输出说明

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

每个训练目录下会生成：

- `checkpoints/best.pt`
- `metrics.json`
- `predictions/...`

`metrics.json` 当前会额外记录训练治理信息，例如：

- `max_epochs`
- `epochs_ran`
- `best_epoch`
- `early_stopped`
- `scheduler`
- `scheduler_factor`
- `scheduler_patience`
- `min_lr`
- `early_stop_patience`

由于实验文档和实验结果默认不纳入 GitHub 仓库，本仓库主要保留代码、依赖和使用说明。

## 参考仓库

- [clguo/SA-UNet](https://github.com/clguo/SA-UNet)
- [clguo/SA-UNetv2](https://github.com/clguo/SA-UNetv2)
