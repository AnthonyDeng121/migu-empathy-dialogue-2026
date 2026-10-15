# C Day 4：画像长上下文 2×2 对照与 exact-match 校准

**结论：本轮仍 no-go，没有训练出的画像模型超过全空基线。** 对照了 max length `{256, 512}` 与正类加权 `{0.5, 0}`。512-token + 不加权候选按训练留出集做 exact 阈值校准后，验证宏 exact 为 **0.628292**，与全空基线完全持平；宏 Micro-F1 为 **0.019807**。由于没有优于基线，不能把它标为达标或发布。

## 训练对照

共有四个组合，其中 Day3/Day4 的 256-token 两格已在前序报告记录，本次补齐两个 512-token 条件。情感 encoder/head 固定，画像为三个独立线性 head；seed `20261015`，校准集为训练数据中 15% conversation-disjoint 留出，60 epochs，学习率 `0.01`，batch size 32（特征头优化批次 512）。设备为 Apple M5 / MPS，Python 3.12.14、PyTorch 2.14.0、Transformers 5.17.0；平台 GPU 卡时为 0。

| max length / 正类权重指数 | 默认 Micro-F1 阈值 macro exact | 默认阈值 macro Micro-F1 | 训练内 exact 校准后验证 macro exact / Micro-F1 |
|---|---:|---:|---:|
| 256 / 0.5（Day3） | 0.429340 | 0.2525 | 未运行 |
| 256 / 0（Day4） | 0.382470 | 0.2682 | 未运行 |
| 512 / 0.5 | 0.411923 | 0.266647 | 0.626451 / 0.034942 |
| 512 / 0 | 0.417021 | 0.278164 | **0.628292 / 0.019807** |
| 512 / 0 时全空基线 | 0.628292 | 0.000000 | 0.628292 / 0.000000 |

512-token 不加权是当前非空模型中默认阈值 macro Micro-F1 最高的候选，但默认 exact 仍只有 0.417021。按训练留出集逐维 exact 校准后，与验证全空基线持平，预测正例也很少，不能证明学到的画像比全空输出更好。

## 训练内 exact 阈值校准

新增 `scripts/calibrate_profile_exact_thresholds.py`，冻结训练出的权重，仅使用训练内 15% conversation-disjoint 留出集逐维选 exact-match 阈值；验证集只评一次。阈值为 `personality_traits=0.70, interests=0.64, style=0.80`。训练留出集模型 macro exact 0.606005，训练留出全空基线 0.601944；独立验证模型与全空基线都为 0.628292。

这说明阈值校准没有带来验证集提升，Micro-F1 也大幅下降。没有按验证结果改维度或挑阈值。官方评分实现仍未确认，`0.628292` 只是内部全空对照，不代表官方合格线。

## 数据与产物

- B1 v1：train 18,653 条、validation 2,354 条，conversation overlap 为 0；训练 SHA-256 `4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5`，验证 SHA-256 `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201`。
- 本机 checkpoint：`weights/profile_long_context_b1_20261015` 和 `weights/profile_long_context_unweighted_b1_20261015`，均在忽略目录，不进入 Git 或公开 Release。
- 机器可读训练历史、阈值校准、哈希与分维结果：[JSON 指标报告](c_day4_profile_long_context_metrics.json)。
- 复现长上下文 + 不加权训练：

  ```bash
  python scripts/train_profile_heads.py \
    --train data/processed/b1_full_v1/classification_train.jsonl \
    --val data/processed/b1_full_v1/classification_val.jsonl \
    --emotion-checkpoint weights/emotion_balanced_b1_20261015 \
    --emotion-report results/emotion_balanced_b1_20261015.json \
    --output-dir weights/profile_long_context_unweighted_b1_20261015 \
    --report results/profile_long_context_unweighted_b1_20261015.json \
    --seed 20261015 --max-length 512 --batch-size 32 --epochs 60 \
    --learning-rate 0.01 --calibration-fraction 0.15 \
    --positive-weight-exponent 0 --device mps
  ```

- 复现 exact-match 阈值校准：

  ```bash
  python scripts/calibrate_profile_exact_thresholds.py \
    --train data/processed/b1_full_v1/classification_train.jsonl \
    --val data/processed/b1_full_v1/classification_val.jsonl \
    --checkpoint weights/profile_long_context_unweighted_b1_20261015 \
    --report results/profile_long_context_unweighted_exact_calibration_20261015.json \
    --seed 20261015 --batch-size 32 --device mps
  ```

## 后续决策

本次额外训练没有让画像候选超过内部基线。不要称为“已达标”或发布 no-go 权重。由于官方指标仍未公布，进一步训练前需要官方评分口径或更多带画像标注的数据，避免继续在同一验证集上试参。当前情感模型指标保持不变；画像只能作为实验候选交 D 复核，不能声称画像任务已通过。
