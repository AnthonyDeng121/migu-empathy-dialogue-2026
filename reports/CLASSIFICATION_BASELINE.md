# 多数类分类 baseline

**状态：Integration Gate。** 情感标签和三维画像枚举复用 `src/data/schema.py`。
多数类分类器按训练集 prediction points 统计情感类别，以及每个画像维度的完整标签集合；空数组作为合法类别，平票按标签字典序稳定裁决。

## 官方 train/val 全量实跑

训练 prediction points：18657；验证 prediction points：2354。

情感多数类：`anxiety`，支持数 6340；验证集 Accuracy 31.90%。

画像按完整标签集合计算 exact-match，空数组按合法类别处理：

| 维度 | 多数类预测 | 训练支持数 | 验证 exact-match accuracy |
|---|---|---:|---:|
| personality_traits | `[]` | 10492 | 61.85% |
| interests | `[]` | 12919 | 74.89% |
| style | `[]` | 8731 | 51.74% |

画像三维平均 exact-match accuracy：62.83%。

画像指标按每个维度的完整标签集合 exact-match 计算，空数组视为合法集合；情感指标为单标签 Accuracy。该结果用于本地 baseline 对比，官方评分实现和权重仍以赛方说明为准。

使用方法：

```bash
python scripts/evaluate_majority_classifier.py \
  --train data/raw/训练-验证-数据集/train/train_public.jsonl \
  --val data/raw/训练-验证-数据集/val/val_public.jsonl \
  --output-json reports/classification_baseline.json \
  --output-md reports/CLASSIFICATION_BASELINE.md
```
