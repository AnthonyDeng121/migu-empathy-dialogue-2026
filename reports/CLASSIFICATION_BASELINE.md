# 多数类分类 baseline

**状态：Integration Gate。** 情感标签和三维画像枚举复用 `src/data/schema.py`。多数类分类器从训练集 prediction points 统计情感类别和每个画像维度的完整标签集合；空数组作为合法类别，平票按标签字典序稳定裁决。

## 基于已提交审计计数的结果

下面的计数取自 `reports/data_audit.json`，验证准确率按验证集审计计数计算。当前工作区没有官方原始 JSONL，因此尚未实际运行新评测脚本。

| 任务 | 多数类预测 | 训练支持数 / 18,657 | 验证集正确数 / 2,354 | 验证准确率 |
|---|---|---:|---:|---:|
| 情感 | `anxiety` | 6,340 | 751 | 31.90% |
| personality_traits | `[]` | 10,492 | 1,456 | 61.85% |
| interests | `[]` | 12,919 | 1,763 | 74.89% |
| style | `[]` | 8,731 | 1,218 | 51.74% |
| 画像三维平均 | — | — | — | 62.83% |

审计报告只统计各画像标签边际频次，但这足以证明空集合是每个维度的完整集合多数类：空集合频次分别高于该维度出现次数最多的单个标签；任一非空集合都至少包含一个标签，其频次不可能超过其中任一标签的边际频次。

## 复算

成员将官方数据放入仓库约定路径后，从仓库根目录运行：

```bash
python scripts/evaluate_majority_classifier.py \
  --train "data/raw/训练-验证-数据集/train/train_public.jsonl" \
  --val "data/raw/训练-验证-数据集/val/val_public.jsonl" \
  --output-json reports/classification_baseline.json \
  --output-md reports/CLASSIFICATION_BASELINE.md
```

此报告是本地分类 baseline，不代表官方指标已经确认。官方画像评分实现仍需以赛方说明为准。
