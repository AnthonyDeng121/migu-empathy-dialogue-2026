# 中文 encoder 情感分类 baseline

**状态：Exploratory sample only。**

模型：[`hfl/chinese-macbert-base`](https://huggingface.co/hfl/chinese-macbert-base)，
revision `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`。
设备：`mps`；seed：`20261012`；
epoch：1；max length：256。
训练集：1600/18657 个 prediction points；
验证集：2354/2354 个 prediction points。

按验证集 Accuracy 选择 epoch 1 的 checkpoint。
验证 Accuracy：**38.87%**；Macro-F1：**0.0648**。
测速：50 optimizer steps，3.2054 秒/step；按完整训练集估算 1871.9 秒/epoch。
多数类 baseline Accuracy：31.90%；提升 6.97 个百分点。

训练冒烟检查：50 steps；loss/梯度有限、分类头权重更新、checkpoint 本地重载均通过。

| 情感类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.0000 | 0.0000 | 0.0000 | 115 |
| anxiety | 0.4167 | 0.7324 | 0.5311 | 751 |
| care | 0.0000 | 0.0000 | 0.0000 | 11 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 21 |
| fear | 0.0000 | 0.0000 | 0.0000 | 124 |
| gratitude | 0.0000 | 0.0000 | 0.0000 | 118 |
| helplessness | 0.0000 | 0.0000 | 0.0000 | 142 |
| joy | 0.0000 | 0.0000 | 0.0000 | 20 |
| loneliness | 0.0000 | 0.0000 | 0.0000 | 115 |
| mixed | 0.0000 | 0.0000 | 0.0000 | 177 |
| neutral | 0.0000 | 0.0000 | 0.0000 | 53 |
| pride | 0.0000 | 0.0000 | 0.0000 | 40 |
| relaxed | 0.3530 | 0.8924 | 0.5059 | 409 |
| sadness | 0.0000 | 0.0000 | 0.0000 | 131 |
| shame | 0.0000 | 0.0000 | 0.0000 | 118 |
| surprise | 0.0000 | 0.0000 | 0.0000 | 9 |

数据 SHA-256：

- train：`184a47471f8d6f70fa21b9265b0a4d4e51c3748ba25af39a801884a2da35341a`
- val：`097c0639ef7370a18a54670ec3c8390ce5dc87c1ed95f436cef07b68dfed7668`

本机 checkpoint：`weights/emotion_baseline_c_day2_smoke_20261013/model.safetensors` （409143304 bytes，SHA-256 `d0d604247c05657942dec865a3a5eadacebc4b1a2ea8d5271dfb66ff9ddadfc3`）。
混淆矩阵、逐 epoch 指标和完整配置见同名 JSON。模型权重保存在本机忽略目录，不提交到 Git。

复现命令：

```bash
python -m pip install -r requirements-emotion.txt
python scripts/train_emotion_encoder.py \
  --train 'data/raw/训练-验证-数据集/train/train_public.jsonl' \
  --val 'data/raw/训练-验证-数据集/val/val_public.jsonl' \
  --device mps \
  --max-train-samples 1600 \
  --output-dir weights/emotion_encoder_repro \
  --report-json reports/emotion_encoder_repro.json \
  --report-md reports/EMOTION_ENCODER_REPRO.md
```
