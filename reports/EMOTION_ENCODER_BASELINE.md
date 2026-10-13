# 中文 encoder 情感分类 baseline

**状态：Integration Gate。**

模型：[`hfl/chinese-macbert-base`](https://huggingface.co/hfl/chinese-macbert-base)，
revision `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`。
设备：`mps`；seed：`20261012`；
epoch：1；max length：256。
训练集：18657/18657 个 prediction points；
验证集：2354/2354 个 prediction points。

按验证集 Accuracy 选择 epoch 1 的 checkpoint。
验证 Accuracy：**67.84%**；Macro-F1：**0.4032**。
多数类 baseline Accuracy：31.90%；提升 35.94 个百分点。

| 情感类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.6864 | 0.7043 | 0.6953 | 115 |
| anxiety | 0.6630 | 0.8935 | 0.7612 | 751 |
| care | 0.0000 | 0.0000 | 0.0000 | 11 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 21 |
| fear | 0.6111 | 0.6210 | 0.6160 | 124 |
| gratitude | 0.8036 | 0.7627 | 0.7826 | 118 |
| helplessness | 0.6250 | 0.6690 | 0.6463 | 142 |
| joy | 0.0000 | 0.0000 | 0.0000 | 20 |
| loneliness | 0.7547 | 0.6957 | 0.7240 | 115 |
| mixed | 0.4211 | 0.0452 | 0.0816 | 177 |
| neutral | 0.6000 | 0.1132 | 0.1905 | 53 |
| pride | 0.0000 | 0.0000 | 0.0000 | 40 |
| relaxed | 0.6874 | 0.8924 | 0.7766 | 409 |
| sadness | 0.7121 | 0.3588 | 0.4772 | 131 |
| shame | 0.7549 | 0.6525 | 0.7000 | 118 |
| surprise | 0.0000 | 0.0000 | 0.0000 | 9 |

数据 SHA-256：

- train：`184a47471f8d6f70fa21b9265b0a4d4e51c3748ba25af39a801884a2da35341a`
- val：`097c0639ef7370a18a54670ec3c8390ce5dc87c1ed95f436cef07b68dfed7668`

本机 checkpoint：`weights/emotion_baseline/model.safetensors` （409143304 bytes，SHA-256 `50183b12b36ab33848c81ecbe068310ec233ceeccdedd4ad2a2c702d7eecc5e8`）。
混淆矩阵、逐 epoch 指标和完整配置见同名 JSON。模型权重保存在本机忽略目录，不提交到 Git。

复现命令：

```bash
python -m pip install -r requirements-emotion.txt
python scripts/train_emotion_encoder.py \
  --train data/raw/训练-验证-数据集/train/train_public.jsonl \
  --val data/raw/训练-验证-数据集/val/val_public.jsonl
```
