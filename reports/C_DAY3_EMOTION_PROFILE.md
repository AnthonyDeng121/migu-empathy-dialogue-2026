# Day 3 C：情感与用户画像分类 dry-run

**状态：Exploratory sample only。** 本次是本机准备/冒烟证据，不代表平台 GPU 全量训练。

- 设备：`mps`；模型：`hfl/chinese-macbert-base`@`a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`
- 训练样本：96/18653；
验证样本：32/2354
- B1 全行校验：18653 train + 2354 val；
conversation overlap：0
- 训练步数：3；batch size：16；
最大长度：256；实测耗时：8.79s
- 环境：Python 3.12.14 / PyTorch 2.14.0 / Transformers 5.17.0
- 情感验证 Accuracy：0.3125；Macro-F1：0.0298
- 画像三维平均 exact match：0.0000；
空集合 baseline：0.6667
- 训练数据 SHA-256：`4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5`
- 验证数据 SHA-256：`30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201`
- checkpoint 目录哈希：`9ad18bd5720a69c65837d63f16fe0b8415df4672aa72da81edd96cbbbfefef2f`；
head 哈希：`c3e71877be4ff216ec9573a9f443c1cb39ca49aa141448ad51e90677d34c5c45`

| 画像维度 | Exact match | Micro-F1 | 空集合 baseline |
|---|---:|---:|---:|
| personality_traits | 0.0000 | 0.1119 | 0.5938 |
| interests | 0.0000 | 0.0164 | 0.8438 |
| style | 0.0000 | 0.1124 | 0.5625 |

正式输入接口冒烟：官方 test 输入 3 行；输出 3 行，
ID 顺序和 `id, emotion_label, user_profile` 字段检查通过（mps）。

本机 checkpoint 保存并重载后 logits 一致性检查通过。正式训练仍需在大赛 GPU 环境复跑；
官方 vLLM 镜像的依赖兼容性需在该镜像内验证后才能确认。

## 复现命令

```bash
python scripts/train_emotion_profile.py --train 'data/processed/b1_full_v1/classification_train.jsonl' --val 'data/processed/b1_full_v1/classification_val.jsonl' --config 'configs/emotion_profile.json' --output-dir 'weights/emotion_profile_repro' --device mps --local-files-only --max-train-samples 96 --max-validation-samples 32 --max-steps 3
```
