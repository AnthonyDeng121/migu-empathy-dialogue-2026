# C Day5：情感分类长尾权重优化

**状态：候选已训练并完成保存后复核，等待 B 复核指标、D 复核训练改动。** 在 B1 v1 同一验证集上，权重指数 `0.75` 候选的 Accuracy 为 **0.716653**、Macro-F1 为 **0.584843**；相较 Day3 指数 `0.5` 候选分别增加 **0.007222** 和 **0.011091**。这是一项本机 MPS 实验，尚无官方评分标准或平台镜像兼容结论。

## 假设与变更

Day3 的弱项集中在长尾类别：`care`（训练 79 条、验证 11 条）F1 为 0，`surprise`（64 / 9）F1 为 0.363636；`mixed` 验证 F1 为 0.271318。Day5 只改变交叉熵类别权重的逆频率指数：从 `0.5` 提到 `0.75`，提高稀有类权重；其余使用同一基线 checkpoint、seed、epoch、batch、学习率、最大长度和 B1 数据。没有用验证集选择阈值或继续搜索参数。

| 候选 | 权重指数 | Accuracy | Macro-F1 | 相对 Day3 |
|---|---:|---:|---:|---:|
| Day3 B1 类别加权 | 0.50 | 0.709431 | 0.573752 | — |
| Day5 长尾加权 | 0.75 | **0.716653** | **0.584843** | Accuracy +0.007222；Macro-F1 +0.011091 |

## 逐类错误复核

| 类别 | 验证支持数 | Day3 F1 | Day5 F1 | 观察 |
|---|---:|---:|---:|---|
| anger | 115 | 0.731915 | 0.731278 | 基本持平 |
| anxiety | 751 | 0.790879 | 0.800247 | 提升 |
| care | 11 | 0.000000 | 0.000000 | 未改善；正例少，需补标注或复核标签边界 |
| disgust | 21 | 0.810811 | 0.808511 | 基本持平 |
| fear | 124 | 0.650519 | 0.648855 | 基本持平 |
| gratitude | 118 | 0.791667 | 0.782609 | 小幅下降 |
| helplessness | 142 | 0.655052 | 0.661818 | 小幅提升 |
| joy | 20 | 0.473684 | 0.500000 | 小幅提升 |
| loneliness | 115 | 0.770563 | 0.786026 | 提升 |
| mixed | 177 | 0.271318 | 0.166667 | **明显下降**；召回从 0.197740 降至 0.101695，是本轮主要回归 |
| neutral | 53 | 0.385965 | 0.356436 | 下降 |
| pride | 40 | 0.392157 | 0.415094 | 提升 |
| relaxed | 409 | 0.802737 | 0.801332 | 基本持平 |
| sadness | 131 | 0.565401 | 0.571429 | 小幅提升 |
| shame | 118 | 0.723735 | 0.738956 | 提升 |
| surprise | 9 | 0.363636 | 0.588235 | 提升；样本量很小，需谨慎解释 |

整体 Macro-F1 和 Accuracy 都有提升，但这不是所有类别都变好：`care` 仍未学到，`mixed` 明显退步。推荐把 Day5 权重作为本轮**待复核候选**，保留 Day3 权重作为回退；在官方指标确定前，不将其宣称为官方达标模型。

## 训练与数据证据

- 训练日期：2026-10-16；Apple M5 / MPS；Python 3.12.14、PyTorch 2.14.0、Transformers 5.17.0；平台 GPU 卡时为 0。
- 训练集 18,653 条，验证集 2,354 条，conversation overlap 为 0。Train SHA-256：`4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5`；validation SHA-256：`30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201`。
- 同一 `weights/emotion_baseline` 起点；seed `20261015`；1 epoch / 583 optimizer steps；batch size 32；learning rate `5e-6`；max length 256；权重指数 `0.75`，归一化后裁剪至 `[0.35, 4.0]`。
- 训练循环计时 1,255.565 秒。训练命令从 09:45:02 到 10:06:43（Asia/Shanghai），包括数据准备、基线推理和保存的端到端墙钟时间约 21 分 41 秒。
- 保存 checkpoint 后重新加载，对完整验证集重新推理；Accuracy、Macro-F1 和逐类指标与训练报告完全一致。
- 机器可读指标、完整混淆矩阵与类别权重见 [JSON 报告](c_day5_emotion_longtail075_metrics.json)；完整训练 stdout 随 checkpoint 一并放入公开 Release 供复核。
- 本地 checkpoint 目录：`weights/emotion_longtail075_b1_20261016`；`model.safetensors` SHA-256：`aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37`；目录树 SHA-256：`258c43a7120741d9281e42edf0ddd8c3c543228a690a7e6c13f2549b6f6d4e79`。
- 交付包：`C_DAY5_EMOTION_LONGTAIL075_20261016.zip`，约 391 MiB，ZIP 完整性检查通过；SHA-256：`67db558687a42c6b10f31075ff1fecf14ffc6ba881c4d14d0bcd387b0ea0c526`。公开预发布：[Day5 checkpoint 与训练日志](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-day5-emotion-longtail075-20261016)。

## 复现

```bash
/Users/7bacslash/miniforge3/envs/comfyui-yue2/bin/python \
  scripts/finetune_emotion_balanced.py \
  --train data/processed/b1_full_v1/classification_train.jsonl \
  --val data/processed/b1_full_v1/classification_val.jsonl \
  --base-checkpoint weights/emotion_baseline \
  --output-dir weights/emotion_longtail075_b1_20261016 \
  --report results/emotion_longtail075_b1_20261016.json \
  --seed 20261015 --batch-size 32 --learning-rate 5e-6 \
  --class-weight-exponent 0.75 --device mps
```

权重包只包含模型与 tokenizer 文件，不含训练数据。官方评分口径、官方镜像加载与真实提交结果仍待确认。
