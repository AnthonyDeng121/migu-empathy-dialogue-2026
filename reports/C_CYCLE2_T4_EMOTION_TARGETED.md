# C T4 情感长尾针对性实验

日期：2026-10-28（Asia/Shanghai）。状态：单次候选训练与保存后复算完成；等待同行复核，不作为默认模型或正式提交权重。

## 结论

只将类别权重指数从 Day3 的 `0.50` 与 Day5 的 `0.75` 之间调到 `0.65`，其余训练条件固定。候选在 B1 固定验证集上的 Macro-F1 为 **0.590474**，高于 Day3 的 `0.573752` 和 Day5 的 `0.584843`。`mixed` F1 为 **0.212766**：优于 Day5 的 `0.166667`，仍低于 Day3 的 `0.271318`；`care` 仍为 **0/11**。因此保留为内部候选供后续复核，不宣布长尾问题解决，也不替换交付默认权重。

Day3/Day5 没有可复用的情感校准概率文件；本地 `results/calibration_probabilities.jsonl` 是画像概率，不是情感 logits。两版情感模型都在完整训练集上训练，无法用同一训练集概率构造不泄漏的校准对照。本轮没有做验证集阈值搜索、融合选择或伪造校准收益。

## 预先固定的实验

Day5 总体 Macro-F1 上升，但 `mixed` 从 Day3 的 `0.271318` 降到 `0.166667`，`care` 在两版都为零（验证支持数 11）。按 T4 只允许一次有依据的情感重训，预先选定 `0.65` 作为两个已测权重指数的中点，以同一基线和训练设置检查整体与长尾折中。没有根据本轮验证结果继续搜索指数。

| 项目 | 固定设置 |
|---|---|
| 起始权重 | `weights/emotion_baseline` |
| 训练/验证数据 | B1 v1：18,653 / 2,354 条，会话交集 0 |
| Train SHA-256 | `4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5` |
| Validation SHA-256 | `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201` |
| seed / epochs / steps | `20261015` / 1 / 583 |
| batch / learning rate / max length | 32 / `5e-6` / 256 |
| 唯一变量 | inverse-frequency class-weight exponent `0.65`（Day3 `0.50`、Day5 `0.75`） |
| 设备 | Apple M5 / PyTorch MPS；公共 GPU 卡时 0 |
| 训练入口耗时 | 1,246.111 秒（脚本计时） |

训练命令：

```bash
/Users/7bacslash/miniforge3/envs/comfyui-yue2/bin/python \
  scripts/finetune_emotion_balanced.py \
  --train data/processed/b1_full_v1/classification_train.jsonl \
  --val data/processed/b1_full_v1/classification_val.jsonl \
  --base-checkpoint weights/emotion_baseline \
  --output-dir weights/emotion_longtail065_b1_20261028 \
  --report results/emotion_longtail065_b1_20261028.json \
  --seed 20261015 --batch-size 32 --learning-rate 5e-6 \
  --class-weight-exponent 0.65 --device mps
```

## 固定验证结果

| 版本 | 权重指数 | Accuracy | Macro-F1 | mixed F1（n=177） | care F1（n=11） | surprise F1（n=9） |
|---|---:|---:|---:|---:|---:|---:|
| Day3 | 0.50 | 0.709431 | 0.573752 | 0.271318 | 0.000000 | 0.363636 |
| Day5 | 0.75 | 0.716653 | 0.584843 | 0.166667 | 0.000000 | 0.588235 |
| T4 候选 | 0.65 | 0.714953 | **0.590474** | 0.212766 | 0.000000 | 0.625000 |

本轮相对 Day5 的 Accuracy 低 `0.001700`，Macro-F1 高 `0.005631`；`mixed` 有回升但没有回到 Day3 水平。`care` 样本数很少且本候选仍未预测正确。综合结果只能支持“内部折中候选”，不能支持“达标”或隐藏测试提升。

## 保存与复算

- checkpoint：`weights/emotion_longtail065_b1_20261028/`；`model.safetensors` SHA-256：`bc0c7cab2259d2aea5ca77e812caff1363575da48ee94b92c03e84c39709e895`。
- 训练指标及混淆矩阵：[`c_cycle2_t4_emotion_weight065_20261028.json`](c_cycle2_t4_emotion_weight065_20261028.json)。
- 保存后从 checkpoint 与随包 tokenizer 重载，对完整 2,354 条验证集复算；Accuracy、Macro-F1 和逐类结果与机器报告完全一致。checkpoint tokenizer 的 `truncation_side` 为 `left`，复算沿用保存值。
- 完整训练 stdout 与验收材料留在本机 `.codex_tmp/acceptance/cycle2/c_t4_20261028/`，不提交训练日志或权重。

## 验收边界

T4 C 的一次针对性候选已完成，结论为内部保留、暂不采用。画像保持固定；没有情感概率校准/融合结果，原因是没有不泄漏的既有情感概率。B1 验证集被历次实验使用过，不是盲测；官方评分或平台表现均未验证。按周期计划，仍需同行审查后进入后续候选选择。
