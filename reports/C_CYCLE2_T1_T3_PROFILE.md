# C Cycle2 T1–T3：画像专用编码器与校准

日期：2026-10-26（Asia/Shanghai）。状态：T1/T2 完成；T3 完成一次校准目标对照。v1 通过本地冻结门槛，等待 D 代码审查与 B 独立指标复核；尚未作为正式提交模型采用。

## 结论

- **v1：内部候选通过。**全量 B1 验证集 macro exact 为 **0.6468422543**，比冻结空画像基线 **0.6282922685** 高 **0.0185499858**。结果由 B 的 `scripts/b_profile_eval.py` 对 2,354 个相同 ID 独立复算，900 个会话配对为 151 胜、110 负、639 平，平均配对差 **+0.015357**。
- **改善分布不均。**`interests` exact 从 0.748938 升到 0.819456；`personality_traits` 从 0.618522 降到 0.607052；`style` 从 0.517417 降到 0.514019。不能说三维都提升。
- **T3 v2 no-go。**只把校准目标从逐维 exact 改为逐维 micro-F1，复用 v1 的概率、不重训。v2 macro exact 为 **0.5935995469**，低于空画像基线；虽然 micro-F1 提高，但主指标退步，因此不采用 v2，保留 v1。
- B1 的 2,354 条验证集已被历史实验多次使用，不是盲测；本地结果不是官方分数，也不保证隐藏集提升。v1 仍需 B 独立复核及 D 代码审查。

## T1：训练实现与小样例检查

新增 `scripts/train_profile_encoder.py`：以本机缓存的通用 MacBERT 初始化，编码器可训练，损失只训练三个画像头，不使用情感损失。它严格消费 B T1 冻结的 `reports/b_t1_train_internal_split.jsonl`，逐 ID 验证 fit/calibration 清单完整、会话归属一致，并拒绝跨子集会话。

通用预训练模型的本机快照 SHA 与训练数据指纹写入机器报告。profile 专用编码器未从已经在完整 B1 训练集上微调过的情感 checkpoint 起步，避免校准会话通过该编码器进入训练。为兼容 A 的 packaged loader，checkpoint 内保留 C Day5 情感分类头作格式字段；独立画像模式不采用该情感输出。

CPU 小样例冒烟运行 2 个 optimizer step，完成概率输出、保存及真实 `EmotionProfileCheckpoint` 重载检查；检查的 4 条预测画像与训练入口输出相同。当前清单解析专项为 3/3 通过。小样例只证明接口链路，不作为模型效果结果。

## T2：画像 v1 全量训练

| 项目 | 实测 |
|---|---|
| 设备 / 环境 | Apple M5 MPS；Python 3.12.14；PyTorch 2.14.0；Transformers 5.17.0 |
| 训练 / 校准 | 15,944 / 2,709 条，完全按 B T1 冻结会话清单；会话交集 0 |
| 固定验证 | 2,354 条、900 个会话；train/validation 会话交集 0 |
| 配置 | seed 20261026；1 epoch；batch 16；max length 256；AdamW `2e-5`；画像 BCE 正类权重指数 0.5 |
| 训练步数 / 用时 | 997 步 / 1,531.851 秒；首批 loss 0.703716，末批 loss 0.281558；loss 均有限 |
| 训练内校准阈值 | `personality_traits=0.64`、`interests=0.81`、`style=0.56`；逐维 exact 最大化，验证标签未参与阈值选择 |
| v1 验证 macro exact | **0.6468422543**，空画像 **0.6282922685** |

逐维验证 exact / micro-F1：

| 画像维度 | 空画像 exact | v1 exact | v1 micro-F1 |
|---|---:|---:|---:|
| personality_traits | 0.618522 | 0.607052 | 0.076984 |
| interests | 0.748938 | 0.819456 | 0.491304 |
| style | 0.517417 | 0.514019 | 0.151199 |

模型保存在被 `.gitignore` 忽略的 `weights/profile_encoder_v1_b1_20261026/`。目录树 SHA-256：`e7a996a95c723301f85c4e11326d02450de6e221042234abc43913b60e783e33`。checkpoint 真实重载检查通过。逐条验证预测（含 2354 个 ID、三维标签概率及阈值解码画像）保存在 `reports/c_cycle2_profile_v1_predictions.jsonl`，供 B 复核；B 工具摘要在 `reports/c_cycle2_profile_v1_b_eval_summary.json`；逐会话完整 JSON 留在本机 `.codex_tmp/acceptance/cycle2/c_v1_b_eval_full.json`，SHA-256 `0ab2e74e287b50659bfa16a7f04076d3ab1f0ca82ef33339b9b6d1d0c261890c`。

给组长交接的 ZIP 已在本机打包并通过 `unzip -t`：`.codex_tmp/acceptance/cycle2/C_CYCLE2_PROFILE_V1_20261026.zip`，379,744,878 字节，SHA-256 `2e1a187aa97446a2df22a641e9e39f56fe24d019885e658728c6af64fac96aea`。它目前尚未发布到公开 GitHub Release。

## T3：单变量阈值目标对照

权重和概率固定，只将阈值目标从逐维 exact 改为逐维 micro-F1。阈值只用同一个训练内 calibration 分区确定：`personality_traits=0.45`、`interests=0.61`、`style=0.38`。验证集 macro exact 为 **0.5935995469**，比空画像低 **0.0346927216**；会话配对为 195 胜、292 负、413 平，平均配对差 **−0.042710**。v2 的逐维 micro-F1 上升，但 exact 主指标未过线，结论为 **no-go**。完整比较仅保存在本机 `.codex_tmp/acceptance/cycle2/c_v2_b_eval.json`；阈值、来源概率与复算哈希记录在 `reports/c_cycle2_profile_v2_micro_calibration_20261026.json`。

## 可复现与交接

完整训练入口（本机需已缓存 `hfl/chinese-macbert-base@a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`）：

```bash
python scripts/train_profile_encoder.py \
  --base-model "$HOME/.cache/huggingface/hub/models--hfl--chinese-macbert-base/snapshots/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1" \
  --output-dir weights/profile_encoder_v1_b1_20261026 \
  --report reports/c_cycle2_profile_v1_20261026.json \
  --predictions results/profile_encoder_v1_b1_20261026_val.jsonl \
  --max-length 256 --batch-size 16 --epochs 1 \
  --learning-rate 2e-5 --positive-weight-exponent 0.5 --device mps
```

复算固定验证结果：

```bash
python scripts/b_profile_eval.py compare \
  --reference data/processed/b1_full_v1/classification_val.jsonl \
  --baseline .codex_tmp/acceptance/cycle2/c_v1_empty_baseline.jsonl \
  --candidate results/profile_encoder_v1_b1_20261026_val.jsonl \
  --manifest reports/b_t1_screening_manifest.jsonl \
  --output reports/c_cycle2_profile_v1_b_eval.json
```

权重不进入普通 Git 历史。模型不是官方最终模型：需先完成 D 的代码审查、B 的独立指标复核，再决定是否用于后续集成/公开分发。
