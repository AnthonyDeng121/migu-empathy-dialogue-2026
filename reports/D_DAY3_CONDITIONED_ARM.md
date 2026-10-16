# D 岗位 Day 3：C2 预测条件臂复跑（条件注入是否帮助回复）

**状态：Done（本轮推理与指标已完成并入档）。** D2 候选此前只有无条件臂；本报告补上**预测条件臂**——用 C 的 joint checkpoint 对同一冻结 B1 val 预测 emotion_label 与三维 user_profile，作为推理时条件注入，回答 `docs/TRAINING_WORKPLAN.md` 里"D，C复核：情感/画像条件注入是否帮助回复"这一问题。

**结论：用 C 当前的 checkpoint，条件注入基本中性（−0.2%），推理耗时无可测增加。不建议在部署里打开。**（耗时口径 2026-10-15 夜修正：初版"慢 2.32 倍"不成立，见文末「耗时口径修正」。）

## 运行配置与证据

| 项 | 值 |
|---|---|
| 远程实例 | `ai-traininfer-c8096171-2bd12a93-6ddb877685-v68zc`（`183.222.230.87:58346`），RTX 4090 24GB |
| 起止 | 2026-10-15 16:20 → 18:44 +08，墙钟 **8,664.81s ≈ 2.41h** |
| 模型 | `/root/base_qwen15b`（Qwen2.5-1.5B-Instruct，revision `989aa798…`，`model.safetensors` SHA-256 `dd924a11…`，与本地固定指纹逐字节一致）+ adapter `weights/response_sft_d2_1p5b`（`adapter_model.safetensors` SHA-256 `62a695ea…`，与交付件一致） |
| 条件来源 | C 的 joint checkpoint，GitHub Release `c-day3-retrained-20261015`（zip SHA-256 `550e9c4743671eb736338c8193b7cdc3f82386c63a86ac2d143dcf034c2f7b50`，下载后核验一致），由新增 `scripts/build_condition_file.py` 驱动 `EmotionProfileCheckpoint.predict_contexts` 生成 |
| 数据 | 冻结 B1 `response_val.jsonl` 2,354 行，SHA-256 `ddf76588…`；条件文件 `results/response_gen/c2_conditions_val_full.jsonl`，id 与 B1 `sample_id` 逐行同序、三维齐全 |
| 环境 | Python 3.10.12、torch 2.5.1+cu124、transformers 5.17.0、peft 0.21.2、accelerate 1.15.0；`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |
| 生成参数 | greedy、`max_new_tokens=256`、`use_cache=True`、bfloat16、`condition_mode: predicted`、`condition_coverage: 2354/2354`——除条件外与无条件臂完全一致 |
| 吞吐 | 3.6809 秒/条、23.2 tok/s、0 空输出 |

产物（回传后哈希与远程逐一致）：

| 文件 | SHA-256 |
|---|---|
| `results/response_gen/d2_1p5b_val_full_conditioned.jsonl` | `9977fbada385b28f5c64b36cc6c893519d5b68fbad3aa78908d03725fd29175c` |
| `results/response_gen/d2_1p5b_val_full_conditioned_report.json` | `f76ca68cade6ff3d91c9d6429ea34c99ac73ce1e6120c96bd4d12484087ad71f` |
| `logs/d2_1p5b_conditioned_val_full.txt` | `28c7fbc2d318b8049600bcbbf618aa9796ad18d693d91112c689d54cde9065ae` |

复现命令：

```bash
# 1) 生成条件文件（C 的 checkpoint 已解包到 weights/ 下）
python3 scripts/build_condition_file.py \
  --checkpoint weights/C_DAY3_RETRAINED_HANDOFF_20261015/emotion_profile_balanced_b1_20261015 \
  --input data/processed/b1_full_v1/response_val.jsonl \
  --output results/response_gen/c2_conditions_val_full.jsonl --device cuda

# 2) 条件臂推理
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_response_inference.py \
  --input data/processed/b1_full_v1/response_val.jsonl \
  --model /root/base_qwen15b --adapter weights/response_sft_d2_1p5b \
  --condition-jsonl results/response_gen/c2_conditions_val_full.jsonl \
  --output results/response_gen/d2_1p5b_val_full_conditioned.jsonl \
  --performance-report results/response_gen/d2_1p5b_val_full_conditioned_report.json \
  --device cuda --max-new-tokens 256 --full-set-size 2354
```

## 两臂对比（同一冻结 B1 val、同一代码路径、除条件外参数全同）

| 臂 | BLEU-4 | ROUGE-L F1 | 近空回复(≤6 字符) | 平均长度 | 秒/条 |
|---|---:|---:|---:|---:|---:|
| D2 无条件 | 0.107468 | 0.254184 | 32（1.4%） | 135.9 | 3.7572 |
| D2 预测条件 | 0.107244 | 0.253738 | 31（1.3%） | 138.6 | 3.6809 |
| 差异 | **−0.2%** | **−0.2%** | −1 | +2.7 | **−2.0%** |

（秒/条一行的无条件臂值取自 `results/response_gen/d2_1p5b_val_full_report.json`（10/14 远程 4090 实测 3.7572 秒/条）；本臂跑在 10/15 新双 4090 实例，两台 adapter 臂 tok/s 一致（22.35 vs 23.2），跨机偏差在此量级内。初版此列误填 1.5878——那是同机原模型不挂 adapter 的值，修正记录见文末。）

- 逐条：条件臂在 **905/2,354（38.4%）** 个点上 ROUGE-L 更高；其中 468 条两臂文本完全相同，**在剩下 1,886 条真有变化的里，无条件臂赢 981 条、条件臂赢 905 条**——方向是无条件略优。
- 条件确实在改变输出：**80.1% 的回复文本与无条件臂不同**，但聚合指标几乎没动。也就是说条件把文本推向了别的方向，并没有推向金标。
- 耗时：3.6809 vs 3.7572 秒/条（−2.0%，跨机）；与同机同时段的 minlen 臂（同为 D2+adapter）3.6334 秒/条相比 **+1.3%**——prompt 变长推高 prefill 的效应存在但很小（修正记录见文末）。

## 结论与边界

**不建议在部署里打开条件注入**：收益 −0.2%（噪声级）、方向略负，耗时无可测增加（同机 +1.3%）。这与 A 的集成配置 `condition_response: false` 一致，可以作为该选择的量化依据。

**必须说清的边界**：这个实验测的是"**C 当前 checkpoint 预测出来的条件**有没有用"，不是"条件注入这个机制有没有用"。C 自己在交接里写明画像头**低于全空基线**（宏平均完全匹配 0.4293 vs 空集 0.6283），且实测**预测画像全空率 18.8%，而金标全空率 37.2%**——画像头在过度预测非空。D2 训练时用的是**金标**条件，所以金标条件大概率是有用的；但部署时拿不到金标，只能拿预测，而预测质量目前不够。**"换更好的条件源是否能让注入转正"仍是开放问题**，不该由本次结果下结论。

另记：A 的集成管线（PR #17）产出的 `submission.jsonl` 与无条件臂预测 **2,354/2,354 逐字节相同**，所以本报告的无条件臂指标（0.1075 / 0.2542）就是集成交付物里回复字段的指标。

## 耗时口径修正（2026-10-15 夜，A 审查后追加）

初版报告称"条件注入让推理慢 2.32 倍"，**该数字不成立，本版修正；实验结论（不建议开启）不变**。

- **错在哪**：两臂对比表的"D2 无条件"秒/条填了 1.5878——那是**同机原模型（不挂 adapter）**的实测（`results/response_gen/original_1p5b_val_full_report.json`，10/15 15:02–16:05），不是 D2 无条件臂。D2 无条件臂的实测在 `results/response_gen/d2_1p5b_val_full_report.json`：**3.7572 秒/条**（8,844.469s / 2,354 条，10/14 远程 4090，22.35 tok/s）。
- **修正后**：条件臂 3.6809 vs 3.7572 秒/条 = **−2.0%**（跨机：条件臂在 10/15 新双 4090 实例，无条件臂在 10/14 单 4090 实例；两台 adapter 臂 tok/s 一致，22.35 vs 23.2，跨机偏差在此量级内）。同机参照取同时段 minlen 臂（同为 D2+adapter、同机）3.6334 秒/条，条件注入 **+1.3%**。
- **"2.32 倍"的真实出处**：它是 **LoRA adapter 的解码开销**，不是条件注入的代价——同机、同日、同基座、同数据、同生成参数：不挂 adapter 1.5878 秒/条（53.2 tok/s）vs 挂 adapter 3.63–3.68 秒/条（23.2–23.6 tok/s），约 2.3×。这是候选模型自身的服务成本（部署延时预算应据此估），与条件注入无关；merge 路径能否消除该开销未实测，不下结论。
