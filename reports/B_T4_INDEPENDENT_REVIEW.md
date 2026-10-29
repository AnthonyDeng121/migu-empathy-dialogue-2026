# B-T4 独立复核

## A. 范围与快照

本轮 B-T4 只复核两个 T3 交付：C T3 profile v2 与 D T3 unconditional SFT；不把 C T4 emotion long-tail 或 D T4 long-sample weighting 作为输入。另对 A T4 PR #37 做独立工程交付审查，不把 PR 审查结论混入 C/D 模型效果结论。

当前 Git 评测基线：branch=`feature/b-eval-tooling`，`evaluation_base_commit=9b104b114309ce94e3b4d403afb1705646322933`，`origin/main=9b104b114309ce94e3b4d403afb1705646322933`。工作区除本报告和 JSON 报告外无其他已跟踪修改。

## B. C T3 profile v2

状态：**BLOCKED**。

阻塞码：`C_T3_PROFILE_V2_BLOCKED_MISSING_EVIDENCE`。

现有 report `reports/c_cycle2_profile_v2_micro_calibration_20261026.json` 记录了 v2 的阈值和聚合结果：阈值为 `personality_traits=0.45`、`interests=0.61`、`style=0.38`；validation 2354 行、validation SHA-256 为 `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201`；报告声称 v2 macro exact `0.5935995469`，empty-profile baseline `0.6282922685`，paired conversations 为 195/292/413，mean delta `-0.0427103175`。

但当前仓库不存在报告引用的逐条文件 `results/profile_encoder_v2_micro_thresholds_20261026_val.jsonl`，也没有可供 B 独立 join 的 v2 prediction。因而不能独立验证 sample ID、schema、duplicate/missing/extra、profile exact/micro-F1、paired comparison 或概率/阈值来源；不能从聚合数字重建“独立结果”。

现有 v1 逐条 prediction `reports/c_cycle2_profile_v1_predictions.jsonl`（2354 行、2354 unique ID）不能替代 v2 证据。

## C. D T3 unconditional SFT

### 身份与可比性

candidate：`results/response_gen/t3_unconditional.jsonl`；baseline：`results/response_gen/t3_d4v2_off_samedevice.jsonl`；固定输入/gold：`results/response_gen/t2_selection_input.jsonl`。三者均为 160 行、160 unique ID；candidate 与 baseline 无重复、missing、extra，ID 集合和顺序均与 selection input 一致。历史全量 `results/response_gen/d4_2ep_val_full.jsonl` 为 2354 行背景文件，不替代 160 条 paired baseline。

candidate SHA-256：`64f330b327457ac20d1162f68503f8161e4408a102fa6b710ceae1754f8c5d7e`。

baseline SHA-256：`1cf556d2f5a5fb873a49626f27dc9ee4d19bfbfca6a7a58fec5fef499b06ed67`。

selection input SHA-256：`8d1d487e4e2fb7d66071d72c8d9078744831751bfed7c9fb3103c067d9163959`。

训练报告记录 candidate 使用 Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`、B1 response train 18653/18653、2 epochs、batch 1、gradient accumulation 8、LR `1e-4`、seed `20261013`、max length 2048、LoRA r=8/alpha=16/dropout=0.05、condition_mode=`none`、4664 optimizer steps。adapter SHA-256 为 `0fa04f2d9eeaa70397c4843831669d6142041869aa31b5478e802c137a827ec5`。

与 D4 v2 的训练侧设计是单变量 condition_mode 对照；但报告未提供足以证明所有运行环境细节完全相同的独立 adapter/tokenizer/GPU 指纹，因此可比性定为 `COMPARABLE_WITH_CAVEATS`。

### 独立重算指标

使用仓库既有 `scripts/compare_response_arms.py`，按同一 sample ID、同一 selection gold、字符级 BLEU-4/ROUGE-L 计算：

| 指标 | candidate T3 | baseline D4v2 | delta |
|---|---:|---:|---:|
| samples | 160 | 160 | 0 |
| char BLEU-4 mean | 0.1033757503 | 0.1051868971 | -0.0018111469 |
| char ROUGE-L F1 mean | 0.2501903809 | 0.2525907930 | -0.0024004122 |
| mean response length | 129.05 | 125.56875 | +3.48125 |
| empty responses | 0 | 0 | 0 |

Paired ROUGE metric direction：candidate win 65、tie 30、loss 65；ROUGE delta median 0.0，p25 `-0.0149121773`，p75 `0.0196189092`。完全相同回复 26；按 `longer/same/shorter` 字段顺序，candidate 更长/等长/更短为 67/36/57。结论只使用 metric win/tie/loss，不解释为人工质量胜负。

### Slice

历史切片按项目已冻结的 history bucket 复用：

| history bucket | support | BLEU delta | ROUGE delta | candidate longer/same/shorter |
|---|---:|---:|---:|---:|
| short 1–5 | 65 | -0.004406 | +0.001364 | 29/10/26 |
| medium 6–15 | 69 | -0.002093 | +0.000476 | 28/19/22 |
| long 16+ | 26 | -0.001116 | -0.008451 | 6/8/12 |

按 gold emotion 的主要切片复核也已保留在既有 paired artifact；小支持数类别不作稳定结论。总体没有平均 BLEU/ROUGE 改善，长 history 两项指标均下降。

### D 结论

状态：`NO_CLEAR_SIGNAL`。

160 条固定筛选集只支持候选筛选和观察性比较，不支持推广到 2354 条完整 validation 的结论，也不支持宣称无条件训练优于 D4v2。

## D. B-T4 总体状态

`READY_SFT_ONLY`。

C 因缺 v2 逐条 prediction 仍 blocked；D 的 T3 unconditional prediction 已可独立复算，但结论为 `NO_CLEAR_SIGNAL`。因此 B-T4 已完成 D 侧独立复核，未完成 C v2 的独立指标复核。

## E. A PR #37 工程审查

PR #37 HEAD：`419febc214a4d324220423925679a907c16b3e43`。

相对 `origin/main` 的 8 个变更文件：离线组装脚本、8 项专项测试、两个 160 行组装候选及 manifest、说明文档和 `.gitignore` 更新。组装器按 sample ID 对齐 response/emotion/profile，拒绝输入/来源重复 ID、缺失 ID 和额外 response，记录 source path/SHA，保留 response 原文，不读取 gold emotion/profile 作为候选字段，并明确 `model_inference_performed=false`。

PR37 专项测试在该 HEAD 上为 8/8 PASS。工程审查结论：**APPROVE**。该结论只表示离线字段组装交付的工程证据充分，不表示 C/D 模型效果通过。

可直接粘贴的 GitHub review comment：

> APPROVE（工程交付审查）。PR37 的组装器按 sample ID 对齐 response/emotion/profile，拒绝 duplicate/missing/extra 和会话错配，记录输入与来源 SHA，保持 response 原文，并显式标记未执行模型推理。两个 160 条产物的 manifest 通过 ID/schema/空回复校验，专项测试 8/8 PASS。该 approve 仅覆盖离线字段组装工程，不代表 C/D 模型效果或正式端到端条件推理已通过。

## F. 验证与限制

- D 独立 paired 重算：PASS，结果与既有 `t3_arms_comparison_same_device.json` 一致。
- C v2：BLOCKED，缺逐条 prediction；未从 report 聚合数字伪造独立结果。
- JSON parse：PASS。
- Python compileall：PASS。
- CI smoke：PASS。
- `git diff --check`：PASS。
- Repository unittest：369 total；353 PASS、7 FAIL、1 ERROR、8 SKIP。FAIL/ERROR 集中于 `tests/test_remote_scripts.py` 的 Windows Bash、Unix path、remote shell、subprocess / GBK decoding 行为。
- 未运行 GPU、未训练、未修改 A/C/D 实现。
- 本报告不保存 validation 对话正文、gold response、candidate response 或 baseline response。
