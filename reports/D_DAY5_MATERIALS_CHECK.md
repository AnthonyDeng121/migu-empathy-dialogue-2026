# D Day5：最终候选材料完整性检查（2026-10-17）

分支 `feature/d-day5-materials-check`（自 `origin/main` `d741d11` 切出，已合入 PR #20 合并后的 main `6778aa5`）。对应执行计划 10/17 D 行「不新增长训练；必要解码/长度检查；**材料 1h**」与官方要求「技术方案、实验设计/报告、训练日志、调优过程……**10/17 检查完整性**，最终镜像包含材料」。

**结论：D 岗位材料完整、三处数字一致、交叉引用全部可达，检查通过。** 候选 v2（2-epoch）已由 A 于 10/17 15:17 合并 PR #20（merge `6778aa5`）成为 main 上的交付候选；A 的专家材料打包脚本需按 §4 增量更新（PR #17 评论已知会）。

## 1. 检查方法

1. **存在性**：D 全部交付文件被 Git 跟踪（非仅本机），且 `.gitignore` 的 negate 放行规则与跟踪状态一致（`comm` 双向 diff 无意外）。
2. **数字一致性**：每项关键数字在三个载体间核对——Markdown 报告 ↔ JSON 证据 ↔ stdout 日志。
3. **交叉引用**：报告引用的路径、Release、commit、digest 逐一可达。
4. **权重复核**：用 `scripts/verify_response_artifact.py` 的 `artifact_digest()` 原方法复算本地 adapter 目录 digest。

哈希基准：全部以 main（PR #20 已合并，merge `6778aa5`，2026-10-17 15:17 +08）现内容为准；下表哈希为合并前后同值复算（merge 不改文件内容；抽样 4 项已在 main 上复算逐位一致）。打包前可复跑 §6 命令复核字节一致。

## 2. D 岗位材料清单（按最终候选 v2）

### 2.1 最终候选核心材料（建议进专家材料包）

| 文件 | 字节 | SHA-256 前缀 | 说明 |
|---|---|---|---|
| `reports/D_DAY4_SFT_FULL_2EP.md` | 3,727 | `7a536dff146c2d30` | 2-epoch 训练运行报告 |
| `reports/d_day4_sft_full_2ep.json` | 7,418 | `158254493dcc54f4` | 训练指标/门禁/复现命令 |
| `logs/D_DAY4_SFT_FULL_2EP.txt` | 21,515 | `c2712cde91069e39` | 训练 stdout 全量日志（step 级） |
| `results/response_gen/d4_2ep_val_full_report.json` | 695 | `57c8f6bebc9d6143` | 全量推理性能报告 |
| `logs/d4_2ep_val_full.txt` | 439 | `9e213117d852228b` | 推理 stdout 摘要 |
| `reports/d_day4_2ep_comparison_summary.txt` | 1,413 | `b6d6f9f4cfefd05d` | 同集对比摘要（分桶/逐条） |
| `reports/D_DAY4_RESPONSE_ERROR_ANALYSIS.md` | 13,797 | `e0f455b627c87a5d` | 错误分析 + cap512 负结果 + v2 执行全程与结论 |
| `configs/response_sft_2epoch.json` | 826 | `082b70e670e557c1` | 候选 v2 训练配置（与 D2 首轮仅 epochs 1→2） |
| `reports/D_DAY4_GPU_RUNBOOK.md` | 5,986 | `98008fdde8009af5` | 远程执行序列（含出网实测） |
| `docs/TECHNICAL_SOLUTION.md` | 5,212 | `47c4c2d28f158fcd` | 技术方案（共享文档，设计层无逐候选数字） |
| `weights/README.md` | 2,531 | `dd7ccc70cb96038b` | 权重 Release 指针（D2 与 v2）；本哈希即 main 现值（PR #20 合并后） |

### 2.2 调优过程记录（呈现完整实验设计，建议随包或留仓库）

| 文件 | 字节 | SHA-256 前缀 | 说明 |
|---|---|---|---|
| `reports/D_DAY3_RESPONSE_LORA_SFT.md` | 10,280 | `ed1410a72f531a1a` | D2 首轮（1-epoch）设计与报告 |
| `reports/D_DAY3_SFT_FULL_1P5B.md` | 3,726 | `bdb5265aef1d1782` | D2 首轮训练报告 |
| `reports/d_day3_sft_full_1p5b.json` | 7,169 | `8687f9ab5c6ec55c` | D2 首轮指标 |
| `logs/D_DAY3_SFT_FULL_1P5B.txt` | 10,987 | `9569f2e3dcf74d86` | D2 首轮训练日志 |
| `logs/D_DAY3_SFT_SMOKE_1P5B.txt` | 811 | `c46294bce1b34710` | D2 冒烟日志 |
| `reports/D_DAY3_MINLENGTH_PROBE.md` | 7,929 | `01ce24eb212e8a8f` | minlen 探针负结果（压短有害） |
| `reports/D_DAY3_CONDITIONED_ARM.md` | 7,476 | `7589982d0935bbfc` | 条件注入消融（−0.2%，不建议开启） |
| `reports/D_DAY3_ORIGINAL_MODEL_VAL_FULL.md` | 7,808 | `f94c2197dcc16313` | 原模型全量臂（三路对比基准） |
| `reports/D_DAY3_ARTIFACT_VERIFICATION.md` | 10,927 | `ed621ec5e8003acc` | 离线自包含/冷启动确定性门禁 |
| `configs/response_artifact.json` | 1,234 | `54c74c0511fa484d` | **当前固定 D2 digest `af4f0e38…`**；v2 采用时需 A 同步 |

### 2.3 全量逐行证据（体积大，留仓库、包内放摘要）

| 文件 | 字节 | SHA-256 前缀 |
|---|---|---|
| `reports/d_day4_2ep_comparison.json` | 1,314,154 | `9e2e8b443e1f7857` |
| `reports/d_day4_response_error_analysis.json` | 1,952,015 | `cb13ec701a8dbb6c` |
| `results/response_gen/d4_2ep_val_full.jsonl` | 1,050,402 | `6eb7509decfd589d` |
| `results/response_gen/d2_1p5b_val_full.jsonl` | （B2 join 用） | 历史 force-add 跟踪，无 negate 规则 |

### 2.4 权重（不入 Git，Release 分发）

- v2 adapter 本地目录 `weights/response_sft_d4_2ep/`：digest **`b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb`**，10 文件 / 52,866,512 字节——本检查用 `artifact_digest()` 原方法**复算一致**。
- Release [`d-day4-response-sft-2ep-20261017`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-day4-response-sft-2ep-20261017)（Pre-release）：zip `e82edaf0…` + 边车；发布后已按包内文档实跑核验（下载→边车哈希→包内 11/11→10 文件 digest）。

## 3. 数字一致性核验结果

| 数字 | 报告（md） | 证据（json） | 日志（stdout） | 一致 |
|---|---|---|---|---|
| 训练步数 | 4,664 | `optimizer_steps: 4664` | `steps=4664` | ✓ |
| 速度 | 1.7514 秒/step | `seconds_per_step: 1.751365` | `sec/step=1.7514` | ✓ |
| mean loss | 2.017（D2 2.159） | `mean_train_loss: 2.017` | step 末 2.0172–2.0175 | ✓ |
| first loss | 2.969118（与 D2 逐位相同） | `first_loss: 2.969118` | — | ✓ |
| 重载/生成复现 | 392/392 + 3/3 | `lora_weights_match: true`、`generation_match: true` | `reload=True` | ✓ |
| 显存峰值 | 7.2 GiB | `vram_peak_bytes: 7219716608` | `vram=7219716608` | ✓ |
| 推理条数 | 2,354/2,354 | `samples: 2354` | `2354 samples` | ✓ |
| 推理耗时 | 7,809.678s、3.3176 秒/条 | `total_seconds: 7809.678`、`mean_seconds_per_sample: 3.3176` | `7809.7s (3.318s/sample)` | ✓ |
| 吞吐/空回复 | 24.95 tok/s、0 空 | `tokens_per_second: 24.95`、`empty_responses: 0` | — | ✓ |
| ROUGE-L / BLEU-4 | 0.2542→0.2598 / 0.1075→0.1116 | summary 与 comparison json `arms` 一致 | — | ✓ |
| 长金标桶 | >200：0.1904→0.1958；101-200：0.2058→0.2142 | 同上 | — | ✓ |
| 逐条 head-to-head | 1,255 胜/1,035 负/64 平 | `head_to_head` 一致 | — | ✓ |
| 附带变化 | near_empty 32→40、heading 23→33、blockquote 28→29 | `arms.*.issues` 一致 | — | ✓ |

推理"比 D2 快 12%"：3.3176 / 3.7572（D2 首轮同机同路径）= 0.883。第二 epoch loss 降幅 6.6%：(2.159−2.017)/2.159。派生数字与独立测量同量级，交叉自洽。

## 4. 对 PR #17 专家材料打包清单的增量（v2 已成 main 交付候选）

`scripts/package_expert_materials.py` 的 `MATERIALS` 元组当前只含 D2 首轮材料，`manifest.candidate` 为 `"C retrained emotion; empty profile; D Day3 1.5B LoRA"`。v2 已随 PR #20 合并（merge `6778aa5`），建议将元组改为（D 部分；C 部分由 C 定）：

```python
MATERIALS = (
    "docs/TECHNICAL_SOLUTION.md", "docs/TRAINING_WORKPLAN.md",
    "reports/EXPERIMENT_TEMPLATE.md",
    # C 材料（不变，由 C 维护）
    "reports/C_DAY3_EMOTION_PROFILE.md",
    "reports/C_DAY3_RETRAINED_HANDOFF_20261015.md",
    "reports/C_DAY4_PROFILE_UNWEIGHTED.md",
    "reports/c_day3_retrained_metrics.json",
    "reports/c_day4_profile_unweighted_metrics.json",
    # D 历史候选（消融基线，保留呈现调优过程）
    "reports/D_DAY3_RESPONSE_LORA_SFT.md", "reports/D_DAY3_SFT_FULL_1P5B.md",
    "reports/d_day3_sft_full_1p5b.json",
    "logs/D_DAY3_SFT_FULL_1P5B.txt", "logs/D_DAY3_SFT_SMOKE_1P5B.txt",
    # D 候选 v2（新增）
    "reports/D_DAY4_SFT_FULL_2EP.md", "reports/d_day4_sft_full_2ep.json",
    "logs/D_DAY4_SFT_FULL_2EP.txt", "reports/d_day4_2ep_comparison_summary.txt",
    "reports/D_DAY4_RESPONSE_ERROR_ANALYSIS.md",
    "configs/response_sft_2epoch.json",
    "configs/integrated.json",
    "configs/integrated_artifacts.json",
)
```

并将 `candidate` 更新为 `"C retrained emotion; empty profile; D Day4 2-epoch LoRA (v2, adapter digest b3e9017e…)"`（以 A 最终决定措辞为准）。逐行大文件（§2.3）不进包，包内放摘要 txt；`reports/d_day4_2ep_comparison.json` 如需入包注意 1.3MB 体积。D2 首轮材料保留在清单中作为消融基线（调优过程），不是回退交付物。

## 5. 本次检查中发现并修复的问题

- `logs/README.md` 与 `results/README.md` 与实际 `.gitignore` 漂移：两文件还称"只放行 D 岗位 Day2 证据"，实际 negate 规则已按轮次扩展到 Day3/Day4（训练日志、全量推理日志、cap512 冒烟臂、2-epoch 候选证据），且 `results/response_gen/artifact_verify_day3/*`、`d2_1p5b_val_full*.jsonl` 等为历史 force-add 跟踪（无 negate 规则）。本 PR 同步修正两份 README，并在其中登记完整放行清单，避免后续维护者误判。
- 无其他缺失：D 报告引用的路径/Release/commit/digest 全部可达；`.gitignore` negate 清单与跟踪文件双向一致（negate 的全部被跟踪；反向多出的 9 个文件为 C 的 `C_DAY2_EMOTION_SMOKE.txt`、`artifact_verify_day3/` 6 个冷启动证据与 `d2_1p5b_val_full.jsonl(+_report.json)`，均为历史 force-add 跟踪、无 negate 规则，有跟踪即有效）。

## 6. 复现命令

```bash
# 存在性与哈希（在含 PR #20 内容的分支上执行；合入 main 后同值）
sha256sum reports/D_DAY4_SFT_FULL_2EP.md reports/d_day4_sft_full_2ep.json \
  logs/D_DAY4_SFT_FULL_2EP.txt logs/d4_2ep_val_full.txt \
  results/response_gen/d4_2ep_val_full_report.json \
  reports/d_day4_2ep_comparison_summary.txt reports/D_DAY4_RESPONSE_ERROR_ANALYSIS.md \
  configs/response_sft_2epoch.json reports/D_DAY4_GPU_RUNBOOK.md
# adapter 目录 digest（本地 weights/ 就位后）
python3 -c "import sys; sys.path.insert(0,'scripts'); from pathlib import Path; from verify_response_artifact import artifact_digest; print(artifact_digest(Path('weights/response_sft_d4_2ep')))"
# 期望：digest b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb, files 10, bytes 52866512
```

## 7. 边界与遗留

- 本检查是材料完整性核验，不替代 B2 同集指标复核（B2 可将 v2 臂纳入，预测已在 main `results/response_gen/d4_2ep_val_full.jsonl`）；v2 已由 A 合并 PR #20 成为 main 交付候选，D2 首轮保留为消融基线与回退依据。
- v2 已采用：`configs/integrated.json` 的 adapter 与 `configs/response_artifact.json` 的 digest 更新属跨组改动，由 A 执行（PR #20/#17 评论已知会）；截至本报告 `configs/response_artifact.json` 仍固定 D2 digest `af4f0e38…`（见 §2.2）。
- 执行计划 D 行"材料检查 Done"登记已在本 PR 追加（PR #20 合并后无表尾冲突）。
- 专家材料最终清单以 A 的 `package_expert_materials.py` 为准，本报告 §4 为 D 侧建议。
