# D 岗位 Day 2：回复生成选型与训练准备

**状态：候选选型、LoRA SFT 训练模板与本机 CUDA 冒烟已产出；正式训练仍 Blocked。** 平台 GPU/卡时未由 A 核实、B1 数据 v1 未落地，本 PR 不声称启动平台训练；本机 RTX 4060 冒烟不消耗平台卡时，与 C 的 MacBERT 冒烟同口径。

## 按新版 main 的任务范围

对齐 `origin/main` 的 `docs/EXECUTION_PLAN.md`（10/13–20 训练关键路径）与 `docs/TRAINING_WORKPLAN.md`：D 负责 D1（比较 ≤2 个生成候选，选 1 个做微调冒烟；交付 revision/许可、提示与训练模板、可训练样本数、显存、ETA）与 D2（10/14 18:00 前冻结训练集首轮 LoRA/SFT 可评测候选）。本 PR 只覆盖 D1 的 CPU/本机 GPU 准备部分，模型选型结论与训练模板冻结如下，首轮训练等 A 的平台 GPU 结论与 B 的数据 v1。

## 候选与决定

| 对照 | 加载/结构 | 同 100 条验证点字符级指标 | 资源（本机 4060，LoRA bf16，batch 1×累积 8，max_length 2048） | 决定 |
|---|---|---|---|---|
| 既有检索 baseline（保底） | 纯 Python n-gram | BLEU-4 0.0267 / ROUGE-L 0.1375 | CPU，索引 25s、全量推理约 6 分钟 | 保留为保底链路，正式候选不采用 |
| `Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775` | 标准 chat template | BLEU-4 0.0259 / ROUGE-L 0.1377 | 4.6 GiB 显存，0.93 秒/step，全量 ETA 36 分钟/epoch，生成 43.1 tok/s | 备选/回退候选 |
| `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 同上 | **BLEU-4 0.0377 / ROUGE-L 0.1565** | 6.7 GiB 显存，1.18 秒/step，全量 ETA 46 分钟/epoch，生成 35.7 tok/s | **主候选** |

同切片三路对比见 `reports/d_day2_response_comparison_val100.json`。原模型（未微调）在同切片已优于检索 baseline，与"微调是否进一步提升"的研究问题衔接；字符级 BLEU/ROUGE 仅本地诊断，非官方口径。两个模型仓库页面均标注 Apache-2.0；这只确认仓库所列许可证，**赛事对预训练模型的使用许可仍待 A 按官方规则核实**。

## 训练数据与模板契约

- 可训练样本：官方 raw train 全部 **18,657 个 prediction points**（8,338 会话），由 `src/response/sft_data.py` 从 `turns` + `prediction_points` 生成；样本 ID `{conversation_id}_t{target_user_turn_id}`。B1 转换产物未落地，本 PR 只用官方 raw 数据。
- 防泄漏契约：上下文只含 `turn_id <= target_user_turn_id` 的轮次，被预测的助手回复及其后所有轮次全部排除；监督标签取 `target.response_text`（与转写正文 18,657/18,657 一致，`response_turn_consistency` 记录在运行报告）。`memory_refs` 不进入提示：官方无 memory bank，最终候选恒输出 `[]`，不伪造引用。
- 条件注入：训练用 `condition_mode: gold`（金标 emotion_label + user_profile 注入 system 提示）；推理侧 `build_eval_sample` 用 C 的预测条件或 `none`，提示结构与训练完全一致。"预测条件注入是否优于无条件注入"是 D/C 的待回答问题，**不得把验证真值标签用于最终候选冒充部署结果**。
- 长度策略：回复 token p50=106、p99=826、max 865；长的是历史（总长 p95 3550）。`max_length: 2048`，超长时**先按消息级丢弃最旧历史轮**（保住 system 条件块），仅剩 system+目标轮仍超长才 token 级左切；本批 512 样本截断 0 条、跳过 0 条。
- 19 个契约测试（`tests/test_response_sft.py`）：截断/泄漏、条件注入、窗口、重复点、空回复、eval 输入形状、一致性统计；全仓 45/45 通过。

## 冒烟运行证据（本机 CUDA，不计平台卡时）

设备：本机 NVIDIA RTX 4060 Laptop 8GB；torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2、accelerate 1.15.0；seed 20261013；batch 1 × 梯度累积 8；LoRA r=8 alpha=16 dropout=0.05（q/k/v/o/gate/up/down）；lr 1e-4；梯度检查点开。

| 项 | 0.5B（备选） | 1.5B（主候选） |
|---|---|---|
| optimizer steps / mean loss | 50 / 3.1871（first 3.1247） | 50 / 2.6953（first 3.0751） |
| 秒/step / 全量 ETA | 0.9259 / 2160s（36 分钟，未含 +30% 余量） | 1.1756 / 2743s（46 分钟，未含 +30% 余量） |
| 显存峰值 | 4.59 GiB | 6.72 GiB |
| 首步 LoRA 梯度范数（有限非零） | 1.9985 | 1.2746 |
| LoRA 权重变化 / adapter 重载一致 | 是 / 是 | 是 / 是 |
| 生成探针（贪心，非空） | 3/3，中文共情风格正常 | 3/3，中文共情风格正常 |
| adapter | 17,640,136 B，SHA-256 `f1d60e0362d38a2f3b5e433d93d7550b92267261f9e02ce4b3ea76151b16fccc` | 36,981,072 B，SHA-256 `f8072be7d27f8bac22edaac660083b48812ae106c643419fc0aee53462fb697c` |

50 steps 冒烟只证明链路（loss 有限、梯度有效、权重改变、checkpoint 可重载、生成非空），**不代表候选质量**；未微调 1.5B 的同切片指标已单独记录。训练与推理吞吐用于全量评测预算：1.5B 原模型 100 条 241.4s（35.7 tok/s），全量 2,354 点外推约 95 分钟；0.5B 147.2s（43.1 tok/s）。

运行脚本 SHA-256（本次冒烟运行时）：训练 `f6eb307f1e34a2ea7d63bf46a8616c1728ba7c0c66d0124cdbb335b8cf3ed3c3`；推理 `d487bd726dc2fb9270bfafd3e4cc1806e8f92a35aa94a0c64e87dc632eb35684`。逐步 stdout 与逐样本指标：`logs/D_DAY2_SFT_SMOKE_0P5B.txt`、`logs/D_DAY2_SFT_SMOKE_1P5B.txt`、`logs/D_DAY2_ORIGINAL_{0P5B,1P5B}_VAL100.txt`、`reports/d_day2_sft_smoke_{0p5b,1p5b}.json`；权重在本机忽略目录，不提交 Git。

## 验收与下一步

- 选型、模板与冒烟已提交 PR，等待 A 验收（A 审 D 的 PR）。
- A 核实平台 GPU/卡时/环境、B 发布数据 v1 后：用同一脚本与 conversation 分组契约在冻结训练集上跑 D2 首轮（目标 10/14 18:00 可评测候选），先 ≥50 step 短跑重算 ETA（+30% 余量），再决定 epoch/长度。
- 条件注入对比（无条件 vs C 预测条件）与三路同集评测（检索/未微调/微调）由 B2 执行；D 提供生成侧复现命令与接口。
- 本机 4060 冒烟不替代平台训练、全量评测、官方镜像集成或赛事许可确认；保底检索链路继续可运行。
