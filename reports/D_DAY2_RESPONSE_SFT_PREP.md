# D 岗位 Day 2：回复生成选型与训练准备

**状态：D1 审查修复已交付（PR #8 更新），本机冒烟证据已用修正后脚本重跑；正式训练仍 Blocked。** 平台资源已由 A 调查确认（PR #5 已合 main：40 个 RTX 4090 卡时、10 小时无卡模式、30GB 镜像/20GB 文件额度、10/20 23:59 截止、vLLM 0.19 大赛专属镜像）；B1 数据 v1（PR #7）尚未合并。本机 RTX 4060 冒烟不消耗平台卡时。

## D1 审查修复记录（2026-10-13，回应 A 对 PR #8 的五条必须修复项）

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 1 | 计时每个 micro batch 重置、只累加累积窗口最后一个 micro batch，accum=8 时秒/step 低估约 8 倍（1.5B 报 1.18 秒/step、ETA 46 分钟，严重低估） | 按完整梯度累积窗口计时（窗口首个 micro batch 起、optimizer.step 后止），两端 `torch.cuda/mps.synchronize`，跳过前 3 个 warm-up step 后取平均；ETA 显式给 +30% 故障余量 | 两份冒烟用修复后脚本重跑（下节）；ETA 修正为 1.5B ≈ 7.8h、0.5B ≈ 5.0h（裸估） |
| 2 | `--eval-prompts` 默认 3 但正式模式（无 `--max-steps`）训练保存后才抛异常，正式命令必然失败 | 正式模式同样允许探针；非法参数（0/负数）在加载模型前由 `_validate_run_limits` 拒绝 | `tests/test_train_response_sft.py` 锁定；CPU 回归运行不带 `--max-steps` 正常跑完 |
| 3 | `epochs` 只进 scheduler/ETA，循环实际只跑一遍；`randperm` 生成的 order 未被使用 | 每 epoch 用同一 seeded RNG 独立洗牌取样（`_epoch_micro_batches` 纯函数），实现真正多 epoch 循环 | 单测锁定 shuffle/覆盖/确定性；`reports/D_DAY2_EPOCHLOOP_CPU.md` 双 epoch 实跑（epoch=1/2、2/2 各 16 个 micro batch） |
| 4 | 所谓 adapter 重载只比对一个 lora_A 张量、正式模式直接记 `reload_ok=True` | 保存后释放训练模型，离线加载 base + `PeftModel.from_pretrained` 重载 adapter，逐个比对全部 LoRA 张量，并用贪心确定性生成与训练内模型逐条复现；未执行通过的检查不记通过，失败即报错中止 | 冒烟报告 `checks.adapter_reload`：1.5B 392/392、0.5B 336/336 张量一致，3/3 生成复现相同 |
| 5 | 冒烟报告复现命令缺 `--max-train-samples/--max-steps`；0.5B 命令误用默认 1.5B config | 报告新增 `reproduction` 段（完整命令、config/model-dir、运行代码 commit 与 dirty、脚本 SHA-256）与运行时版本；新增 `configs/response_sft_0p5b.json` | 两份 MD 复现命令可原样复跑；0.5B 报告 model.name/revision/loaded_from 均正确 |

同步修复：`.gitignore` 显式放行 D Day2 审查证据（`logs/D_DAY2_*.txt`、`results/response_gen/` 逐条预测与性能报告），README 约定随之更新——权重仍不入 Git。

## 按新版 main 的任务范围

对齐 `origin/main` 的 `docs/EXECUTION_PLAN.md`（10/13–20 训练关键路径）与 `docs/TRAINING_WORKPLAN.md`：D 负责 D1（候选比较、LoRA SFT 模板与冒烟）与 D2（10/14 18:00 前冻结训练集首轮 LoRA/SFT 可评测候选）。本 PR 覆盖 D1 全部与 D2 的代码准备；首轮训练等 B1 v1 合并与 A 的开机门槛（初版代码就绪后 A 开 1 卡）。

## 候选与决定

| 对照 | 加载/结构 | 同 100 条验证点字符级指标 | 资源（本机 4060，LoRA bf16，batch 1×累积 8，max_length 2048） | 决定 |
|---|---|---|---|---|
| 既有检索 baseline（保底） | 纯 Python n-gram | BLEU-4 0.0267 / ROUGE-L 0.1375 | CPU，索引 25s、全量推理约 6 分钟 | 保留为保底链路，正式候选不采用 |
| `Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775` | 标准 chat template | BLEU-4 0.0259 / ROUGE-L 0.1377 | 4.59 GiB 显存，7.70 秒/step，全量 1 epoch ≈ 17,971s（4.99h），+30% 余量 23,363s（6.49h） | 备选/回退候选 |
| `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 同上 | **BLEU-4 0.0377 / ROUGE-L 0.1565** | 6.72 GiB 显存，12.11 秒/step，全量 1 epoch ≈ 28,243s（7.85h），+30% 余量 36,717s（10.20h） | **主候选** |

同切片三路对比见 `reports/d_day2_response_comparison_val100.json`（逐条预测与性能报告：`results/response_gen/original_{0p5b,1p5b}_val100.jsonl`，运行日志 `logs/D_DAY2_ORIGINAL_*_VAL100.txt`）。原模型（未微调）在同切片已优于检索 baseline，与"微调是否进一步提升"的研究问题衔接；字符级 BLEU/ROUGE 仅本地诊断，非官方口径。两个模型仓库页面均标注 Apache-2.0；这只确认仓库所列许可证，**赛事对预训练模型的使用许可仍待 A 按官方规则核实**。

**ETA 口径更正**：此前"46 分钟/36 分钟每 epoch"来自被审查确认的计时 bug（只计累积窗口最后一个 micro batch），已撤回并以完整窗口计时重测为准。D2 决策需按修正后预算：workplan 给 D2 暂留 2–8 GPU h，1.5B 单 epoch 裸估 7.85h 已顶到该预算上限（+30% 余量后超出），0.5B 为 4.99h；若平台门槛满足后短跑复测确认，可按 workplan"缩短上下文/减少 epoch/选更小可许可模型"条款决策，候选质量差距（1.5B BLEU-4 0.0377 vs 0.0259）是主要权衡依据。

## 训练数据与模板契约

- 可训练样本：官方 raw train 全部 **18,657 个 prediction points**（8,338 会话），由 `src/response/sft_data.py` 从 `turns` + `prediction_points` 生成；样本 ID `{conversation_id}_t{target_user_turn_id}`。B1 转换产物（PR #7，18,653 条 D 样本、排除 4 条异常）未合并，本 PR 只用官方 raw 数据。
- 防泄漏契约：上下文只含 `turn_id <= target_user_turn_id` 的轮次，被预测的助手回复及其后所有轮次全部排除；监督标签取 `target.response_text`（与转写正文 18,657/18,657 一致，`response_turn_consistency` 记录在运行报告）。`memory_refs` 不进入提示：官方无 memory bank，最终候选恒输出 `[]`，不伪造引用。
- 条件注入：训练用 `condition_mode: gold`（金标 emotion_label + user_profile 注入 system 提示）；推理侧 `scripts/run_response_inference.py --condition-jsonl` 接入 C 的预测条件（按 id 全覆盖，缺一条即报错，拒绝混合条件/无条件提示），提示结构与训练完全一致。"预测条件注入是否优于无条件注入"是 D/C 的待回答问题，**不得把验证真值标签用于最终候选冒充部署结果**。
- 长度策略：回复 token p50=106、p99=826、max 865；长的是历史（总长 p95 3550）。`max_length: 2048`，超长时**先按消息级丢弃最旧历史轮**（保住 system 条件块），仅剩 system+目标轮仍超长才 token 级左切；本批 512 样本截断 0 条、跳过 0 条。
- 测试：`tests/test_response_sft.py`（21 个，样本转换/泄漏/条件/eval 形状）+ `tests/test_train_response_sft.py`（18 个，epoch 计划、shuffle、steps 公式、参数校验、复现命令、adapter key）；全仓 66/66 通过。

## 冒烟运行证据（本机 CUDA，不计平台卡时）

设备：本机 NVIDIA RTX 4060 Laptop 8GB；torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2、Python 3.12.3；seed 20261013；batch 1 × 梯度累积 8；LoRA r=8 alpha=16 dropout=0.05（q/k/v/o/gate/up/down）；lr 1e-4；梯度检查点开。计时口径：每个 optimizer step 按完整累积窗口计时（两端同步 GPU），前 3 个 warm-up step 不计入平均。

| 项 | 0.5B（备选） | 1.5B（主候选） |
|---|---|---|
| optimizer steps（计时步） | 53（50） | 53（50） |
| 秒/step（完整窗口均值） | 7.7031 | 12.1061 |
| 全量 1 epoch ETA（+30% 余量） | 17,971s / 4.99h（23,363s / 6.49h） | 28,243s / 7.85h（36,717s / 10.20h） |
| mean loss（first loss） | 3.1320（2.8770） | 2.6410（2.4118） |
| 显存峰值 | 4.59 GiB | 6.72 GiB |
| 首步 LoRA 梯度范数（有限非零） | 2.3836 | 1.1483 |
| LoRA 权重变化 | 是（336 张量） | 是（392 张量） |
| 离线 adapter 重载 | 通过：336/336 张量一致 + 3/3 确定性生成复现 | 通过：392/392 张量一致 + 3/3 确定性生成复现 |
| 生成探针（贪心，非空） | 3/3，中文共情风格正常 | 3/3，中文共情风格正常 |
| adapter | 17,640,136 B，SHA-256 `d8e65e72148ea090fa27873621887b4ec38b4bf284f3f7b7f8a120529322381a` | 36,981,072 B，SHA-256 `10b595d0330b304f6ef1f1c4985cad37176532bf8c9605a57b78528d7969bf85` |

epoch 循环回归（CPU，0.5B、epochs=2、16 样本）：`reports/D_DAY2_EPOCHLOOP_CPU.md`——两个 epoch 各 16 个 micro batch 全部执行，336 张量离线重载一致，3 条生成复现相同；该运行不带 `--max-steps`，同时证明正式模式探针路径不再崩溃。

53 步冒烟只证明链路（loss 有限、梯度有效、权重改变、checkpoint 离线可重载且生成复现、生成非空、多 epoch 循环），**不代表候选质量**；未微调 1.5B 的同切片指标已单独记录。训练与推理吞吐用于全量评测预算：1.5B 原模型 100 条 241.4s（35.7 tok/s），全量 2,354 点外推约 95 分钟；0.5B 147.2s（43.1 tok/s）。

可独立核验材料：逐步 stdout `logs/D_DAY2_SFT_SMOKE_{0P5B,1P5B}.txt`、`logs/D_DAY2_EPOCH_LOOP_CPU.txt`、`logs/D_DAY2_ORIGINAL_{0P5B,1P5B}_VAL100.txt`；运行 JSON `reports/d_day2_sft_smoke_{0p5b,1p5b}.json`、`reports/d_day2_epochloop_cpu.json`（含完整命令、commit、脚本 SHA-256、适配器 SHA-256）；逐条预测 `results/response_gen/*_val100.jsonl`。权重在本机忽略目录，不提交 Git。

## 验收与下一步

- D1 审查修复与本 PR 等 A 复验收（A 审 D 的 PR）。
- 正式训练开机门槛（剩余工作，按 10/14 计划）：① B1 v1（PR #7）经 C 验收合并后，适配 response JSONL（`context_text` 格式，当前 eval 输入为 `history` 列表）；② A 按 PR #5 门槛开 1 卡实例后 ≥50 step 短跑复测 ETA，再决定 1.5B/0.5B 与 epoch/长度；③ 条件注入对比与三路同集评测由 B2 执行，D 提供 `--condition-jsonl` 接口与复现命令。
- 卡时预算提醒：1.5B 单 epoch 裸估 7.85 GPU h（+30% 余量 10.2h），已顶到 D2 暂留 2–8 GPU h 上限；40 个 4090 卡时还需覆盖 C 的两轮训练，选型需在 A 的队列安排下确认。
- 本机 4060 冒烟不替代平台训练、全量评测、官方镜像集成或赛事许可确认；保底检索链路继续可运行。
