# D 岗位 CYCLE2 T2：关闭 vs 仅情感两臂比较（固定分层筛选集）

日期：2026-10-23（Asia/Shanghai）。分支：`feature/d-day7-deadline-standby`。状态：T2 两臂比较完成（报告待随 PR 交 A 审）。

对应执行计划 T2（D 列）：「用 T1 脚本在固定分层筛选集比较最新 C Day5＋D Day4：关闭 vs 仅预测情感；固定权重/解码，匹配的已有基线不重算，不立即跑两臂全量」。

## 结论摘要

- **注入预测情感不改变可测的回复质量**：同设备（本机 4060）同权重同解码下，仅情感臂 78.8% 的回复文本发生变化，但本地字符级诊断指标几乎不动——ROUGE-L 0.2497→0.2490（−0.0007）、BLEU-4 0.1028→0.1000（−0.0028）、平均长度 +0.7 字；逐条方向混杂（63 条变好 / 60 条变差 / 37 条持平），无系统性收益。
- **条件效应小于设备噪声**：同条件同权重同解码下，4090 与 4060 的逐条字节一致率仅 46/160（28.75%），其聚合指标差（ROUGE-L +0.0036）比条件效应（−0.0007）大一个量级。**两臂比较必须同设备做**，否则小幅差异无法归因。
- **建议：维持条件关闭（部署默认不变）**。开启情感条件零成本但也无可测收益，与 10/15 C Day3 条件臂结论（−0.2%，不建议开启）方向一致；本机 160 条诊断不冒充官方分数。
- 未跑两臂全量（符合计划「不立即跑两臂全量」）；T3 无条件 SFT 仍待远程 GPU。

## 一、实验设置（固定变量清单）

| 项 | 值 |
|---|---|
| 输入 | `results/response_gen/t2_selection_input.jsonl`（160 条，160 会话，12 格分层，seed 20261022，T1 冻结） |
| 权重 | `weights/hf/Qwen2.5-1.5B-Instruct` + `weights/response_sft_d4_2ep`（D Day4 v2，digest `b3e9017e…`） |
| 条件来源 | `results/response_gen/c5_conditions_val_full.jsonl`（C Day5，2354 条，Accuracy 0.716653 / Macro-F1 0.584843，T1 与 C 报告逐位对账） |
| 解码 | max_new_tokens=256，其余脚本默认 |
| off 臂 | 历史 D4 v2 全量预测按 id 抽取（4090 生成，零 GPU，`t2_condition_off.jsonl`）+ 本机 4060 同条件复跑（设备对照用，`t2_off_local4060.jsonl`） |
| 仅情感臂 | 本机 4060 复跑，`--condition-fields emotion`（丢 user_profile，T1 交付） |

三臂产物均实跑核验：160 行、id 集合与筛选集完全一致、response_text 全非空、条件覆盖率 160/160、报告回显 device=cuda/dtype=bfloat16/max_new_tokens=256。

## 二、同设备两臂比较（结论以此为准）

本机 4060，off（`t2_off_local4060.jsonl`，sha256 `0498e8b6…`）vs 仅情感（`t2_emotion_only.jsonl`，sha256 `1070bfa8…`）：

| 指标 | off | 仅情感 | 差值 |
|---|---:|---:|---:|
| BLEU-4（字符级，本地诊断） | 0.1028 | 0.1000 | −0.0028 |
| ROUGE-L（字符级，本地诊断） | 0.2497 | 0.2490 | −0.0007 |
| 平均长度（非空白字符） | 124.7 | 125.4 | +0.7 |
| 逐条文本相同 | — | — | 34/160（21.2%） |

逐条 ROUGE-L 差：中位数 0.0000，p25 −0.0136，p75 +0.0165；63 条变好 / 60 条变差 / 37 条持平。长度差：60 条变长 / 62 条变短 / 38 条不变。定性看，情感条件改变的是措辞与收尾（见 `t2_arms_comparison_same_device.json` 的 `divergence_examples`），结构与信息量不变。

## 三、设备对照实验（噪声底，本报告新增）

为排除「off 臂来自 4090、情感臂来自 4060」的混杂，在本机以完全相同的条件（off）、相同权重、相同解码复跑 160 条，与 4090 历史 off 臂逐条比对：

- 逐条字节一致：**46/160（28.75%）**；sha256 分别为 `0498e8b6…`（4060）与 `6e9efeaa…`（4090）。
- 聚合指标差：BLEU-4 +0.0011、ROUGE-L **+0.0036**（4090 侧略高）——与条件效应同量级甚至更大。
- 两侧 mean 4.63 vs 3.32 s/sample，速度差约 40%，但不影响上述结论。

**含义**：在 1.5B + bf16 + greedy 解码下，跨设备数值差异足以改变七成以上样本的生成文本。任何跨设备的 A/B 比较（条件、权重、解码探针）在此粒度上不可归因；T4/T5 及 B 的候选比较如涉及不同设备产物，必须先做同设备对照或重跑。

## 四、结论与建议

1. **不开启情感条件**（部署默认 condition 关闭不变）。证据：同设备比较无可测收益（ROUGE-L −0.0007，低于设备噪声底一个量级），逐条方向混杂，长度无变化；与 10/15 C Day3 条件臂负结果一致。OI 无分数反馈，按本地固定验证规则不做无依据开启。
2. **T5 第三臂（情感＋可靠画像）仍按计划评估**，但必须同设备生成；若画像条件同样无可测收益，维持关闭。
3. **设备一致性作为团队纪律**：后续两臂/多候选比较固定同一张卡；跨设备复用历史预测只能做量级参考，不能做采用依据。本条已写入本报告与计划行，建议 B 在 T3/T4 独立复核时同样核对产物设备。

## 五、验证证据索引

- 全量单测：`python -m unittest discover -s tests` → **277 tests, OK**（合并后基线 266 + T1 收尾 `extract_off_arm_subset` 6 + 本轮 `compare_t2_arms` 5）。
- CI 等价检查实跑：`python -m compileall -q src scripts tests`、`bash -n start.sh`、`bash -n scripts/cloud_smoke.sh`、`python scripts/ci_smoke.py` 均 PASS。
- 主会话独立复验（非采信子智能体结论）：三臂行数/id 集合/非空/条件覆盖率/报告字段/两个 off 臂 sha256 与设备对照一致率（46/160）均由主会话重跑确认。
- 产物（均随 commit 入库）：`results/response_gen/t2_condition_off.jsonl`、`t2_off_local4060.jsonl`、`t2_emotion_only.jsonl`、`t2_emotion_only_report.json`、`t2_arms_comparison_same_device.json`、`t2_device_control_comparison.json`。
- 本地 BLEU-4/ROUGE-L 为字符级诊断实现（`src/evaluation/metrics.py`），官方指标未公布，本报告所有分数不冒充官方分数。

## 六、T3 衔接

- T3（无条件两轮 SFT）配置 `configs/response_sft_unconditional_2ep.json` 就绪，待远程 GPU 时段（本机外推 15.7h 不可行）；本机 4060 实测 4.63 s/sample 可用于重估全量生成耗时（2354 条约 3.0h @4060，@4090 约 2.2h）。
- 远端实例 183.222.230.87:58425 于 10/23 两次实测不可达（Connection closed）；T3 训练与全量复跑需先恢复实例或更换，恢复命令见 `reports/D_CYCLE2_T1_PREP.md` 第四节。
