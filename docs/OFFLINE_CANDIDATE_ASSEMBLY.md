# A T4：离线字段对齐与候选组装

日期：2026-10-28。实现及真实数据组装完成；团队验收按既有 A→B 流程进行。验收记录留在本地 `.codex_tmp/acceptance/cycle2/a_t4_20261028/`。

## 固定输入与交付范围

以同步后的 main `72d5c37` 为输入基线，沿用 A 固定分支 `feature/a-mlops`。

本次直接复用最新已交付模型的预测，不启动模型推理：

| 字段 | 模型与预测来源 |
|---|---|
| 回复（新实验臂） | D T3 `weights/t3_unconditional_2ep`；`results/response_gen/t3_unconditional.jsonl` |
| 回复（同设备对照臂） | D4 v2；D T3 阶段同卡重跑的 `results/response_gen/t3_d4v2_off_samedevice.jsonl` |
| 情感 | C Day5 `weights/emotion_longtail075_b1_20261016`；`results/response_gen/c5_conditions_val_full.jsonl` |
| 画像 | C 保留的 `weights/profile_encoder_v1_b1_20261026`；`reports/c_cycle2_profile_v1_predictions.jsonl` |

C T3 v2 已交付 no-go，故采用 v1。C T4 情感改进和 D T4 长回复候选属于同期工作，不是本次输入。D T3 最新预测只有冻结筛选集 160 条，本次两臂各组装 160 条，不用历史 D4 回复补齐或冒充 T3 全量结果。

候选保存在 `results/cycle2_t4_20261028/`：

- `t3_unconditional_c5_profile_v1.jsonl` 和同名 `.manifest.json`。
- `t3_d4v2_off_samedevice_c5_profile_v1.jsonl` 和同名 `.manifest.json`。

模型来源按已有生成报告及 C/D 交付记录登记；本次不重新加载 checkpoint，不将模型路径声明冒充权重重载验证。组装不更改部署默认，也不决定 T3 是否替换 D4。

## 使用

从项目根目录运行，输出路径必须尚不存在；对照臂将回复文件及对应报告换为 `t3_d4v2_off_samedevice` 即可。下面命令可在新的输出目录复现：

```bash
python scripts/assemble_candidate.py \
  --input results/response_gen/t2_selection_input.jsonl \
  --responses results/response_gen/t3_unconditional.jsonl \
  --response-report results/response_gen/t3_unconditional_report.json \
  --emotions results/response_gen/c5_conditions_val_full.jsonl \
  --profiles reports/c_cycle2_profile_v1_predictions.jsonl \
  --emotion-model weights/emotion_longtail075_b1_20261016 \
  --profile-model weights/profile_encoder_v1_b1_20261026 \
  --source-record reports/t3_uncond_2ep_report.json \
  --source-record reports/c_cycle2_profile_v1_20261026.json \
  --source-record reports/c_cycle2_profile_v2_micro_calibration_20261026.json \
  --source-record results/response_gen/c5_conditions_val_full_report.json \
  --output results/t4_reproduce/t3_unconditional_c5_profile_v1.jsonl \
  --manifest results/t4_reproduce/t3_unconditional_c5_profile_v1.manifest.json
```

`--input` 支持官方推理输入或 B1 reference。B1 仅提取 ID、会话、目标轮、历史，绝不读取金标情感或画像作为候选输出。

## 校验与来源

复用已有 `_index_by_id`、B 提交校验器和官方字段/标签校验。所有来源拒绝缺失、非法或重复 ID；回复来源必须与输入 ID 集合完全一致。分类来源允许覆盖更大的验证集，清单记录未使用行数。本次情感及画像各有 2354 行，选取 160 行、各剩余 2194 行。

按输入 ID 顺序发布，源顺序不一致时按 ID 修正并报告错位数。回复、情感和画像来源如显式携带 conversation_id，均与输入核对；错配（含 null/空值）直接拒绝，不携带该字段的历史预测仍可通过 ID 对齐。检查非空回复、官方标签、三维画像、memory_refs、输出数量和 ID 顺序。只替换分类字段，回复字符串逐字保留；没有 memory_refs 时显式使用空数组（没有 memory bank）。

provenance v2 使用带 `path_base` 的相对路径：`repo` 相对仓库根目录，`artifact` 相对候选包目录（默认输出父目录，可用 `--artifact-root` 指定）。优先将包内文件记录为 artifact-relative，移动输出目录不改变 manifest 字节。外部来源须通过 `--provenance-root` 声明根目录，不接受无法归属的主机绝对路径。模型 ID 相对 repo；代码指纹按 LF 规范化。T4/T5 清单已重新生成，预测内容未改变。

manifest 保存所有来源文件、交付记录及输出的 SHA-256，另记 LF 规范化哈希，以区分 Windows CRLF 与 Git LF。回复必须匹配生成报告中的输出哈希与行数，且报告须声明无条件生成；带条件回复不允许套用该工具冒充分类替换。两次读取指纹确认来源在组装期间未变化。输入、输出和 manifest 不得同路径，且不覆盖已存在产物。

新实验臂输出 SHA-256：`6e40e3ebba507dbec193e6a64a939511d2e41f9ad9789afc30af10be26cf069b`。

同设备对照臂输出 SHA-256：`9c7e5a4ac9615b14573a903540551d2652ee29db58e11ef31a0f53b47a985f91`。

两臂均通过独立 B 提交校验器，重复组装输出字节一致。新增 8 项单测覆盖错序修正、缺行、重复、额外回复、分类超集、非法标签、空文本、会话错配、报告哈希及条件不符、禁止覆盖和金标排除。

本地检查：新增 8 项测试、compileall、CI synthetic smoke 均通过。使用 Git Bash 后全套 377 项中 365 项通过、11 项跳过，已有 `test_remote_scripts.WatchAndGenerateTests.test_proceeds_when_training_completed` 在 Windows 失败：测试将未加引号的 Windows 路径拼入 Bash 重定向命令，目标文件未落在预期位置。该测试和远程脚本未在本次改动中修改；完整日志仅留本地验收目录。含实际 checkpoint 训练/恢复的跳过项不作为本次验证证据。

## 评测边界

这是离线候选，不是正式端到端推理或速度证据。manifest 显式标记 `formal_end_to_end_timing=false`、`model_inference_performed=false`、`classification_used_to_generate_response=false`。输出分类字段没有参与已有回复的生成，不能用这些结果宣称情感/画像条件改善了回复；条件消融仍需按后续实验计划真实生成。T5 可复用该工具组装届时已经交付的固定候选。
