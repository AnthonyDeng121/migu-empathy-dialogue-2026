# B1 data v1 验收报告

本报告依据 `data/processed/b1_full_v1/manifest.json`、`leakage.json`、`anomalies.jsonl`、`SHA256SUMS`，以及转换脚本、测试和 `docs/TRAINING_WORKPLAN.md` 的 B1 要求编写。转换版本为 `b1_v1`，正式产物生成时间为 **2026-10-13 14:31:58（Asia/Shanghai，UTC+08:00）**。统计和哈希摘录自正式产物；本次仅编写报告，没有重新运行全量转换或计算哈希。

## 1. B1 目标

本轮完成 prediction point 全量展开、保留官方 train/val split、按 `conversation_id` 隔离、C 分类样本生成、D 回复样本生成、异常留痕与排除、SHA-256 清单及泄漏检查，交付内容符合工作计划的 B1 数据门槛。

C 样本提供 `emotion_label` 与三维 `user_profile`，复用现有 schema 的标签校验并保留合法空画像数组；D 样本提供 `response_text` 与 `memory_refs`。`source_split` 显式传入，manifest、分类和回复产物一致保留来源。

输入只包含 `turn_id <= target_user_turn_id` 的历史，必须包含目标 user turn。泄漏判断依据轮号及 prediction point 的 assistant 引用关系。`context_text` 按轮号升序排列，每行一个紧凑 JSON 对象，字段顺序为 `turn_id, role, content`，使用 `ensure_ascii=False`、`separators=(',', ':')`，以 LF 连接且末尾无 LF；内容原样保留并进行 JSON 转义。

## 2. 运行环境

正式运行环境由执行者确认：Windows、项目 `.venv`、Python **3.9.13**。默认 seed 为 **20261013**，仅用于 smoke 会话抽样；full 模式不重新随机划分 train/val，也不移动官方会话归属。

## 3. 正式运行命令

在已激活项目 `.venv` 的仓库根目录执行：

```text
python -B scripts/prepare_training_data.py --mode full --train "data/raw/训练-验证-数据集/train/train_public.jsonl" --val "data/raw/训练-验证-数据集/val/val_public.jsonl" --output-dir "data/processed/b1_full_v1"
```

## 4. 全量统计

以下数据取自 `leakage.json`，与 `manifest.json.key_checks` 一致。

| 指标 | train | val |
|---|---:|---:|
| 会话数（conversations） | 8338 | 900 |
| 原始 prediction points | 18657 | 2354 |
| C 分类样本数 | 18653 | 2354 |
| D 回复样本数 | 18653 | 2354 |
| 排除样本数（excluded） | 4 | 0 |

`split_manifest.jsonl` 共 **21011** 条，覆盖全部预测点；每路 C/D 共 **21007** 条可用样本，另有 **4** 条异常记录。数量关系为 `18657 = 18653 + 4`、`2354 = 2354 + 0`。

## 5. 异常说明

实际检测到 4 个异常 prediction point，原始数据未被删除或修改。它们仍保留在 `split_manifest.jsonl` 和 `anomalies.jsonl` 中，均为 `usable=false`、`classification_included=false`、`response_included=false`，不进入 C/D 训练样本。

| conversation_id | target_user_turn_id | assistant_response_turn_id | prediction_point_index |
|---|---:|---:|---:|
| train_conv_000628 | 15 | 14 | 1 |
| train_conv_000817 | 33 | 32 | 1 |
| train_conv_003644 | 7 | 6 | 1 |
| train_conv_003695 | 9 | 8 | 1 |

这 4 个点均引用目标 user 之前的 assistant，而非其相邻后续回复。实际 `reason_counts` 为：

| reason | 命中次数 |
|---|---:|
| assistant_target_not_adjacent_future_turn | 4 |
| target_assistant_in_context | 4 |

这是**同 4 个样本同时命中两个 reason，不是 8 个异常样本**。数量来自本次实际检测，没有硬编码异常数量。

## 6. 泄漏与一致性检查

检查覆盖完整 train+val；以下结果来自 `leakage.json`。

| 检查项 | 实际结果 |
|---|---|
| train/val conversation overlap | 0 |
| duplicate sample_id（全部原始预测点） | 0 |
| context_after_target | false |
| target_assistant_in_context（实际 C/D 样本） | false |
| manifest_samples_consistent | true |
| checks_passed | true |

异常 manifest 可以带 `target_assistant_in_context` reason，表示该原始预测点的 assistant 引用落在历史范围内，因此已被排除。实际进入训练数据的样本不存在目标 assistant 泄漏，也没有目标轮之后的未来 turn。

## 7. memory_refs

训练 gold 的 `memory_refs` 保留官方原值。由于未提供外部 memory bank，仅标记“无法进一步验证引用：未提供外部 memory bank”，没有擅自把训练标签清空为 `[]`，也没有把推理阶段的空引用规则用于修改训练 gold。

## 8. 产物

输出目录为 `data/processed/b1_full_v1/`，包含以下 9 个文件：

| 文件 | 作用 |
|---|---|
| classification_train.jsonl | C 分类训练样本：历史输入、情感和画像标签 |
| classification_val.jsonl | C 分类验证样本，保留官方 val 来源 |
| response_train.jsonl | D 回复训练样本：历史输入、回复与原始记忆引用 |
| response_val.jsonl | D 回复验证样本，保留官方 val 来源 |
| split_manifest.jsonl | 全部预测点的来源、划分、可用状态、异常原因及纳入状态 |
| anomalies.jsonl | 仅保存 `usable=false` 的 manifest 记录 |
| leakage.json | 全量规模、异常汇总、隔离、泄漏及数量一致性结果 |
| manifest.json | 转换版本、生成时间、输入与输出哈希、记录数、规范和检查结果 |
| SHA256SUMS | 正式输出的逐文件 SHA-256，包含 manifest.json，排除自身 |

`data/processed` 是本地生成产物，不提交 Git。团队成员应使用同一转换脚本及官方 raw 数据复现生成；相同输入和代码生成的数据类 JSONL 内容保持确定性，`manifest.json` 的生成时间允许变化。

## 9. 测试

单元测试命令：

```text
python -B -m unittest discover -s tests -v
```

已有测试执行结果为 **21/21 PASS**。数据流水线包含原有 10 项 smoke 测试及新增 9 项 full 测试，另有 2 项核心测试。覆盖显式 train/val 来源、预测点展开、历史截断、异常留痕与排除、会话/ID 冲突失败、输出覆盖保护、数量一致性、SHA-256 和确定性。本次报告编写未重新运行测试。

## 10. SHA-256

输入哈希直接摘录自 `manifest.json.inputs`：

| 官方输入文件 | SHA-256 |
|---|---|
| data/raw/训练-验证-数据集/train/train_public.jsonl | `184a47471f8d6f70fa21b9265b0a4d4e51c3748ba25af39a801884a2da35341a` |
| data/raw/训练-验证-数据集/val/val_public.jsonl | `097c0639ef7370a18a54670ec3c8390ce5dc87c1ed95f436cef07b68dfed7668` |

以下输出哈希摘录自 `manifest.json.outputs` 和 `SHA256SUMS`，两者共同列出的 7 个文件哈希一致；`manifest.json` 自身哈希仅取自 `SHA256SUMS`。

| 正式输出文件 | 记录数 | SHA-256 |
|---|---:|---|
| classification_train.jsonl | 18653 | `4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5` |
| classification_val.jsonl | 2354 | `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201` |
| response_train.jsonl | 18653 | `a70ca036f2ad28265bbbd3ce6882006f81e7dd7abb39a50dbf0f16cb713e2ff6` |
| response_val.jsonl | 2354 | `ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2` |
| split_manifest.jsonl | 21011 | `0c4f96646fea6b5cdd1b38f43ee4555156d0f9aec88bc471e9bcc696c0ff8b2f` |
| anomalies.jsonl | 4 | `5c2f46bc835f21acc3c23e21e3df7088cedd0e065a9729b8b1268d2de967aee0` |
| leakage.json | 1 个对象 | `9c6167b8fadafa77da677ba30f6294f666d5c6fbcaa6b530e4193adac2e47c8d` |
| manifest.json | 1 个对象 | `4c247dda1e17b21bd2413f20e017e1dc7b87d17d13d8922ca74b169dd6a3c252` |

`SHA256SUMS` 不纳入自身哈希；本报告没有自行计算或编造任何哈希。

## 11. 结论

**B1 data v1 满足当前训练数据交付门槛，可以交 C 验收。** 正式训练仍应以相同脚本和数据 hash 为依据，C/D 使用有效训练样本数 **18653**、验证样本数 **2354**，并保留 4 个异常预测点的证据。
