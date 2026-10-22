# B 岗 Day 3：D2 1.5B val 独立复核

复核基线为当前仓库 `main` 最新提交 `69c1d7c`。本复核只读取冻结 B1 val、官方 raw val、D2 生成结果和已有报告；未训练模型、未使用 GPU、未修改 `data/raw`。

## 结论

**PASS。** `results/response_gen/d2_1p5b_val_full.jsonl` 恰好覆盖 B1 val 的 2,354 个 prediction point：无缺失、无重复，顺序也与 `data/processed/b1_full_v1/response_val.jsonl` 一致。独立重算得到字符级 BLEU-4 **0.1074682476**、ROUGE-L F1 **0.2541841482**，四舍五入到 4 位分别为 **0.1075**、**0.2542**，与 D 报告一致。

## 覆盖、ID 与 schema

- 生成结果、B1 `response_val.jsonl`、官方 raw val prediction points 均为 2,354。
- 生成结果每行字段恰为 `{"id", "response_text"}`，符合 D 推理脚本的中间生成输出契约。
- `id` 全部为非空字符串；重复数 0；空 ID 数 0。
- 生成结果 ID 与 B1 `sample_id` 集合相同且逐行同序；缺失 0，额外 0。
- `response_text` 全部为字符串，空回复 0；额外字段 0。
- 该文件是中间生成文件，不是官方最终 submission 格式；最终 submission 所需分类与 memory 字段不在此文件中。本次未将其误判为最终提交文件。

## 独立指标重算

金标按官方 raw val 的 `(conversation_id, target_user_turn_id)` 对齐，预测按 B1 `sample_id` 对应的同一 prediction point 对齐。BLEU-4 与 ROUGE-L 直接调用 `src.evaluation.metrics`。

| 指标 | B 独立重算 | D 报告 | 结果 |
|---|---:|---:|---|
| 字符 BLEU-4 | 0.1074682476 | 0.1075 | 一致（4 位四舍五入） |
| 字符 ROUGE-L F1 | 0.2541841482 | 0.2542 | 一致（4 位四舍五入） |
| 空回复 | 0 | 0 | 一致 |
| 生成长度（非空白字符，均值） | 131.2247 | — | 复核统计 |
| 生成长度（Python 字符，均值） | 135.8853 | 135.9 | 一致（1 位四舍五入） |

B1 `response_val.jsonl` 中的 `response_text` 与 raw val 每个目标金标逐条相同（2,354/2,354），ID 对齐链路确认无误。

## 推理耗时与输出统计

以下字段来自 `results/response_gen/d2_1p5b_val_full_report.json`，并对可推导项重新计算：

- 总耗时：8,844.469 s（约 2.46 h）。
- mean：3.7572 s/sample；由总耗时/2,354 重算为 3.7572086 s/sample。
- p50：2.5452 s/sample。
- 生成 token：197,717；tokens/s：22.35；重算为 22.3548751 tok/s。
- 空输出：0。
- report 标记 `device=cuda`、`dtype=bfloat16`；本次复核未启动推理，也未使用 GPU。

## 冻结数据与泄漏检查

- 推理输入为 `data/processed/b1_full_v1/response_val.jsonl`，`source_split` 全部为 `val`。
- B1 response val SHA-256：`ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2`，与 `SHA256SUMS` 一致。
- 2,354 行均满足上下文 turn 递增、最后一 turn 为目标 user、无 turn 超过目标 user turn；泄漏违规数 0。
- 原始 val SHA-256：`097c0639ef7370a18a54670ec3c8390ce5dc87c1ed95f436cef07b68dfed7668`，与 B1 manifest/leakage 记录一致。
- 指标金标只来自官方 raw **val**，未读取或引用训练集金标。B1 记录 train/val conversation overlap=0、`target_assistant_in_context=false`。

## 交叉核对与问题

D 报告中的样本数、覆盖率、空回复、BLEU-4、ROUGE-L、均值长度和性能报告耗时字段均通过本次复核。未发现数值或数据口径不一致。

## 验证命令

仓库未安装 pytest（`No module named pytest`），因此使用标准库 unittest：

```text
python -m unittest discover -s tests -p 'test_run_response_inference.py' -v
python -m unittest discover -s tests -p 'test_data_pipeline.py' -v
python -m unittest discover -s tests -p 'test_response_sft.py' -v
python -m unittest discover -s tests -p 'test_core.py' -v
```

结果分别为 **4/4、22/22、39/39、2/2 PASS**。
