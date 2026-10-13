# 检索回复 baseline 验证记录（10/12，D 岗位）

## 结论

- 官方数据已归档 `data/raw/`，SHA-256 与 `docs/OFFICIAL_MATERIALS_MANIFEST.md` 核对一致（train `184a4747…`、val `097c0639…`、test `194f824a…`）。
- 全量端到端跑通：train 8338 会话建 n-gram 检索库（约 25s），val 2354 个 prediction points 推理（约 346s），`validate_io.py` 校验 2354/2354 通过。
- 全量产物在 `results/baseline/`（被 .gitignore 排除，不入 Git）；指标在 `reports/baseline_val_full_metrics.json`（入 Git）。
- 全量单测 18/18 通过（`tests/test_retrieval.py` 16 项锁定检索契约：四项字段、`memory_refs=[]`、tie 取初筛排序靠前者（票数高优先，票数并列取训练顺序靠前）、空库抛 `ValueError`；另含 `tests/test_core.py` 2 项）。

## 本地诊断指标（val 全量 2354 点）

| 指标 | 值 |
|---|---|
| emotion_accuracy | 0.1559 |
| profile_dimension_accuracy | 0.4506 |
| bleu4_char（诊断） | 0.0294 |
| rouge_l_char_f1（诊断） | 0.1439 |

官方分词、平滑、BERTScore 模型与指标权重未公布，以上仅同环境相对比较，不冒充官方分数。

## 复现命令

```bash
# 1. val 转官方 inference 输入格式
python scripts/make_eval_input.py "data/raw/训练-验证-数据集/val/val_public.jsonl" data/processed/val_eval_input.jsonl

# 2. train 建检索库并对 val 全量推理
python scripts/run_baseline.py data/processed/val_eval_input.jsonl results/baseline --train "data/raw/训练-验证-数据集/train/train_public.jsonl"

# 3. 输出合法性校验
python scripts/validate_io.py --input data/processed/val_eval_input.jsonl --submission results/baseline/submission.jsonl

# 4. 本地诊断指标
python scripts/evaluate_local.py --gold-dataset "data/raw/训练-验证-数据集/val/val_public.jsonl" --input data/processed/val_eval_input.jsonl --submission results/baseline/submission.jsonl --output reports/baseline_val_full_metrics.json
```

## 已知契约（由 tests/test_retrieval.py 锁定）

- `history_until` 拼 `role:content` 行，查询与候选至少共享 `"user:"` 前缀的 3-gram，votes 极少为空。
- 候选池内 cosine 最高者胜出；并列时取初筛排序靠前的候选（票数高者优先，票数并列时取训练顺序靠前者）。
- 空检索库 `predict` 抛 `ValueError`（不静默返回）。
- `memory_refs` 恒为 `[]`；官方确认提供 memory bank 后再实现本地检索。
