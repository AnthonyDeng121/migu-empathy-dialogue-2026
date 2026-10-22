# B2 统一候选评测汇总

基线为冻结 B1 validation 的 **2,354 prediction points**。本次未训练、未启动 GPU、未修改 `data/raw`，也未修改 C/D 原始报告或预测文件。所有 response 指标按 `src.evaluation.metrics` 重算；分类多数类按 raw train 拟合并在 raw val 重算。

## 结论

当前可正式比较的 full-set 候选：

- 分类：多数类 baseline；C 的 MacBERT emotion encoder full-val 报告结果（但没有预测 JSONL，无法逐条 replay）。
- 回复：retrieval full-val 报告结果；D2 1.5B LoRA/SFT full-val 结果。

当前推荐：分类使用 C 的 full-val emotion encoder（Accuracy 0.67842、Macro-F1 0.4032，报告级结果）；回复使用 D2 1.5B LoRA/SFT（独立重算 BLEU-4 0.107468、ROUGE-L 0.254184）。

不可公平比较：C Day3 emotion/profile 仅 32 条 val dry-run；D 原始 1.5B、0.5B 仅各 100 条；可见 retrieval submission 文件只有 3 行，full-set retrieval 仅有既有 metrics 报告、缺少可 replay 的 full prediction 文件。

## 分类候选

| 候选 | 覆盖 | Accuracy | Macro-F1 | profile 三维 exact-match | 状态 |
|---|---:|---:|---:|---|---|
| Majority baseline | 2354/2354 | 0.319031 | — | personality 0.618522；interests 0.748938；style 0.517417；macro 0.628292 | FULL SET，独立重算 |
| C emotion encoder | 2354/2354 | 0.678420 | 0.403200 | 未提供 profile 预测 | FULL SET，报告级；无预测 JSONL |
| C Day3 emotion/profile dry-run | 32/2354 | 0.312500 | 0.029762 | 三维 exact-match 均 0；macro 0 | **NOT FULL SET** |

C emotion encoder 报告中的 per-class recall：anger 0.704348、anxiety 0.893475、care 0、disgust 0、fear 0.620968、gratitude 0.762712、helplessness 0.669014、joy 0、loneliness 0.695652、mixed 0.045198、neutral 0.113208、pride 0、relaxed 0.892421、sadness 0.358779、shame 0.652542、surprise 0。少数类/失败类主要是 care、disgust、joy、pride、surprise（recall=0），mixed 0.0452、neutral 0.1132 也明显偏低。上述 C 指标来自已提交 full-val report；仓库没有对应预测 JSONL，因此不能声称本次逐条独立 replay。

## 回复候选

| 候选 | 覆盖 | BLEU-4 | ROUGE-L F1 | 空输出 | 耗时 |
|---|---:|---:|---:|---:|---|
| Retrieval baseline | 2354/2354（报告） | 0.029367 | 0.143870 | 未在可 replay 文件中重算 | 报告未提供 full-set 推理耗时；可见 performance 文件仅 3 行 |
| Original Qwen 1.5B | 100/2354 | 0.0377（100 条切片报告） | 0.1565 | 0 | 241.356s；2.4136s/sample；p50 1.9328；35.7 tok/s |
| Original Qwen 0.5B | 100/2354 | 0.0259（100 条切片报告） | 0.1377 | 0 | 147.157s；1.4716s/sample；p50 1.183；43.11 tok/s |
| D2 1.5B LoRA/SFT | 2354/2354 | **0.107468** | **0.254184** | **0** | 8844.469s；3.7572s/sample；p50 2.5452；22.35 tok/s |

原模型两路均标记 **NOT FULL SET**，不能与 D2 full-set 结果同级排名。D2 低分切片抽样（来自实际 full-val 文件）包括：`pp:["val_conv_000180",21,3]`（ROUGE-L 0.00538）、`pp:["val_conv_000180",15,2]`（0.00692）、`pp:["val_conv_000322",7,1]`（0.00935）；这些回复分别为极短“我听着。”、“有。”等，属于明显异常/严重欠生成案例。

## 数据与缺口

B1 response val SHA-256 为 `ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2`；raw val SHA-256 为 `097c0639ef7370a18a54670ec3c8390ce5dc87c1ed95f436cef07b68dfed7668`。指标金标只取 raw val，未使用 train 金标。

尚缺：C full-val 分类预测 JSONL（含 emotion/profile）；D retrieval full prediction JSONL；D original Qwen 1.5B full-val 输出（0.5B 若需同集比较也应补齐）。建议使用 GPU 补跑 C full-val 推理及 D 原模型 full-val 推理；retrieval 可 CPU 补跑并保存完整预测文件。补跑后应以同一 B1 val 和本报告的重算逻辑更新比较。

相关 unittest：`test_run_response_inference` 4/4、`test_data_pipeline` 22/22、`test_response_sft` 39/39、`test_core` 2/2，合计 **67/67 PASS**。pytest 未安装，使用标准库 unittest。
