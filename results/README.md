# 本地结果目录

推理输出和临时实验结果默认不提交。经过整理的小型指标摘要放入 `reports/`，并在 PR 中链接对应实验配置。

例外：`results/response_gen/` 下 D 岗位各轮的审查证据以 negate 规则显式放行，作为对应 PR 的独立核验依据随 PR 提交；其余推理输出仍只在本地保留。当前放行清单（与 `.gitignore` 末尾 negate 规则一一对应，新增放行时同步本清单）：

- Day2 三路对比（原模型 100 条对照）：`original_0p5b_val100.jsonl`、`original_1p5b_val100.jsonl` 及各自 `_report.json`、`three_way_val100.log`
- Day3（原模型全量臂/条件臂/minlen 臂/条件输入）：`original_1p5b_val_full.jsonl(+_report.json)`、`d2_1p5b_val_full_conditioned.jsonl(+_report.json)`、`c2_conditions_val_full.jsonl`、`d2_1p5b_val_full_minlen.jsonl(+_report.json)`
- Day4（cap512 冒烟臂/IO 兼容提交样/2-epoch 候选全量臂）：`day4_smoke_input.jsonl`、`d2_1p5b_val100_cap512.jsonl(+_report.json)`、`d2_val_full_as_submission.jsonl`、`d4_2ep_val_full.jsonl(+_report.json)`

另有历史 force-add 跟踪、无 negate 规则的文件：`artifact_verify_day3/`（D3 门禁三轮冷启动证据）、`d2_1p5b_val_full.jsonl(+_report.json)`（D2 首轮全量预测，B2 join 基准）。全量逐行对比数据（如 `reports/d_day4_2ep_comparison.json`，1.3MB）只在仓库保留，不进专家材料包（包内放摘要，见 `reports/D_DAY5_MATERIALS_CHECK.md`）。
