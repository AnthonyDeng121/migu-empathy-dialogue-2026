# 日志目录

本地完整日志不提交。需要审查的摘要指标和关键错误应写入 `reports/`；正式候选的完整训练日志可随对应 GitHub Release 保存。

例外：`.gitignore` 用 negate 规则显式放行 D 岗位各轮的审查证据日志（stdout 摘要、冒烟/训练/推理运行记录）。这些文件小且是 PR 独立核验冒烟、测速与全量推理的依据，随 PR 提交；其余日志仍按上文约定只在本地保留。当前放行清单（与 `.gitignore` 末尾 negate 规则一一对应，新增放行时同步本清单）：

- Day2（冒烟/epoch 回归/原模型 100 条对照）：`D_DAY2_SFT_SMOKE_0P5B.txt`、`D_DAY2_SFT_SMOKE_1P5B.txt`、`D_DAY2_EPOCH_LOOP_CPU.txt`、`D_DAY2_ORIGINAL_0P5B_VAL100.txt`、`D_DAY2_ORIGINAL_1P5B_VAL100.txt`
- Day3（冒烟/全量训练/全量推理/条件与 minlen 臂/条件构建）：`D_DAY3_SFT_SMOKE_1P5B.txt`、`D_DAY3_SFT_FULL_1P5B.txt`、`D_DAY3_INFERENCE_1P5B_VAL.txt`、`original_1p5b_val_full.txt`、`d2_1p5b_conditioned_val_full.txt`、`d2_1p5b_minlen_val_full.txt`、`build_conditions.log`
- Day4（2-epoch 候选）：`D_DAY4_SFT_FULL_2EP.txt`、`d4_2ep_val_full.txt`
- Day6（10/18 G5 冻结日任务提前执行，v2 权重核对/冷启动门禁）：`D_DAY6_ARTIFACT_VERIFY.txt`

另有 C 的 `C_DAY2_EMOTION_SMOKE.txt` 在跟踪中（早于 negate 机制的提交）。模型权重不进入 Git，经 GitHub Release/LFS + SHA-256 分发。
