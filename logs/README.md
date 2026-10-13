# 日志目录

本地完整日志不提交。需要审查的摘要指标和关键错误应写入 `reports/`；正式候选的完整训练日志可随对应 GitHub Release 保存。

例外：`.gitignore` 用 negate 规则显式放行 D 岗位 Day2 的审查证据日志（`logs/D_DAY2_*.txt`，即冒烟 stdout、epoch 循环回归运行与原模型 100 条对照 stdout）。这些文件小且是 PR 独立核验冒烟与测速的依据，随 PR 提交；其余日志仍按上文约定只在本地保留。模型权重不进入 Git，经 GitHub Release/LFS + SHA-256 分发。
