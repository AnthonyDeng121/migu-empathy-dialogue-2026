# 实验记录模板

- 实验 ID / 日期 / 负责人：
- 假设和相对基线唯一主要变化：
- 代码 commit / 工作区是否干净：
- 训练与验证文件 SHA-256 / conversation 分组与泄漏审计证据：
- 配置路径及 SHA-256 / seed：
- Python 与依赖版本 / 硬件：
- 复现命令：
- 样本数 / 情感 accuracy / 画像维度 accuracy / 字符 BLEU-4 / 字符 ROUGE-L：
- 构建或加载耗时 / 推理耗时 / 内存或显存 / 模型大小：
- 输出和权重 SHA-256 / 对应 Release（有权重时）：
- 失败、局限、相对基线结论及采用决定：
- 验收人 / review / 状态：Integration Gate

字符指标仅用于本地诊断；没有官方实现时不得称作官方分数。检索 baseline 的 seed 被记录但算法本身不使用随机采样；配置中 max_history_chars/default_emotion 当前没有被检索实现使用。

从仓库根目录一条命令重跑（每次使用新的输出目录）：

```bash
python scripts/run_experiment.py --train "data/raw/训练-验证-数据集/train/train_public.jsonl" --val "data/raw/训练-验证-数据集/val/val_public.jsonl" --config configs/baseline.json --limit 100 --output-dir results/day2_baseline_100
```

移除 `--limit 100` 即全量验证。脚本依次转换、推理、校验和评测，失败返回非零退出码并写 failed 记录；输出 experiment.json 包含 commit、dirty、源哈希、配置、seed、命令、环境、指标及产物哈希。完整结果留在忽略的 results/，只提交脱敏小型摘要。性能报告的耗时不参与两次结果一致性比较。
