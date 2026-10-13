# Day2 A 本地交付记录

日期：2026-10-13（Asia/Shanghai）。状态：Integration Gate；固定验收人 B。

已 fetch 并在 feature/a-mlops 同步 origin/main，基线 commit `22c4416`，祖先检查通过。未修改官方数据，explain.md 继续忽略。

交付：GitHub PR 模板、main 保护设置说明、Python 3.12 CI、实验记录模板、run_experiment.py 一条命令转换/推理/校验/评测及追溯记录。CI 不上传数据和生成结果；运行权限为 contents: read。

本地验收：

- `python -m unittest discover -s tests -v`：2/2 通过。
- `python -m compileall -q src scripts tests`：通过。
- `python scripts/ci_smoke.py`：合成数据完整实验两次通过，预测字节一致。
- Linux 断网开发容器 `bash -n /workspace/start.sh`：通过。
- 官方验证集前 100 个预测点：100/100 输出合法；情感 accuracy 0.11、画像维度 accuracy 0.2066667、字符 BLEU-4 0.0266666、字符 ROUGE-L F1 0.1375136，与已有保底记录一致。
- 完整实验记录位于忽略的 `results/day2_baseline_100/experiment.json`；本次工作区 dirty=true，不能宣称只由基线 commit 复现新增脚本。

本次 100 条实验命令（从仓库根目录）：

```powershell
python scripts/run_experiment.py --train "data/raw/训练-验证-数据集/train/train_public.jsonl" --val "../official_materials/训练-验证-数据集/val/val_public.jsonl" --limit 100 --output-dir results/day2_baseline_100
```

重跑须换新的输出目录。CI 在 Linux 额外执行两参数 Shell 入口，并验证保留已有结果目录文件；该新增检查等待远程 Actions 验证，本机 Windows 合成测试跳过此步骤。本次没有重新构建模型镜像。

用户已授权提交、推送并安装 GitHub CLI；交付提交到 A 分支，通过 base 为 main 的 PR 等待 B 复核。仓库文件只是规则与 CI 的交付，不能当作远程分支保护已生效。B 复核与合并前保持 Integration Gate；远程 Actions 结果以 PR 检查为准。
