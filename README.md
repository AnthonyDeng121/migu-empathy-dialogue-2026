# 数字人综合情感陪伴对话模型初赛工程

本目录是与官方原始资料分离的开发区。当前版本先提供确定可运行的端到端保底链路：数据审计、输入输出校验、字符级本地指标、检索式 baseline 和单元测试。官方文件不在这里改写。

## GitHub 中保存什么

仓库提交代码、配置、目录说明、测试、指标摘要和复现脚本。所有成员已经拥有的官方原始数据不重复上传；训练 checkpoint、模型权重、镜像、缓存和完整日志也不进入普通 Git 历史。

克隆仓库后，`data/raw/README.md`、`data/processed/README.md`、`weights/README.md` 等文件会保留完整目录结构和放置规则。成员把本机已有数据放到约定的相对路径即可。需要共享的冻结模型通过私有仓库 GitHub Release 或 Git LFS 分发，并用 SHA-256 校验。

## 快速运行

从仓库根目录执行：

```bash
python team_project/scripts/audit_dataset.py --train "official_materials/训练-验证-数据集/train/train_public.jsonl" --val "official_materials/训练-验证-数据集/val/val_public.jsonl" --output-json team_project/reports/data_audit.json --output-md team_project/docs/DATA_AUDIT.md
python team_project/scripts/run_baseline.py "official_materials/test_inference_data.jsonl" team_project/results/baseline --train "official_materials/训练-验证-数据集/train/train_public.jsonl"
python team_project/scripts/validate_io.py --input "official_materials/test_inference_data.jsonl" --submission team_project/results/baseline/submission.jsonl
python -m unittest discover -s team_project/tests -v
```

开发用离线镜像可从仓库根目录构建：

```bash
docker build -f team_project/Dockerfile -t emotion-dialogue-baseline:dev .
docker run --rm -v "$PWD/official_materials:/input:ro" -v "$PWD/team_project/results/docker:/output" emotion-dialogue-baseline:dev /input/test_inference_data.jsonl /output
```

该 Dockerfile 是可运行的标准库检索保底镜像，不是最终提交镜像。正式提交必须在官方基础镜像名称和 digest 确认后重新基于官方镜像构建，并把最终模型替换进官方 `participant/` 框架。

`evaluate_local.py` 的 BLEU-4/ROUGE-L 是明确标注的字符级诊断实现，不冒充官方实现。官方尚未在本地材料中给出分词、平滑、BERTScore 模型与三项权重。

