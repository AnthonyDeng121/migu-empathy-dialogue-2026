# 数字人综合情感陪伴对话模型初赛工程

本目录是与官方原始资料分离的开发区。当前版本先提供确定可运行的端到端保底链路：数据审计、输入输出校验、字符级本地指标、检索式 baseline 和单元测试。官方文件不在这里改写。

## GitHub 中保存什么

仓库提交代码、配置、目录说明、测试、指标摘要和复现脚本。所有成员已经拥有的官方原始数据不重复上传；训练 checkpoint、模型权重、镜像、缓存和完整日志也不进入普通 Git 历史。

克隆仓库后，`data/raw/README.md`、`data/processed/README.md`、`weights/README.md` 等文件会保留完整目录结构和放置规则。成员把本机已有数据放到约定的相对路径即可。需要共享的冻结模型通过私有仓库 GitHub Release 或 Git LFS 分发，并用 SHA-256 校验。

## 放置官方数据

仓库里不包含官方数据和模型权重。大家把已经下载的官方数据放到：

```text
data/raw/
└─ 训练-验证-数据集/
   ├─ train/
   │  ├─ train_public.jsonl
   │  └─ train_public.schema.json
   └─ val/
      ├─ val_public.jsonl
      └─ val_public.schema.json
```

测试样例 `test_inference_data.jsonl` 放在 `data/raw/` 根目录。放置后可根据 `docs/OFFICIAL_MATERIALS_MANIFEST.md` 核对 SHA-256。

## 团队分支与工作规则

四位成员从最新的 `main` 分支分别建立自己的开发分支：

```bash
# A 只执行这一条
git switch -c feature/a-mlops

# B 只执行这一条
git switch -c feature/b-data-eval

# C 只执行这一条
git switch -c feature/c-emotion-profile

# D 只执行这一条
git switch -c feature/d-response-memory
```

每个人只执行属于自己的那一条命令，不要一次创建四个分支。创建前先运行 `git switch main` 和 `git pull --ff-only origin main`，保证新分支基于最新主分支。

工作规则：

- 不直接修改或推送 `main`。
- 每项工作通过 Pull Request 合并。
- 数据、模型权重、checkpoint、完整日志不提交普通 Git。
- 每天收工前在 `docs/EXECUTION_PLAN.md` 更新任务状态和证据。
- `Ready`：依赖满足，可以开始或正在做，不表示已经完成。
- `Blocked`：被明确问题阻塞，并写明阻塞原因、责任人、预计解除时间和降级方案。
- `Integration Gate`：个人工作完成，正在等待代码审查、接口测试或集成验收。
- `Done`：已经合并 `main` 且测试通过，证据已记录。
- 可以提前做后续任务，但不能绕过尚未满足的依赖或 `Integration Gate`。
- PR 至少由指定审查人检查；最终发布由 A 执行、B 独立复核并与 A 双签。

## 快速运行

从克隆后的仓库根目录执行：

```bash
python scripts/audit_dataset.py --train "data/raw/训练-验证-数据集/train/train_public.jsonl" --val "data/raw/训练-验证-数据集/val/val_public.jsonl" --output-json reports/data_audit.json --output-md docs/DATA_AUDIT.md
python scripts/run_baseline.py "data/raw/test_inference_data.jsonl" results/baseline --train "data/raw/训练-验证-数据集/train/train_public.jsonl"
python scripts/validate_io.py --input "data/raw/test_inference_data.jsonl" --submission results/baseline/submission.jsonl
python -m unittest discover -s tests -v
```

开发用离线镜像可从仓库根目录构建：

```bash
docker build -t emotion-dialogue-baseline:dev .
docker run --rm -v "$PWD/data/raw:/input:ro" -v "$PWD/results/docker:/output" emotion-dialogue-baseline:dev /input/test_inference_data.jsonl /output
```

该 Dockerfile 是可运行的标准库检索保底镜像，不是最终提交镜像。正式提交必须在官方基础镜像名称和 digest 确认后重新基于官方镜像构建，并把最终模型替换进官方 `participant/` 框架。

`evaluate_local.py` 的 BLEU-4/ROUGE-L 是明确标注的字符级诊断实现，不冒充官方实现。官方尚未在本地材料中给出分词、平滑、BERTScore 模型与三项权重。

