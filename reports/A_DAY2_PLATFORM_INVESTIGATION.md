# A Day2 平台与提交环境调查

日期：2026-10-13（Asia/Shanghai）  
状态：Integration Gate  
固定验收人：B

## 结论

A 已完成今天的平台调查。当前无需启动云端 GPU；明日先让 C/D 将训练初版准备到可直接运行，再开1卡 RTX 4090 做短跑和正式训练。调查不等于完成最终镜像验收，专属实例内的软件版本、显存占用和统一入口仍需模型初版到位后实测。

## 已确认事实

| 项目 | 结论 | 证据来源 |
|---|---|---|
| 截止时间 | 2026-10-20 23:59 | 队长登录赛事作品页面核对 |
| GPU 卡时 | 两张 RTX 4090 券，各20卡时，共40卡时 | 队长账户卡时券页面；生效时间 2026-10-13 16:02:53/55，失效时间 2026-10-20 16:02:53/55 |
| CPU 调试 | 无卡模式10小时 | 队长账户卡时券页面 |
| 免费存储 | 文件存储20GB/机房；用户镜像30GB/机房 | 平台计费说明；超额按日峰值计费 |
| 训练镜像候选 | `PyTorch 2.5.1 / Ubuntu 22.04 / Python 3.11 / CUDA 11.8`，页面显示18.89GB | 咪咕仝学公有镜像版本列表；仅用于训练候选 |
| 正式专属镜像 | `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8`，页面只有一个版本 | 创建实例页“大赛专用镜像-对话模型” |
| 提交流程 | 从大赛专属公有镜像创建实例，集成代码/权重后保存并上传参赛镜像，回作品页选择镜像提交 | 创建实例页明确提示“其他镜像无法上传参赛作品”及作品提交页操作手册 |
| 许可边界 | 允许开源模型和公开数据 | 队长根据赛事渠道确认；选用模型仍记录模型卡、revision、许可证 |
| 人员投入 | A/B/C/D 可按任务完成，不以固定小时数裁剪任务 | 队长确认；仍受代码依赖和机器墙钟时间约束 |

RTX 4090 的24GB显存只来自平台实例创建演示与常见规格，正式评测硬件仍未知。卡时券失效早于23:59截止，因此需要在10月20日16:02前结束依赖GPU的训练和最终检查。

## 仓库与数据核对

- A 分支工作前已 `git fetch origin` 并合并最新 `origin/main`；同步时包含已合并的 C PR #6 和 D PR #4，无未解决冲突。
- 官方 train/val 与两个 Schema 已整理到 `data/raw/训练-验证-数据集/`，四个 SHA-256 与 `official_materials/` 只读副本一致；原始数据继续由 `.gitignore` 排除。
- B PR #7 已提交但尚未合并：提供 B1 全量转换、泄漏检查及测试。报告记录有效 train 18,653、val 2,354，排除4条异常；先由 C 验收再合并。
- C PR #6 已合并：主候选 `hfl/chinese-macbert-base`，训练脚本与 MPS 冒烟已交付。当前只覆盖情感任务，依赖固定方式与 B1 输入、三维画像 head 尚需补齐。
- D PR #4 已合并：检索回复保底已全量验证；生成模型选型、LoRA/SFT 初版尚未交付。

## 明日交付门槛

明日各成员的具体任务写入 `docs/EXECUTION_PLAN.md` 的10/14行。开 GPU 前必须同时满足：B1 数据已合并；C/D 的脚本可解析B1输入并完成CPU/MPS dry-run；模型名称、revision、许可证和下载体积已记录；命令、输出目录和失败恢复方式明确。任何一项缺失都先在本地修复。

## B 验收步骤

```bash
git fetch origin
git switch feature/a-mlops
git pull --ff-only origin feature/a-mlops
git merge-base --is-ancestor origin/main HEAD
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/ci_smoke.py
git diff --check origin/main...HEAD
```

B 还需人工核对：本报告中的平台事实与队长提供的页面记录一致；计划没有把通用 PyTorch 镜像称为正式提交镜像；没有把40卡时误写成40小时双卡并行；10月20日GPU券失效时间与作品截止时间已经区分。通过后在 PR #5 提交 review；合并前保持 Integration Gate。
