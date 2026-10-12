# 验收记录

验收日期：2026-10-12（Asia/Shanghai）

| 项目 | 结果 | 证据 |
|---|---|---|
| 官方资料只读归档 | PASS | `official_materials/` 共 14 个文件，均设置只读属性；清单含 SHA-256 |
| 全量数据审计 | PASS WITH WARNINGS | 训练 8,338 会话/18,657 预测点；验证 900 会话/2,354 预测点；发现 4 条训练引用异常与 18 个跨集重复回复 |
| Python 编译 | PASS | `python -m compileall -q team_project/src team_project/scripts team_project/tests` |
| 单元测试 | PASS | 2/2 通过 |
| 官方测试样例 baseline | PASS | 3/3 行完成，输出通过 ID、字段和枚举校验 |
| 验证集 100 条回归 | PASS | 情感 Acc 0.11，画像维度 Acc 0.2067，字符 BLEU-4 0.0267，字符 ROUGE-L F1 0.1375 |
| 所有 Python 脚本 `--help` | PASS | 6 个脚本均正常返回 |
| Dockerfile 构建 | PASS | Docker Desktop Linux engine 启动后构建成功；3 条样例及验证集前 100 条断网运行通过，详见下方 Docker 验收 |
| Shell 语法检查 | PASS | 在断网容器内执行 `bash -n /root/participant/start.sh` 通过 |

当前 Python 与开发 Docker 端到端链路均已通过。正式参赛镜像仍必须等待或取得官方基础镜像的准确名称和 digest；B 的独立复核尚待完成。

## Day1 本地链路复跑（2026-10-12）

- 使用当前 `main` 代码，直接读取 `../official_materials/` 的归档数据；本次未修改官方数据或源码。仓库 `data/raw/` 中尚未放入原始数据，因此没有采用 README 的默认数据路径。
- 全量训练数据构建检索索引后，对官方 3 条测试样例完成推理；进程退出码为 0。
- 索引构建耗时 56.694181 秒，3 条推理阶段耗时 0.828774 秒，seed 为 20261012。这里是本地 baseline 脚本计时，不是官方前 100 条性能测评；本次没有重跑验证集 100 条指标。
- 运行中观察到 Python CPU 时间持续增加、内存占用约 3.6 GB；此前两次运行因人为设置的 60 秒超时终止，不能据此判定程序异常。取消超时后的本次运行成功完成。
- 输入输出校验：`PASS input=3 submission=3`；单元测试 2/2 通过；`python -m compileall -q src scripts tests` 通过。
- 本地输出：`results/day1_local_rerun/submission.jsonl` 和 `results/day1_local_rerun/performance_report.json`，按 `.gitignore` 排除，不提交生成结果。
- `submission.jsonl` SHA-256：`4812193b102e1d2376fd1317975602648383a7519adc622124f73c01c100cd15`。

从 `team_project/` 目录复现：

```powershell
python scripts/run_baseline.py "../official_materials/test_inference_data.jsonl" results/day1_local_rerun --train "../official_materials/训练-验证-数据集/train/train_public.jsonl"
python scripts/validate_io.py --input "../official_materials/test_inference_data.jsonl" --submission results/day1_local_rerun/submission.jsonl
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

本次仅完成本地 Python 链路验收，Docker 验收和 B 的独立复核仍待完成，A 的任务状态继续保持 `Integration Gate`。

## Day1 Docker 断网验收（2026-10-12）

- 远程与本地已创建 `feature/a-mlops`，本地跟踪 `origin/feature/a-mlops`；后续 A 的变更先上传该分支，再经 PR 审查合并主分支。
- 已启动 Docker Desktop Linux engine（29.7.2），将归档训练数据和测试样例复制到 `data/raw/` 的约定路径。训练文件源/目标 SHA-256 均为 `184a47471f8d6f70fa21b9265b0a4d4e51c3748ba25af39a801884a2da35341a`，数据仍被 Git 忽略。
- Docker 基础镜像层下载长时间停滞；改由宿主机从 Docker 官方 registry 下载同一索引中的 Linux amd64 镜像，校验索引、manifest、config、压缩层及解压层 SHA-256 后导入本地缓存，再执行 `docker build --pull=false -t emotion-dialogue-baseline:dev .` 成功。未修改 Dockerfile 的基础镜像选择。
- 官方源索引：`python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`。导入归档后 Docker 会生成本地 manifest，因此本地 manifest digest 不等同于源索引 digest。
- `.dockerignore` 新增 `.codex_tmp/`，避免本地下载缓存进入镜像；容器内确认 `.codex_tmp/`、`.git/` 和 `explain.md` 均不存在，Shell 语法检查通过。
- 开发镜像 ID：`sha256:661129ba5a92e8d0aaa270a3de72509b70650f123636cf1f08077d5a240a3391`，大小 165,196,880 字节。这是本地开发镜像 ID，未发布镜像到远程 registry。

| 验收项 | 官方样例 | 验证集前 100 个预测点 |
|---|---|---|
| 容器 | `migu-day1-offline-sample` | `migu-day1-offline-val100` |
| 网络模式 / 退出码 | `none` / 0 | `none` / 0 |
| 输出校验 | 3/3 通过 | 100/100 通过 |
| 索引构建耗时 | 39.495561 秒 | 35.557385 秒 |
| 推理阶段耗时 | 0.778678 秒 | 18.879714 秒 |

- 两次均使用 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`，只传两个位置参数；训练数据从镜像内 `/opt/data/train.jsonl` 加载，输入只读挂载，输出目录单独挂载。
- 官方样例输出 SHA-256 与本地 Python 输出一致：`4812193b102e1d2376fd1317975602648383a7519adc622124f73c01c100cd15`；输出目录预置的 `keep.txt` 内容被保留。
- 100 条输出 SHA-256：`eafb9e67d34ab9eac88c44fcd3a35fdc7055e62be958485f7a135739a9f69dc4`。本地诊断指标与原有 100 条基线记录一致：情感 Acc 0.11、画像维度 Acc 0.2067、字符 BLEU-4 0.0267、字符 ROUGE-L F1 0.1375；完整摘要见 `reports/day1_docker_val100_metrics.json`。
- 耗时来自当前 baseline 的性能报告，不等同于官方完整性能报告口径；此次通过的是开发镜像验收，正式基础镜像兼容性及 B 的独立复核仍待完成。

PowerShell 复现断网运行（从仓库根目录执行，先按 README 放好数据并构建镜像）：

```powershell
$repoPath = (Get-Location).Path
New-Item -ItemType Directory -Force -Path results/docker_day1_sample | Out-Null
docker run --rm --network none --mount "type=bind,source=$repoPath\data\raw,target=/input,readonly" --mount "type=bind,source=$repoPath\results\docker_day1_sample,target=/output" --entrypoint bash emotion-dialogue-baseline:dev /root/participant/start.sh /input/test_inference_data.jsonl /output
python scripts/validate_io.py --input data/raw/test_inference_data.jsonl --submission results/docker_day1_sample/submission.jsonl
```

