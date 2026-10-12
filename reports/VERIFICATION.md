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
| Dockerfile 构建 | BLOCKED BY LOCAL ENVIRONMENT | Docker CLI 29.7.2 可用，但 Docker Desktop Linux daemon 未运行，无法连接 `dockerDesktopLinuxEngine`；不是 Dockerfile 构建错误 |
| Shell 语法检查 | BLOCKED BY LOCAL ENVIRONMENT | 系统 `bash.exe` 指向未运行的 WSL 环境，无法执行 `bash -n` |

当前可直接运行 Python 端到端链路。Docker 构建需先启动 Docker Desktop Linux engine，再执行 README 中的构建与运行命令；正式参赛镜像仍必须等待或取得官方基础镜像的准确名称和 digest。

