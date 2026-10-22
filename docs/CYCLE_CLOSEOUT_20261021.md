# 第一周期收尾与下一周期输入

日期：2026-10-21。原 10/13–10/20 排程保留为历史记录，本文件描述收尾基线；下一周期尚未启动训练或调参。

## 合并范围

| PR | 收尾内容 |
|---|---|
| [#19](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/19) | 提交 JSONL 校验与候选逐条比较工具 |
| [#21](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/21) | D5 候选材料清单和训练/推理指标交叉引用 |
| [#23](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/23) | D6 权重核对状态；验收附件按协作约定归档本地 |
| [#22](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/22) | 最新 C/D 集成；官方配置、兼容入口、Python 路径和源码/镜像打包修复 |
| [#24](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/24) | G6 材料复核历史记录和归档说明 |
| [#25](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/25) | G7 历史待命记录及本周期收尾 |

本次集中整理和合并由用户明确授权。原 PR 中的旧 main、未合并状态、旧 SSH 路径和排程日期是当时快照，不覆盖本文件中的当前基线。

## 固定基线

- C：`c-day5-emotion-longtail075-20261016`，sequence 情感分类；模型 SHA-256 `aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37`。
- D：`d-day4-response-sft-2ep-20261017`，两轮微调 adapter；权重 SHA-256 `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e`。
- 基座：Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 配置：`condition_response=false`、`profile_mode=empty`、`max_new_tokens=256`；记忆引用为空。
- 官方入口：`bash /root/participant/start.sh TEST_FILE RESULT_DIR`。部署保留实际 participant 目录、所选 Python 环境、官方及集成配置、兼容推理入口。
- 用户已反馈官方测评成功；官方镜像 ID、评测编号、分项分数和平台截图尚未提供，不在仓库中填造。

## 验证与证据边界

本轮执行代码编译、标准库单测、合成数据 CI、打包入口回归和差异检查。单测在 Linux CI 中额外覆盖暂存目录的 Python 路径固化。此前实例内完成两次各 3 条样例的断网启动，原官方启动脚本可运行且输出一致。

本轮没有重跑训练或全量 GPU 验收。历史 2,354 条评测、画像对照和生成长度负结果保留其原模型、配置与数据范围，不冒充本次代码版本或新假设的验证结果。

验收/复核附件保存于本地 `.codex_tmp/acceptance/legacy-pr23/`、`.codex_tmp/acceptance/legacy-pr24/`；这些相对路径不是公开下载链接。Git 中继续保留实验指标、技术方案与比赛材料清单。Day1 的 100 条诊断指标仍归实验指标，予以保留。

## 留给下一周期的输入

1. D 主做、C 配合：最新版 C/D 条件关闭与仅注入预测情感的对照；B 独立复核，A 决定集成。当前只找到旧组合报告，不宣称新版对照已完成。
2. C：针对 mixed、care 的错误分析与融合/校准候选；阈值选取使用训练内留出，保留未参与调参的验证集。
3. D：比较训练与推理条件一致的无条件 SFT，以及长回复内容针对性改进。
4. B：结合实际官方评分口径确定候选比较指标、切片和采用门槛。
5. 画像列为下一周期 P1 独立专项，由 C 主做，安排诊断、首版训练、针对性改进及集成验证；记忆、较大基座与推理优化按资源和证据排程。本轮不改变成功版本的能力声明。

下一周期另建计划，明确负责人、输入、预算、可复现命令及采用门槛。

用户已确认第二周期计划，见 [EXECUTION_PLAN_CYCLE2.md](EXECUTION_PLAN_CYCLE2.md)：两天一项任务，共七项，排程为 2026-10-22 至 2026-11-04；具体任务尚未启动。
