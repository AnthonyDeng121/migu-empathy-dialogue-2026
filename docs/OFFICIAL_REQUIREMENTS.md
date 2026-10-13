# 官方要求核验报告

## 结论

当前本地附件足以确定输入输出 Schema、唯一运行入口、两种参考后端、输出文件名、100 条性能计时范围、16 类情感标签和三类画像标签。官网参赛细则已于 2026-10-12 下载快照并交叉核对，确认离线 Docker 自动测评、最多 5 次成功测评、运行失败不占成功次数、最后一次成功成绩覆盖之前成绩。

2026-10-13 已通过队长实际登录的赛事与咪咕仝学平台页面确认提交截止时刻、可用卡时、平台提交路径及数字人赛题专属公有镜像。正式评测 GPU、运行时限、参赛镜像大小上限、官方指标实现与权重、隐藏测试是否提供 memory bank 仍未公布，这些项目不得猜测。

## 已确认要求及来源

| 要求 | 结论 | 来源 |
|---|---|---|
| 初赛提交 | 自动化测评；下载基础镜像，在其中集成模型、权重、推理程序、配置和入口，离线运行 | `downloads_backup/AI技术赛道参赛细则.html`，三（一）数字人条目 ①-⑦ |
| 成功次数 | 每队最多 5 次测评成功；运行失败不占机会；以最后一次成功结果排名 | 同上，三（一）⑦、四（2） |
| 统一入口 | `bash /root/participant/start.sh TEST_FILE RESULT_DIR`，只传两个位置参数 | `official_materials/participant/README.md`，“唯一运行命令” |
| 输出文件 | `RESULT_DIR/submission.jsonl` 与 `performance_report.json` | 同上，“固定产物” |
| 性能范围 | 全量推理，但只统计前 100 条的端到端耗时；模型加载不计入参考脚本 timing scope | `participant/README.md` 与 `participant/run_inference.py` 的 `performance_rounds`/timing_scope |
| 输入 | 每行含 `id`、`conversation_id`、`target_user_turn_id`、`history`；history turn 含 `turn_id/role/content` | `test_inference_data.jsonl` 实际样例 |
| 输出 | 每行含 `id/response_text/emotion_label/user_profile/memory_refs`，禁止额外字段 | `participant/templates/submission_schema.json` |
| 情感 | 16 类：joy、gratitude、relaxed、care、pride、neutral、surprise、mixed、sadness、loneliness、anxiety、anger、fear、disgust、shame、helplessness | 训练/验证 Schema 的 `$defs.emotion` |
| 画像 | personality_traits、interests、style 三个离散多标签数组；各自枚举见 Schema | 训练/验证 Schema 的 `$defs.profile` |
| 记忆引用 | `memory_refs` 是唯一字符串数组，元素形如 `mem_000001` | 训练/验证 Schema 与提交 Schema |
| 专家材料 | 技术方案、实验设计、报告、训练日志、调优过程打包到镜像 `/root/AI技术赛道-数字人综合情感对话模型-队名` | `official_materials/README.md`，“内部技术专家评审所需资料” |
| 平台示例 | 演示截图为 2×RTX 4090 24GB、32 CPU、128GB，公有镜像显示 `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8` | `咪咕仝学平台实例创建demo V1.docx` 第 4 步截图；仅为创建示例，不是评测资源承诺 |
| 截止时间 | 初赛作品提交截止为 2026-10-20 23:59；GPU 券提前于当日约 16:02 失效 | 2026-10-13 队长登录赛事提交页与卡时券页核对 |
| 平台提交路径 | 选择咪咕仝学平台来源时，只有从“大赛专属公有镜像”创建的实例才能保存并上传参赛镜像；上传后回到作品页选择镜像提交 | 2026-10-13 平台创建实例与作品提交页面提示 |
| 数字人专属镜像 | `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8`，框架 vLLM 0.19.0、Python 3.12、CUDA 12.8；页面仅显示一个版本 | 2026-10-13 咪咕仝学“大赛专用镜像-对话模型”选择页 |
| 训练资源 | 两张 RTX 4090 券各 20 卡时，共 40 卡时；另有 10 小时无卡模式；30GB 用户镜像免费额度、20GB 文件存储免费额度 | 2026-10-13 队长账户卡时券页与平台计费说明 |
| 数据与模型许可 | 允许开源模型和公开数据；具体模型仍需保存模型卡、revision 与许可证证据 | 2026-10-13 队长根据赛事渠道确认；本地附件未给出更细限制 |

## 输入 Schema

```json
{"id":"string","conversation_id":"string","target_user_turn_id":1,"history":[{"turn_id":1,"role":"user|assistant","content":"non-empty string"}]}
```

`history` 必须按 turn_id 递增，并以 `target_user_turn_id` 对应的 user 消息结尾。本地样例没有 user_id、显式画像或 memory bank。

## 输出 Schema

```json
{"id":"与输入完全相同","response_text":"非空字符串","emotion_label":"16 类之一","user_profile":{"personality_traits":[],"interests":[],"style":[]},"memory_refs":[]}
```

本地自检以输出 ID 和输入 ID 逐条相等为主，而不使用附件 Schema 的 ID 正则硬拒绝合法样例。

## 材料冲突与数据异常

1. 提交 Schema 规定 ID 正则 `^test_[0-9]{6}_t[0-9]+$`，但官方测试样例使用 `test1_000001_t5` 等 ID。应原样复制输入 ID，并向官方确认最终 ID 模式。
2. 训练数据中 4 个 prediction point 的 `assistant_response_turn_id` 指向目标 user turn 的前一条 assistant：`train_conv_000628` (15→14)、`train_conv_000817` (33→32)、`train_conv_003644` (7→6)、`train_conv_003695` (9→8)。目标对象中的 response_text 仍与所指 turn 一致。训练转换时应按 target 对象监督，并单独标记这 4 条，不宜静默假定 assistant_id=user_id+1。
3. 官网当前参赛细则没有列出 70/30 评分明细、BLEU/ROUGE/BERTScore 具体实现或推理延迟计算公式；这些说法只存在于接管提示的“已知事实”，在本地附件与当前网页快照中未获独立佐证。
4. `participant/requirements.txt` 使用下限约束 `torch>=2.1`、`transformers>=4.40`、`accelerate>=0.29`，不满足可复现所需的精确锁定；参赛镜像应在选型后冻结完整版本和 wheel/hash 清单。

## 运行入口和目录约定

- 官方路径必须保留 `/root/participant/start.sh`，不能用自定义目录替代。
- `RESULT_DIR` 可以预先存在；官方脚本只覆盖两项固定产物，不应删除其中其他文件。
- Transformers 模式要求本地模型目录且 `local_files_only=True`；vLLM 模式启动本机 HTTP 服务，默认端口 8000，离线环境变量开启。
- 当前 `start.sh` 固定 `source activate conda_env` 且 vLLM 默认 `--tensor-parallel-size 2`；需在基础镜像内验证，不能在未知单卡环境盲用。

## 依赖与资源限制

- 已知参考依赖下限：torch 2.1+、transformers 4.40+、accelerate 0.29+。
- 正式参赛实例固定从平台 `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8` 大赛专属公有镜像创建。该名称不等于已确认镜像 digest；首次创建后仍需记录镜像内实际 PyTorch、Transformers、操作系统与驱动版本。
- 训练可使用通用 `PyTorch 2.5.1 / Ubuntu 22.04 / Python 3.11 / CUDA 11.8` 公有镜像；它不是最终参赛基础镜像。团队决定在 C/D 初版脚本、数据加载和依赖清单完成前不开 GPU 实例。
- 40 卡时按“运行小时×GPU 数量”扣除，券在 2026-10-20 约 16:02 失效；最终 GPU 验证不能排到 23:59 截止前才执行。
- 示例平台：2×RTX 4090 24GB、32 核、128GB；不代表官方评测机。
- 未知：正式评测 GPU/CPU/内存、参赛镜像大小上限、总时限、单条或批量延迟口径、专属镜像 digest 及镜像内 PyTorch/Transformers 精确版本。

## 提交检查表

- [ ] 从官方基础镜像构建，记录镜像 digest。
- [ ] `/root/participant/start.sh` 可执行且仅需两个位置参数。
- [ ] 全断网运行样例与至少 100 条回归数据。
- [ ] 输出行数、顺序和 ID 与输入完全一致；每行通过字段枚举校验。
- [ ] 模型、tokenizer、配置、依赖、权重全部在镜像内，无运行时下载。
- [ ] 性能报告完成 100 条计时，无解析失败。
- [ ] 专家评审材料和日志位于规定的 `/root/AI技术赛道-数字人综合情感对话模型-队名`。
- [ ] 镜像安全扫描通过，无账号、密码、API Key、第三方无关标识和违规内容。
- [ ] 保存 Git commit、配置、随机种子、模型哈希、镜像 digest 和提交时间。

## 必须向官方确认的问题

1. 最终测试 ID 格式是否接受 `test1_...`，提交 Schema 正则是否将更新？
2. 数字人专属镜像的 digest、镜像内 PyTorch/Transformers 精确版本及参赛镜像大小上限？
3. 正式评测 GPU 型号/数量/显存、CPU、内存、镜像大小和运行时限？
4. BLEU-4 的分词与平滑、ROUGE-L 实现、BERTScore 模型/层/基线、三项权重和总分归一化方式？
5. 延迟按前 100 条之和、均值还是其他统计量；是否包含服务/模型启动和 I/O？
6. 隐藏测试是否提供 memory bank；若提供，字段、目录和引用规则是什么？
7. 已确认允许开源模型与公开数据；仍需确认合成数据、训练期外部 API 及许可证/署名的细则？
8. 技术方案、实验报告和日志的文件格式、命名、页数或模板要求？

