# D 岗位 CYCLE2 T3：无条件两轮 SFT 与 D4 v2 同设备两臂比较

日期：2026-10-27（Asia/Shanghai）。分支：`feature/d-day7-deadline-standby`。状态：T3 训练 + 同设备两臂比较完成（本报告随 PR 交 A 审）。

对应执行计划 T3（D 列）：「跑无条件两轮 SFT 全量训练（不注入条件），与 D4 v2 同设备同解码比较，检验金标条件训练与部署侧不注入条件的失配代价」。

## 结论摘要

- **160 条筛选集上未观察到无条件训练的优势**：同一张 RTX 4090、同基座、同解码（bf16、greedy、max_new_tokens=256）顺序生成，T3 臂 BLEU-4 0.1034 / ROUGE-L 0.2502，D4v2 臂 BLEU-4 0.1052 / ROUGE-L 0.2526，差 −0.0018 / −0.0024；逐条 65 胜 / 65 负 / 30 平，26/160 文本完全相同。差异方向混杂且量级很小，**在本诊断集规模上不足以证明两臂质量等价，也不证明任一臂更优**——结论限定为「未观察到无条件训练的收益」。
- **按计划维持 condition 关闭，暂保留 D4 v2，不据本轮结果改换候选**：按「无信号不扩大候选」的周期纪律，不做「权重二选一」动作；T3 权重与全套证据留档（无条件路线的正统配置），是否改换候选留待 T6 完整评测。
- T3 臂平均长 3.5 字（129.05 vs 125.57），长回复桶（>200 字）34/160 vs 32/160——长度特性基本持平，**T4 长回复改进仍是唯一 open 项**（见第四节）。
- **无人值守流程仅部分验证**：训练一次跑完（4664 步无中断），周期 checkpoint（18 段）与 cron 守护就位但**本次未触发恢复**——「没中断」不能证明故障自愈能力。恢复正确性的证据是随仓库交付的可复现故障注入测试（`tests/test_train_response_sft.py` 的 tiny 模型 e2e：模拟崩溃→从 checkpoint 恢复→最终权重与未中断运行**逐字节一致**，主会话实跑通过）与守护脚本（`scripts/remote/`，见第五节）。
- **按 A 审查（P1）补充**：checkpoint 现附带实验身份 manifest（训练数据含顺序、条件文件、配置、基座/tokenizer 内容指纹、有效训练预算），恢复前逐项比对，任何不一致拒绝续跑；反向用例（换 seed / 换数据顺序 / 换预算 / 换基座 / 缺 manifest）与轨迹一致性均有测试覆盖。

## 一、实验设置（固定变量清单）

| 项 | 值 |
|---|---|
| 训练宿主 | 内网 10.10.2.104，单卡 RTX 4090 24G（GPU 3，CUDA_VISIBLE_DEVICES=3） |
| 基座 | Qwen2.5-1.5B-Instruct（revision `989aa7980e…`，与本机 `weights/hf/Qwen2.5-1.5B-Instruct` sha256 逐字节一致后上传） |
| 数据 | `data/processed/b1_full_v1/response_train.jsonl`（18,653 条，sha256 `a70ca036…`，B1 冻结划分） |
| T3 配置 | `configs/response_sft_unconditional_2ep.json`——与 D4 v2（`configs/response_sft_2epoch.json`）**唯一差别 `condition_mode` 由 `gold` 改 `none`**，其余（epochs 2、batch 1×8、lr 1e-4、warmup 0.03、seed 20261013、LoRA r8/α16）逐项一致 |
| 环境 | Python 3.11.15 / torch 2.12.0+cu132 / transformers 4.57.6 / peft 0.21.2 / accelerate 1.15.0（transformers/peft/accelerate 与原 pod 环境同版本） |
| 生成集 | `results/response_gen/t2_selection_input.jsonl`（160 条，160 会话，12 格分层，seed 20261022，T1 冻结） |
| 解码 | max_new_tokens=256、greedy、bf16（报告回显 device=cuda / dtype=bfloat16 / max_new_tokens=256） |
| T3 臂 | `weights/t3_unconditional_2ep`（adapter sha256 `0fa04f2d…`，36,981,072 B）→ `t3_unconditional.jsonl` |
| D4v2 参照臂 | `weights/response_sft_d4_2ep`（同卡重新生成，不用 10/16 实例历史产物）→ `t3_d4v2_off_samedevice.jsonl |

两臂均于 2026-10-27 00:34 / 00:47（+08）在**同一张 GPU 3 上顺序生成**，符合同设备比较纪律；两臂 id 集合与筛选集完全一致，response_text 全非空（各 160 行，sha256 取回本机后双侧核验一致）。

## 二、训练侧证据

- 全量 4,664 optimizer step（2 epoch，每 epoch 2,332 步），mean_train_loss **2.0171**（首步 2.99967，LoRA 梯度范数 1.06，392 个 LoRA 张量全部更新）。
- 实测 2.3622 s/step，总时长 11,025.9s（约 3.07h；GPU 3 与他人进程共享，比独占的 1.67 慢 ~40%，ETA 预算含 +30% 余量内）。显存峰值 7.22G/24G。
- **周期 checkpoint 18 段**（每 250 步，`checkpoints/step_000250`…`step_004500`，含 adapter+optimizer+scheduler+RNG+位置+**实验身份 manifest**）；本次一次跑完未触发恢复（wrapper attempts=1）。
- adapter 落盘后离线重载校验通过：392/392 张量逐位一致，3 条 greedy 生成复现一致（`checks.adapter_reload.passed=true`）。
- 训练脚本为 PR #34 版本（sha256 `4fd9b7c7…`）；cron 守护（`t3_supervisor.sh`，每 10 分钟）与 wrapper（`train_resilient_104.sh`）overnight 无人值守验证：训练/管线任何一方死亡 10 分钟内自动重启，本报告取回时 `supervisor.log` 已连续记录 ALL_DONE。**注意：本次运行未触发过恢复，上述只证明守护脚本的触发链路就位，不证明恢复正确性**——恢复正确性证据见第五节（故障注入 e2e）。
- **实验身份 manifest（A 审查 P1 新增）**：每个 checkpoint 附带 `manifest.json`——训练数据 SHA-256（含行序）、条件文件 SHA-256、完整配置、基座/tokenizer 逐文件内容指纹、有效训练预算（epochs/batch/累积/max_steps/max_train_samples/每 epoch 步数/总步数）；恢复时在**加载任何状态之前**逐项比对，任何不一致（含旧版无 manifest 的 checkpoint）直接拒绝。测试：7 项纯函数比对用例 + 7 项 tiny 模型 e2e（含「崩溃→恢复→最终权重与未中断运行逐字节一致」的轨迹一致性证明）。

## 三、同设备两臂比较（结论以此为准）

GPU 3，T3（`t3_unconditional.jsonl`，sha256 见取回核验）vs D4v2（`t3_d4v2_off_samedevice.jsonl`），对 gold 的字符级本地诊断：

| 指标 | T3 | D4v2 | 差值 |
|---|---:|---:|---:|
| BLEU-4（字符级，本地诊断） | 0.1034 | 0.1052 | −0.0018 |
| ROUGE-L（字符级，本地诊断） | 0.2502 | 0.2526 | −0.0024 |
| 平均长度（非空白字符） | 129.05 | 125.57 | +3.48 |
| >200 字桶占比 | 34/160 | 32/160 | +2 |

逐条 ROUGE-L 差：中位数 0.0000，p25 −0.0149，p75 +0.0196；**65 胜 / 65 负 / 30 平**；文本完全相同 26/160。长度：67 变长 / 57 变短 / 36 不变。产物：`results/response_gen/t3_arms_comparison_same_device.json`（由新增脚本 `scripts/compare_response_arms.py` 生成，6 项单测覆盖）。

**证据边界（按 A 审查 P2 收紧）**：160 条诊断集上方向混杂、差值极小，既不能证明两臂等价，也不能证明「失配代价 ≈ 0」；本报告不使用跨设备差值（0.0036）充当设备噪声底——跨设备偏移不是同设备重复测量的噪声估计，也不是等价性界限（0.0024 与 0.0036 仅差 1.5 倍，非一个量级）。结论仅为：**该筛选集上未观察到无条件训练的收益**。

## 四、T4 衔接（长回复改进变量分析）

T3/D4v2 两臂的长度特性与 D4 v2 全量 val（2,354 条）误例分析结论一致，长回复仍是主要短板：

- 全量 val 上 D4 v2：>200 字桶 ROUGE-L **0.1958**（各桶最低，次低 101-200 桶 0.2142）；两长桶（101-200 + >200）合计 1,154/2,354 ≈ **49%**。
- 格式类问题（全量 val）：markdown 粗体 342、列表符 169、近空 40、标题 33、引用 29、重复 n-gram 28、重复句 9。
- T3 臂在 160 条筛选集上 >200 桶 34/160（21.3%），与 D4v2 持平；即**无条件训练不改善也不恶化长回复问题**。
- 结论：T4 的候选变量（长度控制/格式约束/长样本加权）保持 open，下一周期在全量 val 同设备协议下评估；本轮不占 GPU。

## 五、验证证据索引

- 单测：`python -m unittest tests.test_compare_response_arms` → **6 tests, OK**（新增）；全套 `python -m unittest discover -s tests` → **358 tests, OK**（含 manifest 比对 7 项 + checkpoint/resume e2e 7 项，主会话实跑）。
- CI 等价检查实跑：`python -m compileall -q src scripts tests`、`bash -n start.sh`、`python scripts/ci_smoke.py` 均 PASS（提 PR 前复跑 `.github/workflows/ci.yml` 全部检查）。
- 主会话独立复验（非采信子智能体结论）：两臂行数/id 集合/非空、生成报告 device/dtype/max_new_tokens 回显、训练报告 checks 全段、取回后 sha256 双侧核验（adapter、两个 jsonl、报告 json 四项全 OK）均由主会话重跑确认。
- **恢复/故障注入证据（可复现）**：`tests/test_train_response_sft.py::CheckpointResumeEndToEndTests` 在 tiny Qwen2.5 上模拟「存档后进程死亡」，恢复后最终 adapter 与未中断运行**逐字节一致**（sha256 相等）；反向用例覆盖换 seed、换 max_steps、数据行序颠倒、换基座目录、删 manifest，全部按设计拒绝。该测试在装 torch/transformers/peft 的环境实跑通过，裸解释器按设计 skip。
- **守护脚本（随仓库交付）**：`scripts/remote/`（train_resilient.sh 进程级自动续跑 / watch_and_generate.sh 守候后顺序生成 / supervisor.sh cron 自愈 + README），参数经环境变量注入，T3 实例的用法记在 README。
- 产物（随 commit 入库）：`results/response_gen/t3_unconditional.jsonl(+_report.json)`、`t3_d4v2_off_samedevice.jsonl(+_report.json)`、`t3_arms_comparison_same_device.json`、`reports/t3_uncond_2ep_report.{json,md}`、`scripts/compare_response_arms.py`、`tests/test_compare_response_arms.py`、`scripts/remote/`。
- 权重不入 Git：`weights/t3_unconditional_2ep/adapter_model.safetensors`（sha256 `0fa04f2d…`，36,981,072 B）已发布 GitHub Release [`d-cycle2-t3-unconditional-2ep-20261027`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-cycle2-t3-unconditional-2ep-20261027)（Pre-release，zip sha256 `9ba1a006…`，发布后经 API 资产端点下载核验通过）。
- 本地 BLEU-4/ROUGE-L 为字符级诊断实现（`src/evaluation/metrics.py`），官方指标未公布，本报告所有分数不冒充官方分数。

## 六、结论与建议

1. **维持条件关闭，暂保留 D4 v2，不据本轮结果改换候选**：160 条筛选集未观察到无条件训练收益（ROUGE-L −0.0024、逐条 65/65/30、方向混杂）；该诊断集规模不足以证明等价，也不支持「失配代价 ≈ 0」的断言。按「无信号不扩大候选」纪律，候选改换留待 T6 完整评测；T3 权重与全套证据留档。
2. **设备一致性纪律继续有效**：本轮两臂同卡顺序生成，逐条可比；T2 的设备对照（跨设备字节一致率仅 28.75%）结论继续有效，跨设备差值不得用作噪声底或等价界限。
3. **T4 为下一周期主攻**：长回复（>200 字桶 ROUGE-L 0.1958、占 49%）与格式问题（markdown 污染 342/2,354）是仅存的系统性短板，建议候选：长度惩罚/截断策略、格式清洗后处理、长样本 loss 加权。按计划一次只选**一个**变量、先短探针再决定是否完成候选，不把长度惩罚/后处理/全量验证扩成本周期已批准之外的新任务。
4. **无人值守流程的结论限定在实际验证范围**：周期 checkpoint + cron 守护 + 取回脚本的触发链路已就位并在 overnight 运行中未被打断；恢复正确性由故障注入 e2e（轨迹逐字节一致）证明，守护脚本随仓库交付可复现。后续长训练直接复用。
