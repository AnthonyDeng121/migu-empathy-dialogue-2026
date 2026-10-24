# D 岗位 CYCLE2 T1：条件对照准备与无条件 SFT 配置检查

日期：2026-10-22（Asia/Shanghai）。分支：`feature/d-day7-deadline-standby`（自 main `6aec2a0` 同步后的 `9e1ff15` 起，4 个交付 commit）。状态：Integration Gate（待 A 审）。

对应执行计划 T1（D 列）：「用现有最新版 C/D 和脚本准备关闭/仅情感对照，核对可复用的旧预测；完成无条件 SFT 的数据和配置检查，不等 A 新入口」。本轮**未开长 GPU 任务**：仅本机 4060 一次 8 样本/2 步冒烟与 C Day5 条件生成的 CPU 推理。

## 结论摘要

- **关闭 vs 仅情感对照已具备开跑条件**：仅情感臂的推理端开关（`--condition-fields emotion`）、最新 C Day5 的条件文件（2354 条，指标与 C 报告逐位对账）、固定分层筛选集（160 条，12 格全覆盖）三件齐备；off 臂可从既有 D4 v2 全量预测按 id 直接抽取，**零 GPU**。
- **无条件 SFT 配置就绪**：`configs/response_sft_unconditional_2ep.json` 与 D4 v2 单变量（仅 condition_mode gold→none，脚本比对），数据无泄漏、键对齐已核，轻量冒烟端到端 PASS。
- 发现并补齐一个真实缺口：**现有 `build_condition_file.py` 只认旧 packaged 联合格式，无法消费最新 C Day5 标准情感权重**——已新建 `scripts/build_emotion_condition_file.py` 补上（推理约定与 C 训练脚本逐条一致，指标逐位对账即接线正确的证据）。

## 交付物（commit 均在 `feature/d-day7-deadline-standby`，main 侧无开放 PR）

| commit | 内容 |
|---|---|
| `6c44c68` | `scripts/run_response_inference.py` 增加 `--condition-fields {both,emotion}`（默认 both，已记录运行逐字节不变）；emotion 臂只校验 emotion_label 并丢弃 user_profile；性能报告新增 `condition_fields`。6 个新单测（RED：`TypeError: _load_conditions() got an unexpected keyword argument 'fields'` 等 6 项失败 → GREEN）。 |
| `3c31291` | `configs/response_sft_unconditional_2ep.json`（单变量配置）+ 2 个新单测（含变异实验证明有牙）+ 本机 4060 冒烟（PASS，证据 `logs/D_CYCLE2_T1_UNCOND_SMOKE.txt`、`reports/d_cycle2_t1_uncond_smoke.json`）。 |
| `f884735` | `scripts/build_stratified_subset.py` + 25 个单测 + 冻结筛选集 `results/response_gen/t2_selection_input.jsonl`（160 行）+ `t2_selection_manifest.json`。 |
| `5dc9c7c` | `scripts/build_emotion_condition_file.py` + 17 个单测 + `results/response_gen/c5_conditions_val_full.jsonl`（2354 行 emotion-only）+ 生成报告（含指标对账）。 |

## 一、关闭/仅情感对照准备

### 1.1 仅情感臂开关（`6c44c68`）

`system_prompt` 本就能只在 condition 带 emotion 时渲染「情感状态」行，但推理脚本的加载器强制要求 dict 型 user_profile，仅情感文件会被拒。新增 `--condition-fields emotion`：只校验 emotion_label（字符串），user_profile 可缺省/为 null 并被丢弃。同一份 C 条件文件驱动两臂，避免出现第二个会漂移的条件文件版本。回归防护：`both` 模式（默认）仍拒绝缺 profile 的行，所有历史运行参数不变。

### 1.2 最新 C Day5 条件文件（`5dc9c7c`）

- 下载 Release `c-day5-emotion-longtail075-20261016` zip（409,585,966 字节）并三方比对 SHA-256：边车、实算、`weights/README.md` 登记值均为 `67db5586…`；解落 `weights/classification/c_day5_longtail075/`（weights/ 不入 Git），`model.safetensors` SHA-256 `aedbacb1…` 与登记值一致。
- C Day5 是标准 `BertForSequenceClassification`（16 标签，id2label 顺序恰为 `sorted(EMOTIONS)`）的**情感专用**权重，无画像头——旧 packaged 联合加载器（`EmotionProfileCheckpoint`）要求的三件套一个都不存在，这是既有脚本覆盖不到的路径（非缺陷，是格式代差）。
- 新脚本推理约定逐条照抄 C 的 `scripts/finetune_emotion_balanced.py`：SPEAKERS 文本格式、`truncation_side="left"`、max_length 256、batch 32、argmax over `sorted(EMOTIONS)`。行级校验：上下文必须以 user 轮结尾且 `turn_id == target_user_turn_id`（防未来轮泄漏）、拒缺字段/重复 id、预测数与行数不符拒绝写半份。
- **对账**：全量 2354 条，Accuracy **0.716653** / Macro-F1 **0.584843**，与 `reports/c_day5_emotion_longtail075_metrics.json` 的 Day5 值逐位一致（同一 `classification_metrics` 函数、同一推理路径）。
- 产物 2354 行、键恰为 `{id, emotion_label}`、id 与 `classification_val.jsonl` 的 sample_id 逐行同序；`_load_conditions(..., fields="emotion")` 全部加载成功；emotion 模式提示词末行 `[当前用户状态参考]情感状态：<label>`，不含「用户画像」。

### 1.3 固定分层筛选集（`f884735`）

- 160 条（计划区间 128–200 内）/ 160 个不同 conversation（池 900，每会话至多 1 条）；12 格 = 历史长度（≤4 / 5–12 / >12 轮）× 金标画像（空 / 非空）× 情感（主要 top5：anxiety/relaxed/mixed/helplessness/sadness；稀有其余 11 类），最大余数法按格比例分配、非空格下限 2、seed 20261022。
- 各格选中（short/mid/long × empty/non_empty × major/rare）：12/5、10/5、20/9、38/17、9/4、21/10，合计 160；`selected == quota` 一轮填满。
- 复核（主会话重跑）：同 seed 重生成选择与 manifest **字节级一致**；160 行与 response_val 同 id 行逐字段相等、无金标字段泄漏；12 格计数独立重算与 manifest 一致；两个源文件 SHA-256 与 `data/processed/b1_full_v1/manifest.json` 登记值一致；换 seed=999 仅 8/160 重叠（seed 非摆设）。
- 金标仅用于分层（manifest `gold_labels_note` 声明），筛选集即 T2 两臂固定输入。

### 1.4 可复用旧预测核对（T2「匹配的已有基线不重算」）

T2 目标配置 = 固定筛选集 + 最新 C Day5 条件 + D Day4 v2 adapter + max_new_tokens=256。全部历史 full-set 预测（各 2354 条）核对结果：

| 既有预测 | 配置 | T2 可复用 |
|---|---|---|
| `results/response_gen/d4_2ep_val_full.jsonl` | D4 v2，condition_mode=none，max_new_tokens=256 | **是：off 臂按 id 抽取，零 GPU** |
| `results/response_gen/d2_1p5b_val_full.jsonl` | D2，none，256 | 否（D2 非最新，仅历史对照） |
| `results/response_gen/original_1p5b_val_full.jsonl` | 原模型，none，256 | 否（非最新 D） |
| `results/response_gen/d2_1p5b_val_full_conditioned.jsonl` | D2 + 预测条件（来源 c2 = C Day3，双字段），256 | 否（D 与 C 两个维度都旧） |
| `results/response_gen/d2_1p5b_val_full_minlen.jsonl` / `d2_1p5b_val100_cap512.jsonl` | D2，解码参数不同（minlen / cap512 探针） | 否（解码不固定） |
| `results/response_gen/c2_conditions_val_full.jsonl` | C Day3 条件文件（commit `cb11158`） | 否（T2 要求最新 C Day5，已由 1.2 替换） |

off 臂复用证据：`d4_2ep_val_full_report.json`（condition_mode=none、2354 条 id 与 response_val 逐行同序、mean 3.3176 s/sample）+ 本地 `weights/response_sft_d4_2ep` 目录 digest `b3e9017e…`（10 文件/52,866,512 字节）与 Release 登记 MATCH。仅情感臂无任何匹配既有预测，T2 必须新跑（按 3.32 s/sample 估 160 条约 9 分钟，远程 4090 口径；本机 4060 需重测）。

## 二、无条件 SFT 数据与配置检查（`3c31291`）

- **数据**（主会话实核）：response_train 18,653 / response_val 2,354；classification 与 response 两侧 sample_id 集合分别全等；train/val conversation_id 交集 **0**（8,329 vs 900 会话，无泄漏）；val 金标画像全空 875/2,354（37.2%）。
- **配置**：`configs/response_sft_unconditional_2ep.json` 与 `configs/response_sft_2epoch.json`（D4 v2）键序一致、除 `condition_mode`（gold→none）与 `_comment` 外逐字段相等（脚本比对）。基座/数据/预算（epochs 2、batch 1×累积 8、lr 1e-4、seed 20261013 等）全部不变，单变量检验「金标条件训练 vs 部署不注入」的失配。
- **回归防护**：`build_sft_samples_from_b1` 在 condition_mode=none 时即使传入金标行也不注入（变异实验：忽略 condition_mode 回退 gold → 用例 FAIL）；b1 + 非 gold 模式的复现命令不带 `--condition-jsonl`（变异：总是追加 → 用例 FAIL）。
- **冒烟**（本机 4060，8 样本/2 步，local_files_only）：PASS；report 回显 condition_mode=none、`condition_train_sha256=null`、optimizer_steps=2、first_loss 3.12、离线重载 passed、峰值显存 6.71GiB。**本机全量 2 epoch 外推约 15.7h**——T3 正式训练按既定分工上远程 GPU（[[gpu-work-remote-server]]），本机不跑长训练。

## 三、验证证据索引

- 全量单测：`python -m unittest discover -s tests` → **260 tests, OK**（周期前基线 210 + 本轮新增 50：仅情感臂 6、无条件配置 2、分层筛选集 25、C Day5 条件文件 17）。
- CI 等价检查实跑：`compileall -q src scripts tests`、`bash -n start.sh`、`bash -n scripts/cloud_smoke.sh`、`python scripts/ci_smoke.py` 均 PASS。
- 每个 RED→GREEN 的失败输出见各 commit 说明；C Day5 指标对账、筛选集字节级复现、off 臂 digest 匹配为本报告 1.2/1.3/1.4 节主会话重跑结论。

## 四、T2/T3 恢复步骤（下一会话直接可用）

T2（固定筛选集两臂，GPU）：

```bash
# off 臂：零 GPU，从 D4 v2 全量预测按筛选集 id 抽取（T2 时写抽取脚本或一行 jq/python）
# 仅情感臂（约 9 分钟 @4090）：
python scripts/run_response_inference.py \
  --input results/response_gen/t2_selection_input.jsonl \
  --model weights/hf/Qwen2.5-1.5B-Instruct \
  --adapter weights/response_sft_d4_2ep \
  --condition-jsonl results/response_gen/c5_conditions_val_full.jsonl \
  --condition-fields emotion \
  --output results/response_gen/t2_emotion_only.jsonl \
  --performance-report results/response_gen/t2_emotion_only_report.json \
  --device cuda
```

T3（无条件两轮 SFT）：用 `configs/response_sft_unconditional_2ep.json` 上远程 GPU，先短跑测时再全量；与 D4 v2 同评测协议比较（固定筛选集或全量，报告口径写清）。

## 五、未做项与阻塞

- 未跑 T2 两臂推理（等 GPU 时段）；未启动 T3 训练（等远程 GPU，本机外推 15.7h 不可行）。
- 本地遗留 `results/tmp_uncond_smoke/`（47MB 冒烟产物，gitignored；`rm` 被本机策略拦截，留待用户或后续清理，不影响入库）。
- 接口观察：A 的 T1 工作（`origin/feature/a-mlops` `7427735`，独立画像接入）重构了 packaged 元数据校验但未改预测路径，与本轮 D 改动无冲突；D 的新条件文件走标准格式路径，与 A 的 `run_integrated.py` sequence 分支要求的三文件一致。
