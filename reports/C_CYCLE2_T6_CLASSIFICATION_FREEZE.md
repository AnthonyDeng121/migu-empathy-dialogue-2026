# C Cycle2 T6：分类候选全量复核与冻结

日期：2026-11-02（Asia/Shanghai）。状态：Integration Gate，等待 D 同行审查。覆盖 11/01–11/02 的 T6 分类任务。本轮不训练、不改阈值、不替换正式默认模型。

## 结论

- 同一份 B1 validation 上完成空画像、旧画像头与画像 v1 的比较。旧头 macro exact **0.382470**，低于空画像 **0.628292**，NO-GO。画像 v1 macro exact **0.646842**，高于空画像；但预测非空率只有 **24.85%**，非空 precision/recall 为 **0.9043/0.3577**，personality_traits 和 style 两维 exact 仍略低于空画像。保留为内部候选，不单独据此替换组合默认。
- 情感 Day5 默认模型全量复算 Accuracy/Macro-F1 **0.716653/0.584843**；T4 权重指数 0.65 候选为 **0.714953/0.590474**。T4 候选的 mixed F1 从 **0.166667** 到 **0.212766**，但 care 仍为 **0/11**，Accuracy 略降；权重和类别映射作为候选冻结，不做采用结论。
- 本周期组合冻结继续遵循 B T6 的 [PR #43](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/pull/43)：**Day5 情感 + D4v2 + 条件关闭 + 空画像**。这次补齐了 T4 情感候选逐条概率，供之后独立复核使用；不把它倒算成 PR #43 当时已完成的独立复核。

## 固定输入与候选配置

| 项目 | 冻结记录 |
|---|---|
| 数据 | `data/processed/b1_full_v1/classification_val.jsonl`，2,354 条 / 2,354 个 sample ID / 900 个会话，SHA-256 `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201` |
| 情感标签 | 固定 16 类，按 checkpoint `id2label` 顺序；预测为 softmax argmax，无拟合阈值；max length 256、左截断、batch 32 |
| Day5 情感 | `weights/emotion_longtail075_b1_20261016/`；类别权重指数 0.75；模型文件 SHA-256 `aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37` |
| T4 情感候选 | `weights/emotion_longtail065_b1_20261028/`；类别权重指数 0.65；模型文件 SHA-256 `bc0c7cab2259d2aea5ca77e812caff1363575da48ee94b92c03e84c39709e895` |
| 旧画像头 | `weights/profile_unweighted_b1_20261015/`；checkpoint tree SHA-256 `1b1bee3d8cf5175ea7d7c771015f3aa7ecd5455b740c3f93e2bce1ce98bfd4b0`；max length 256；训练内阈值 personality_traits 0.18、interests 0.16、style 0.27 |
| 画像 v1 | `weights/profile_encoder_v1_b1_20261026/`；checkpoint tree SHA-256 `a8dfb90341c4a5d8ad72a6fd38f5741489114b24efd4898805d0f63ca0542bf7`；max length 256；训练内阈值 personality_traits 0.64、interests 0.81、style 0.56 |
| 执行设备 | Apple M5 MPS；公共 GPU 卡时 0；未训练 |

## 画像同集结果

主指标沿用 B T1 预先冻结的三维 exact 等权 macro，采用门槛是严格高于同集空画像。下表逐项来自 B 的 `b_profile_eval.py`，比较输入覆盖均为 2,354/2,354 IDs。

| 画像方案 | Macro exact | personality_traits exact | interests exact | style exact | 预测任一维非空 | 非空 precision / recall / F1 |
|---|---:|---:|---:|---:|---:|---:|
| 空画像 | 0.628292 | 0.618522 | 0.748938 | 0.517417 | 0/2,354 | 0 / 0 / 0 |
| 旧 unweighted 头 | 0.382470 | 0.278250 | 0.531861 | 0.337298 | 2,023/2,354（85.94%） | 0.6604 / 0.9033 / 0.7630 |
| 画像 v1 | **0.646842** | 0.607052 | **0.819456** | 0.514019 | 585/2,354（24.85%） | **0.9043 / 0.3577 / 0.5126** |

画像 v1 对空画像的会话级胜/平/负为 **151/639/110**，平均配对分差 `+0.015357`；旧头为 **136/127/637**，平均配对分差 `-0.258712`。v1 的 macro exact 增益主要来自 interests，另外两维低于空画像；非空召回仍低，因此本次冻结为内部候选，集成默认继续空画像。

## 情感同集结果

| 候选 | Accuracy | Macro-F1 | mixed P/R/F1（support=177） | care F1（support=11） |
|---|---:|---:|---:|---:|
| Day5，权重指数 0.75 | **0.716653** | 0.584843 | 0.461538 / 0.101695 / 0.166667 | 0 |
| T4，权重指数 0.65 | 0.714953 | **0.590474** | 0.431034 / 0.141243 / **0.212766** | 0 |

T4 候选较 Day5 Macro-F1 `+0.005631`，Accuracy `-0.001700`；`care` 未改善。按 B 已完成的 T6 选择维持 Day5。T4 候选现有逐条概率可供后续 B 复核，但尚未通过独立复核，也没有盲测/官方结论。

## 可复核概率文件

文件只有 sample ID、conversation ID、预测标签和概率，不含对话正文或金标。所有文件均为 2,354 条，ID 顺序与固定 validation 一致。

| 文件 | 用途 |
|---|---|
| `reports/c_cycle2_t6_emotion_day5_predictions_20261102.jsonl` | Day5 情感概率与预测 |
| `reports/c_cycle2_t6_emotion_weight065_predictions_20261102.jsonl` | T4 情感候选概率与预测，供后续独立复核 |
| `reports/c_cycle2_t6_profile_unweighted_predictions_20261102.jsonl` | 旧画像头概率与按训练内阈值解码结果 |
| `reports/c_cycle2_profile_v1_predictions.jsonl` | 画像 v1 已有完整概率；本轮重新推理标签 2,354/2,354 一致，最大概率绝对差 `5.96e-8` |
| `reports/c_cycle2_t6_classification_freeze_20261102.json` | 全量指标、模型/数据/附件指纹、配对比较和选择记录 |

使用入口：`scripts/export_classification_probabilities.py`。情感训练/推理输入均沿用 `scripts/build_emotion_condition_file.py` 的格式化、左截断和 256 长度；画像沿用 packaged checkpoint metadata 中的标签顺序、阈值和 max length。概率仅用于留档与诊断，本轮没有用 validation 调阈值。

## 验收边界

- 旧/新画像和两套情感候选都完成全量 ID 对齐；画像 summary/compare 的 `b_profile_eval.py` 检查通过，概率文件结构及范围校验通过。
- 导出器专项单测 4/4 通过，`compileall` 和 `git diff --check` 通过。仓库全量单测 493 项中 479 项通过、13 项跳过、1 项报错：既有 `test_budget_exhaustion_writes_gave_up` 在 `tests/test_workload_scripts.py:1240` 读取不到临时 `relaunch.log`；单独复跑仍失败，失败位置属于未改动的 D workload supervisor 测试。
- B1 validation 在历史实验中复用过，不是盲测；所有数字均为本地诊断，不等于官方分数或隐藏集提升。
- B 已在 PR #43 锁定 Day5 情感、D4v2、条件关闭、空画像。这里补齐预测附件，不改变 B 的既有独立审查范围和结论。
- C T6 自身状态保持 Integration Gate，须由 D 按固定闭环审查后才能标记 Done；未声称已取得 D 验收。
