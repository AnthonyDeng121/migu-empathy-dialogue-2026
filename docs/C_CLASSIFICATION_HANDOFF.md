# C T7 分类模型交接

日期：2026-11-03。责任人 C；代码与技术交接由 D 审查。机器可读清单为 `configs/classification_handoff_t7.json`。本说明供 A 集成、B 独立评测和 D 审查使用。

## 固定选择

沿用 B T6 的选择：Day5 情感、D4v2 回复、条件关闭、空画像。C T6 已通过 PR #44 合入 main（`82f0768`）。本次交接不另训、不调整阈值；T7 交接仍待 D 审查。

| 项目 | 文件目录 | 用途 |
|---|---|---|
| Day5 情感 | `weights/emotion_longtail075_b1_20261016` | 组合默认；`classifier_format=sequence` |
| T4 情感 0.65 | `weights/emotion_longtail065_b1_20261028` | 冻结的研究候选，供 B 后续逐条比较 |
| 画像 encoder v1 | `weights/profile_encoder_v1_b1_20261026` | 独立画像内部候选；不替换默认空画像 |
| 旧 unweighted 画像头 | `weights/profile_unweighted_b1_20261015` | NO-GO，仅留作历史比较 |

`configs/integrated.json` 是当前组合默认。显式研究画像时使用 `configs/integrated_profile_v1_off.json`。画像 checkpoint 内保留的情感头不参与选定情感输出；集成入口的 `apply_profile` 只替换画像字段。

## 输入与解码契约

- 官方入口：`scripts/run_integrated.py`；分类输入仅为目标用户轮及此前的 `history`。目标轮必须为 user，不读取未来回复、验证金标或验证集检索文本。
- 文本按轮次用 `用户：内容` / `助手：内容` 格式，以换行连接。情感入口调用 `emotion_encoder._format_context`，B1 概率导出调用 `build_emotion_condition_file.read_rows`；两者使用相同的上下文范围和中文前缀。
- 两类模型均使用左截断，显式 `max_length=256`。tokenizer 的 `model_max_length` 是超大哨兵值，不能替代显式长度。
- 情感为 16 类 logits argmax，没有拟合阈值；下标必须按照 checkpoint `id2label`。选定 checkpoint 的映射与 `tuple(sorted(EMOTIONS))` 一致，完整顺序见机器清单。
- 默认画像三维均为空数组。独立画像使用各 checkpoint metadata 的标签顺序，以 sigmoid 概率 **大于等于**逐维阈值纳入标签；等于阈值时保留。

| checkpoint | personality_traits | interests | style |
|---|---:|---:|---:|
| 画像 v1 | 0.64 | 0.81 | 0.56 |
| 旧 unweighted 头 | 0.18 | 0.16 | 0.27 |

两套画像模型的 metadata、heads、encoder 和 tokenizer 必须成套使用；阈值来自训练内 calibration，本轮没有重新选择。

## 选择依据与边界

沿用 C T6 已冻结的本地实验数据：情感 Day5 Accuracy/Macro-F1 为 0.716653/0.584843，T4 为 0.714953/0.590474；care 两者均为 0/11。T4 暂无独立采用结论。

画像三维等权 macro exact：空画像 0.6282922685、旧头 0.3824695554、v1 0.6468422543。v1 严格高于同集空画像，增量约 0.01855，但 personality_traits 和 style 低于空画像；任一维预测非空率为 585/2354，非空存在检测 precision/recall 为 0.9043/0.3577。这两个数描述是否预测出任意非空画像，不能当作逐标签 precision/recall；v1 的 global label micro precision/recall 为 0.5678/0.1276。

验证集曾被历史实验复用，以上属于本地诊断。画像 v1 高于空画像并不证明组合采用或官方成绩提升。正式组合继续遵循 B 的 Day5 + 空画像选择。

## 权重、概率与完整性

- 默认情感：[Day5 Release](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-day5-emotion-longtail075-20261016)，ZIP SHA-256 `67db558687a42c6b10f31075ff1fecf14ffc6ba881c4d14d0bcd387b0ea0c526`。
- 内部画像候选：[profile v1 Release](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-cycle2-profile-v1-20261026)，ZIP SHA-256 `761761f619643fd8209d309d0d4cbc684d1aa2374c9542c26283ba961e514060`。
- 机器清单列出四套 checkpoint 的逐文件路径、字节数和 SHA-256，以及四份 2354 条概率附件的路径/哈希。原始数据和权重不进入 Git。
- ZIP 摘要、单文件摘要、目录树摘要是三种对象。T6 的历史 tree 摘要按 `scripts/train_emotion_profile.py::_tree_sha256` 算法生成，清单注明算法；不得与采用其他序列化规则的打包树摘要直接比较。跨打包环境可逐文件对照清单。
- 解包后把模型文件放到清单指定路径，保持相对目录结构。Release 的 checksum 和交接文件不要额外放入模型目录后再与历史目录摘要比较。

概率复现示例（需要本地 B1 数据和推理依赖）：

```bash
python scripts/export_classification_probabilities.py --kind emotion --input data/processed/b1_full_v1/classification_val.jsonl --checkpoint-dir weights/emotion_longtail075_b1_20261016 --output .codex_tmp/acceptance/c_t7/emotion_day5.jsonl --device cpu
python scripts/export_classification_probabilities.py --kind profile --input data/processed/b1_full_v1/classification_val.jsonl --checkpoint-dir weights/profile_encoder_v1_b1_20261026 --output .codex_tmp/acceptance/c_t7/profile_v1.jsonl --device cpu
```

跨设备概率可能有浮点误差，应核对全部 ID、标签、阈值解码和误差范围。已有 T6 概率附件仅含 ID、预测与概率，不含正文和金标。A 负责最终镜像与官方入口验证；B 负责独立选择与不确定性；C 提供本清单及分类资产；D 按固定闭环审查 C 交付。验收日志及其附件仅保存在本地 `.codex_tmp/acceptance/`。
