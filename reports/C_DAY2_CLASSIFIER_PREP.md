# C 岗位 Day 2：分类选型与训练准备

**状态：CPU/MPS 准备与冒烟测速已产出；正式 GPU 训练仍 Blocked。**

## 按新版 main 的任务范围

本次对齐 `origin/main` 的 `docs/EXECUTION_PLAN.md` 与 `docs/TRAINING_WORKPLAN.md`：C 负责分类候选、训练脚本和小样本测速；正式训练依赖 B 的数据 v1 及 A 核实的 GPU/卡时。当前 `data/processed/` 只有目录说明，没有 B1 生成的分类 JSONL，因此本次只用官方 raw train/val 验证准备链路，不声称已使用 B1 数据，也不把本机 MPS 计作平台 GPU 卡时。

## 候选与决定

| 对照 | 验证 Accuracy | Macro-F1 | 用途/决定 |
|---|---:|---:|---|
| 既有多数类 baseline | 31.9031% | 见分类 baseline 报告 | 同集参照下限 |
| `hfl/chinese-macbert-base`，revision `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1` | 既有全量一轮 MPS 结果 67.8420% | 0.4032 | 选为 C1 主候选，先完成训练/导出链路冒烟 |

模型仓库页面标注 Apache-2.0；这只确认模型仓库所列许可证，赛事对预训练模型的使用许可仍需 A 按官方规则核实。[模型卡](https://huggingface.co/hfl/chinese-macbert-base)

本次 1,600 条小样本训练后的全验证结果是 Accuracy 38.8700%、Macro-F1 0.0648。由于训练只用 1,600/18,657 个点且样本按原始展开顺序截取，类别覆盖很不均衡；该结果只证明脚本可跑和 checkpoint 可导出，不用于模型质量结论或最终采用。

前 1,600 点中 `care=2`、`disgust=5`、`surprise=5`；该切片非分层样本。此前对全量官方 raw train 做的一轮 MPS 实验结果单独记录在 [全量 baseline 报告](EMOTION_ENCODER_BASELINE.md)，也仍须按 B1 v1 和平台 GPU 条件复核。

## 冒烟运行证据

- 设备：Apple M5 本机 PyTorch MPS；PyTorch `2.14.0`，Transformers `5.17.0`；seed `20261012`，batch size `32`，max length `256`，1 epoch。
- 数据：官方 raw train `1,600/18,657` 个 prediction points；官方 val `2,354/2,354`；train/val conversation ID overlap `0`。B1 转换产物尚未落地。
- 训练：50/50 optimizer steps；mean train loss `2.3379`；训练 step 耗时合计 `160.269s`，`3.205388s/step`。按全量单轮 `584` steps 外推约 `1,871.947s`（31.2 分钟），这是本机 MPS 估算，不是平台 GPU ETA。
- 脚本报告的训练、全量验证和保存总耗时 `215.266s`。日志结束时间为 2026-10-13 11:29:59（Asia/Shanghai）；起始墙钟时间未被本次命令记录，故不补写估算值。
- 检查：loss 有限；首步分类头梯度范数 `8.33948231` 且有限非零；分类头权重发生变化；保存后从本地 checkpoint 重载，权重逐项一致。MPS 冒烟没有消耗平台 GPU 卡时。
- 初次 cache-only 预检因本机没有基础 MacBERT 缓存而在训练前失败；随后从固定 revision 在线下载模型并完成本次运行。离线交付使用本地 checkpoint，不依赖该基础模型缓存。
- 本机 smoke checkpoint：`weights/emotion_baseline_c_day2_smoke_20261013/model.safetensors`，409,143,304 bytes，SHA-256 `d0d604247c05657942dec865a3a5eadacebc4b1a2ea8d5271dfb66ff9ddadfc3`。权重未提交 Git，符合仓库约定。
- 训练脚本 SHA-256（本次冒烟运行时）：`ba3db44cbc5a9ab755881eb90390d0805bc786a15ab45e1c486b30d3564ebd84`。

完整逐类指标、混淆矩阵、数据哈希和复现命令见 [smoke 结果](C_DAY2_EMOTION_SMOKE.md) 与 [JSON 记录](c_day2_emotion_smoke.json)；逐步 stdout 见 `logs/C_DAY2_EMOTION_SMOKE.txt`。

## 验收与下一步

- C 的选型、脚本和 MPS 小样本准备已提交 PR，等待 D 验收。
- B 发布并固定数据 v1 后，用相同脚本和 conversation 分组契约重跑至少 50 optimizer steps；若 B1 schema 与 raw official schema 不兼容，先更新转换适配，不绕过契约。
- A 核实平台 GPU、可用卡时、训练环境和存储后，才把任务从 Blocked 改为可启动，并基于平台 GPU 短跑重算正式 ETA。
- 本机 MPS 结果不能替代平台训练、全量候选评测、官方镜像集成或赛事许可确认。
