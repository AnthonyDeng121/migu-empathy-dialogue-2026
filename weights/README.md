# 模型权重目录

模型权重不进入普通 Git 历史。这里只提交说明文件和模型清单。

建议工作流：

1. 训练中的临时 checkpoint 只保留在负责人本机或训练平台。
2. 需要其他成员集成时，选出一个冻结候选并导出。
3. 将冻结候选上传为私有仓库的 GitHub Release 附件或 Git LFS 对象。
4. 在 PR 中提交模型版本、下载位置、SHA-256、对应 commit、配置和验证指标。
5. 其他成员下载后放入 `weights/<component>/<version>/`。

不要每个 epoch 都上传权重。

## 已发布权重（GitHub Release）

2026-10-17 A 集成测试锁定最新 C Day5 与 D 2-epoch 候选。C 使用标准 sequence classification 权重（非旧 packaged 情感/画像格式）；D 仍使用固定基座加 adapter。最终采用由 A/B 全量复核决定。

- C 最新：[c-day5-emotion-longtail075-20261016](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-day5-emotion-longtail075-20261016)，资产 `C_DAY5_EMOTION_LONGTAIL075_20261016.zip`，包 SHA-256 `67db558687a42c6b10f31075ff1fecf14ffc6ba881c4d14d0bcd387b0ea0c526`；模型权重 SHA-256 `aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37`。
- D 最新：[d-day4-response-sft-2ep-20261017](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-day4-response-sft-2ep-20261017)，资产 `d4_2ep_adapter.zip`，包 SHA-256 `e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3`；adapter SHA-256 `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e`。模型目录仅含 10 个模型/tokenizer 文件，排除 HANDOFF/SHA256SUMS 后摘要为 `b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb`。

| 候选 | Release tag | 资产 | 关键哈希 |
|---|---|---|---|
| D 回复候选 D2 LoRA/SFT（Qwen2.5-1.5B-Instruct + adapter，2026-10-15） | [`d-day3-response-sft-20261015`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-day3-response-sft-20261015)（Pre-release） | `D_DAY3_RESPONSE_SFT_D2_1P5B_20261015.zip`（adapter + tokenizer 10 文件 + `HANDOFF.md` + `SHA256SUMS.txt`）与 `.sha256` 边车；`…_MERGED_20261015.zip.001/.002` 为 merge 备选分卷（非字节等价，采用需同集重评，见包内 `MERGED_HANDOFF.md`） | adapter_model.safetensors `62a695ea…`；adapter 目录 digest `af4f0e38…`（固定于 `configs/response_artifact.json`）；merged 目录 digest `9dcd82e9…` |
| D 回复候选 v2 LoRA/SFT 2-epoch（Qwen2.5-1.5B-Instruct + adapter，2026-10-17，**候选 v2 待 B2/A 决策，未采用前 D2 仍是交付候选**） | [`d-day4-response-sft-2ep-20261017`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-day4-response-sft-2ep-20261017)（Pre-release） | `d4_2ep_adapter.zip`（adapter + tokenizer 10 文件 + `HANDOFF.md` + `SHA256SUMS`）与 `.sha256` 边车 | zip SHA-256 `e82edaf0…`；adapter 目录 digest `b3e9017e…`（10 文件 / 52,866,512 字节）；同集 val ROUGE-L 0.2598 / BLEU-4 0.1116（vs D2 0.2542/0.1075） |
| D CYCLE2 T3 无条件两轮 SFT（Qwen2.5-1.5B-Instruct + adapter，2026-10-27，**实验候选**；与 D4 v2 同设备两臂比较无可测差异，金标条件训练的部署失配代价 ≈ 0，可按工程需要二选一） | [`d-cycle2-t3-unconditional-2ep-20261027`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-cycle2-t3-unconditional-2ep-20261027)（Pre-release） | `D_CYCLE2_T3_UNCONDITIONAL_2EP_20261027.zip`（adapter + tokenizer 10 文件 + `HANDOFF.md` + `SHA256SUMS.txt`）与 `.sha256` 边车 | zip SHA-256 `63e451ee…`；adapter_model.safetensors（36,981,072 B）SHA-256 `0fa04f2d…`；训练 4,664 步 / loss 2.0171 / 单卡 4090 11,025.9s；160 条筛选集 ROUGE-L 0.2502 / BLEU-4 0.1034（同设备 D4v2 0.2526/0.1052） |
| C 情感/画像候选 Day3 重训（2026-10-15） | [`c-day3-retrained-20261015`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-day3-retrained-20261015)（Pre-release） | `C_DAY3_RETRAINED_EMOTION_PROFILE_20261015.zip` 与 `.sha256` 边车 | 见 C 的交接报告 `reports/C_DAY3_RETRAINED_HANDOFF_20261015.md` |

下载后放入 `weights/<component>/<version>/`，先跑 `sha256sum -c` 边车/包内校验再使用。
基座模型（Qwen2.5-1.5B-Instruct 等）为 HuggingFace 公开权重，不随 Release 分发，revision 见各 HANDOFF。


