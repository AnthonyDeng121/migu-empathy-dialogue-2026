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

| 候选 | Release tag | 资产 | 关键哈希 |
|---|---|---|---|
| D 回复候选 D2 LoRA/SFT（Qwen2.5-1.5B-Instruct + adapter，2026-10-15） | [`d-day3-response-sft-20261015`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-day3-response-sft-20261015)（Pre-release） | `D_DAY3_RESPONSE_SFT_D2_1P5B_20261015.zip`（adapter + tokenizer 10 文件 + `HANDOFF.md` + `SHA256SUMS.txt`）与 `.sha256` 边车；`…_MERGED_20261015.zip.001/.002` 为 merge 备选分卷（非字节等价，采用需同集重评，见包内 `MERGED_HANDOFF.md`） | adapter_model.safetensors `62a695ea…`；adapter 目录 digest `af4f0e38…`（固定于 `configs/response_artifact.json`）；merged 目录 digest `9dcd82e9…` |
| D 回复候选 v2 LoRA/SFT 2-epoch（Qwen2.5-1.5B-Instruct + adapter，2026-10-17，**候选 v2 待 B2/A 决策，未采用前 D2 仍是交付候选**） | [`d-day4-response-sft-2ep-20261017`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/d-day4-response-sft-2ep-20261017)（Pre-release） | `d4_2ep_adapter.zip`（adapter + tokenizer 10 文件 + `HANDOFF.md` + `SHA256SUMS`）与 `.sha256` 边车 | zip SHA-256 `e82edaf0…`；adapter 目录 digest `b3e9017e…`（10 文件 / 52,866,512 字节）；同集 val ROUGE-L 0.2598 / BLEU-4 0.1116（vs D2 0.2542/0.1075） |
| C 情感/画像候选 Day3 重训（2026-10-15） | [`c-day3-retrained-20261015`](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-day3-retrained-20261015)（Pre-release） | `C_DAY3_RETRAINED_EMOTION_PROFILE_20261015.zip` 与 `.sha256` 边车 | 见 C 的交接报告 `reports/C_DAY3_RETRAINED_HANDOFF_20261015.md` |

下载后放入 `weights/<component>/<version>/`，先跑 `sha256sum -c` 边车/包内校验再使用。
基座模型（Qwen2.5-1.5B-Instruct 等）为 HuggingFace 公开权重，不随 Release 分发，revision 见各 HANDOFF。


