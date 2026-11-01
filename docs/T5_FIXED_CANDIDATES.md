# A T5 固定候选与来源

日期：2026-10-30。输入基线：origin/main `b4c07f7`（C PR #38、D PR #39 已合并）。离线候选组装及真实三模型加载自检已完成，待 B 独立复核。

## 最新模型决定

- 情感：C Day5 `emotion_longtail075_b1_20261016` 仍是确定交付版本。C T4 `emotion_longtail065_b1_20261028` 仅作为内部候选，不替换默认；当前没有其本地 checkpoint 和完整逐 ID 预测，不能伪造“仅换情感”结果。
- 画像：C v1 `profile_encoder_v1_b1_20261026` 保留为独立画像研究候选；v2 no-go。B T3 CONDITIONAL GO 不等于批准向 D 注入画像。
- 回复：D T4 长样本加权 NO-GO，继续采用 D4 v2 `response_sft_d4_2ep`。D T3 同样未证明改换收益，不采用其替换 D4 v2。
- 条件：D 最新 T5 四组比较均未过预注册触发线，维持关闭；对应预测与决策已入 main，无需等待或重跑。

## 交付与复现

运行前从官方原始数据复原 B1（如已有则不要覆盖）：

```powershell
python scripts/prepare_training_data.py --mode full --output-dir data/processed/b1_full_v1
python scripts/build_t5_candidates.py --output-dir results/cycle2_t5_fixed_20261030
```

输出目录必须尚不存在。新目录可重复执行验证。脚本固定读取已有配置和模型决定，不改变正式入口默认配置。

provenance v2 的每个文件路径都带 `path_base`：`repo` 表示仓库根目录，`artifact` 表示本候选包根目录。`path` 使用相对 POSIX 路径，不包含开发机或输出目录的绝对位置；模型 ID 为仓库相对路径。将整个候选包移动到另一输出目录后，11 个交付文件（含 manifest、catalog 及关联哈希）应逐字节一致。来源文件超出声明根目录时拒绝生成。CLI 外部输入可通过 `--provenance-root` 声明来源根目录。

`results/cycle2_t5_fixed_20261030/catalog.json` 是交付索引，记录配置、预测、manifest、源文件指纹、实际本地权重目录可用性、缺失项和画像标签/阈值。两组均为 2,354 条，使用 D 最新 `d4v2_val_newcard.jsonl`；D 输入不变且无分类条件，允许复用回复。

| 候选 | 情感字段 | 画像字段 | 回复字段 | 状态 |
|---|---|---|---|---|
| baseline | C Day5 全量预测 | 三维空数组 | D4 v2 新卡全量预测 | 离线自检完成，待 B 复核 |
| profile_v1 | 同基线 | C v1 全量预测 | 同基线，逐字不变 | 研究候选，待 B 复核 |
| 仅换情感/情感+画像组合 | C T4 0.65 | 空/v1 | 可复用同一无条件回复 | 缺新情感权重及逐条预测，未组装 |
| 仅换回复/含新回复组合 | Day5 | 空/v1 | D T4 | NO-GO，不扩大采用候选；D 原始实验全量结果可供 B 查阅 |
| 条件变化 | 预测分类 | 按条件策略 | 必须使用真实条件生成 | 最新 D T5 NO-GO，不把离线字段替换当作条件收益 |

## 检查与限制

脚本核对 B1 输入与 D 冻结 SHA-256，复用 T4 组装器的回复报告哈希/条数/无条件门禁、ID 完整性/顺序、标签枚举、字段 schema、非空回复及来源前后一致性。画像 packaged 文件、标签、三维阈值与 max_length 通过现有 loader 的静态校验；配置通过正式 `load_config`。

已从 C Release、本地 D Release 包和固定 Qwen revision 补齐全部选定权重。Day5 ZIP 及四个模型文件哈希通过；D 基座目录摘要 `3dad5ec4…`、adapter 摘要 `b3e9017e…` 与仓库钉值一致。画像 v1 六个文件与 C Release ZIP 逐字节一致，ZIP SHA-256 为 `761761f6…`。

使用独立本地环境 `../.codex_tmp/t5-inference-env`（torch `2.5.1+cpu`、transformers `5.17.0`、peft `0.21.2`、accelerate `1.15.0`），通过正式入口的 `Models` 真实加载三模型，CPU 跑两个样例（含空/非空画像）：情感和画像均与交付预测一致，回复非空且 schema 合法。非空画像样例切回空画像后再次生成，回复逐字一致，情感不变，条件始终关闭。本地检查仅覆盖 device/hardware_label，不改交付配置。

这是少量加载和接线检查，不是全量重生成或正式性能测试，未占用公共 GPU。catalog/manifest 中的 `model_inference_performed=false`、`checkpoint_reload_performed=false` 描述离线组装步骤本身；后续真实加载证据仅保存于本地 `.codex_tmp/acceptance/cycle2/a_t5_20261030/runtime/`。仍待 B 规定复核，不标团队 Done。

B 对 `31d89d5` 的工程复核为 REQUEST_CHANGES；本次修复路径可移植性与 response 显式会话错配门禁，等待新 HEAD 复审。B 本地缺少权重，权重内容及真实三模型加载仍为 `NOT_INDEPENDENTLY_VERIFIABLE`；A 的本地自检不替代独立验证。新增无权重跨目录回归测试，覆盖完整 T5 包的 11 个文件及嵌套关联哈希；完整数据跨目录复现也已执行。此次不重跑模型生成或改变选型结论。

B T6 可直接使用两组预测与 catalog；另外提供仅换情感、情感+画像、仅换 T4 回复三份待补/排除配置，`deferred_configs` 标明无组装预测、不可部署。0.65 情感候选需先补齐交付和实际加载检查。正式采用/发布须完成规定复核。全量验证集并非盲测，本文不作效果改善结论。具体自检日志仅保存在 `.codex_tmp/acceptance/cycle2/a_t5_20261030/`。
