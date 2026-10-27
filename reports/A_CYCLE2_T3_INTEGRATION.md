# A Cycle2 T3：真实画像 v1 接入与条件范围

日期：2026-10-27（Asia/Shanghai）。状态：实现及本地自检完成，Integration Gate，待 B 审查与合并。

## 交付

基于 main `9d1cc05`，沿用固定分支 `feature/a-mlops`。正式集成入口增加 `condition_fields=emotion/both`，与原布尔 `condition_response` 组成关闭、仅预测情感、预测情感＋画像三种策略。缺省 both 保持旧配置语义。仅情感路径完全省略画像提示；不注入条件时仍保留模型输出画像。条件对象不携带 ID 或其他无关字段。原默认配置、C Day5 情感来源、D Day4 与 greedy/256 token 参数不变。

提供 `configs/integrated_profile_v1_{off,emotion,both}.json` 三个实验配置；报告新增实际 `condition_fields`。恢复校验沿用完整配置指纹，改变条件时不得共用旧恢复目录。使用说明见 `docs/INDEPENDENT_PROFILE.md`。

## 真实资产核对

- C [v1 Release](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-cycle2-profile-v1-20261026) 的 ZIP 已下载并解压至 `weights/profile_encoder_v1_b1_20261026/`，不进入 Git。
- ZIP 大小 379,744,801 字节；本地实算 SHA-256 `761761f619643fd8209d309d0d4cbc684d1aa2374c9542c26283ba961e514060`，与 C 报告及 Release 一致；ZIP CRC 全通过。
- 使用真实 `EmotionProfileCheckpoint` 离线 CPU 重载：PyTorch 2.5.1+cu121、Transformers 5.14.1。max_length=256、左截断，三维阈值为 0.64 / 0.81 / 0.56。
- 从 C 逐条交付预测中按文件顺序各取前 4 条空画像和非空画像，用官方 val_public 对应 conversation 与 target_user_turn_id 构造 history，仅含目标轮及以前信息。8/8 画像与 C 交付结果完全一致。
- 训练使用的 B1 画像上下文格式与 loader 均为按轮次拼接“用户：/助手：”，检查 predict_contexts 与训练同格式 predict_texts 一致。
- 8 条真实画像均经过正式 `Models.apply_profile`，保留主情感和 ID，输出通过官方 schema；三种实际 system_prompt 范围均符合配置。

## 验证与限制

- unittest 全套 327 项，4 项跳过，其余通过。正式 Models.predict 的三种分支使用假 D 验证，生成上限及确定性解码保持不变；非法条件字段在加载前报错。
- compileall 与 `scripts/ci_smoke.py` 均通过。
- 本地验收脚本、真实模型检查 JSON、测试日志位于忽略目录 `.codex_tmp/acceptance/cycle2/a_t3_20261027/`。
- 未运行真实 D 生成、公共 GPU、远程部署或正式端到端计时；本次证明画像真实加载与条件工程行为，不宣称效果或速度提升。是否采用画像/条件仍由独立评测决定。三模型打包属 T6。

后续：由 B 审查 A 的实现；后续实验可直接使用固定 v1 配置，不等待 C v2 或 D 本期训练。
