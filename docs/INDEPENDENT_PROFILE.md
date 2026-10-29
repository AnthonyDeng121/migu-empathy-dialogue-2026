# 独立画像接入

集成入口支持将情感模型与画像模型分开选择。默认 `configs/integrated.json` 继续使用 C Day5 情感和空画像；本改动提供接入能力，不代表任何画像模型已通过效果评测或获准采用。

## 配置

复制当前集成配置，只修改以下字段，其他情感、回复和生成配置保持原值：

```json
{
  "profile_mode": "independent",
  "profile_model_dir": "weights/your_packaged_profile_checkpoint"
}
```

这只是配置片段；目录必须替换为实际交付的 checkpoint。相对路径以项目根目录为基准，与现有模型目录规则一致。

运行完整配置：

```bash
python scripts/run_integrated.py TEST_FILE RESULT_DIR --config YOUR_CONFIG.json
```

模式说明：

| profile_mode | 情感来源 | 画像来源 |
|---|---|---|
| empty | classifier_dir | 三个空数组，无需画像权重 |
| model（历史默认） | classifier_dir | 同一 joint/packaged 模型；不支持 sequence |
| independent | classifier_dir | profile_model_dir，单独加载 packaged checkpoint |

sequence 必须显式选择 empty 或 independent。独立模型只贡献 `user_profile`，其情感预测和其他字段不会覆盖主分类结果。`condition_response=false` 时仍可输出画像，但不会向 D 注入分类条件。

## T3：正式入口的条件范围

| condition_response | condition_fields | 给 D 的条件 |
|---|---|---|
| false | emotion 或 both | 不注入 |
| true | emotion | 只注入预测情感，完全省略画像字段 |
| true | both | 注入预测情感及画像 |

`condition_fields` 缺省为 `both`，旧配置行为保持兼容；非法值即使在关闭条件时也报错。条件对象只含所选预测字段，不包含 ID、验证真值或未来轮次。输出画像由 `profile_mode` 独立决定，三种条件策略均不清空输出画像。生成参数保持原值。报告的 `condition_mode` 沿用 predicted/none，新增 `condition_fields` 记录实际注入范围（关闭时为 none）；恢复指纹包含配置，切换策略必须使用新的恢复目录。

C v1 已发布于 [Release](https://github.com/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tag/c-cycle2-profile-v1-20261026)。下载 ZIP 及 SHA256SUMS.txt，校验包 SHA-256 `761761f619643fd8209d309d0d4cbc684d1aa2374c9542c26283ba961e514060` 后解压到项目根目录。

三个完整实验配置：`configs/integrated_profile_v1_off.json`、`configs/integrated_profile_v1_emotion.json`、`configs/integrated_profile_v1_both.json`。它们固定 C Day5、D Day4、画像 v1 和 256 token 上限，仅改变条件范围；原 `integrated.json` 仍为默认入口。配置采用原服务器模型路径和 CUDA，运行前须准备既有 C/D 资产。这些是实验配置，不代表已批准采用画像或条件。

```bash
python scripts/run_integrated.py TEST_FILE RESULT_DIR --config configs/integrated_profile_v1_emotion.json
```

## 支持的交接格式

本次复用 `EmotionProfileCheckpoint` 的已有 packaged 格式：

- 根目录包含 `checkpoint_metadata.json`、`heads.safetensors` 和 tokenizer 文件。
- `encoder/` 包含本地编码器配置及权重。
- metadata 包含 `emotion_labels`、三维 `profile_labels`、三维数值 `profile_thresholds` 和正整数 `max_length`。
- 标签数组的顺序对应权重头的输出索引；接受合法的不同顺序，但拒绝缺失、额外或重复标签。
- 每维阈值必须为 [0, 1] 内的有限数值。逐标签阈值等新格式需要另行版本化接入，不自动猜测。

沿用原 loader 的历史预处理：按顺序拼接全部已提供的 history，使用“用户：/助手：”前缀，左侧截断，长度取 metadata。C 交付时需要确认训练预处理与此约定一致；现有 metadata 不足以自动证明这一点。

入口先预检 metadata 和关键文件，随后由原离线 loader 检查 tokenizer、权重形状和头参数。所有加载均使用本地文件。独立模式失败直接抛出错误，不降级为空画像；画像输出在构造 D 提示前通过官方 schema 校验。

## 验证与边界

```bash
python -m unittest tests.test_integrated tests.test_independent_profile -v
python -m unittest discover -s tests -q
```

假模型测试覆盖来源隔离、默认空画像、加载失败传播、非法输出、metadata/文件预检，以及真实 `Models.predict()` 方法在关闭条件时仍输出独立画像且保持固定解码参数。

上述假模型单测不证明模型质量或运行速度。T3 已另行完成画像 v1 的真实 CPU 加载和 8 条交付预测一致性检查，见 `reports/A_CYCLE2_T3_INTEGRATION.md`；未运行真实 D 生成或正式端到端计时。三模型暂存、Docker 和打包支持属于 T6。
