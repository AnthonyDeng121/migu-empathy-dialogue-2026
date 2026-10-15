# D 岗位 Day 3（10/15）：baseline 镜像/加载支持——回复候选离线自包含与可复现性验证

**状态：Integration Gate。** 10/15 平台 GPU 额度已用尽（用户确认），首轮后续动作全部依赖 GPU 的工作今天无法启动；本报告交付 10/15 排班里 D 的"支持镜像/加载"一项：新增 `scripts/verify_response_artifact.py`，对 D2 回复候选（Qwen2.5-1.5B-Instruct + LoRA adapter）做**离线自包含 + 冷启动确定性 + 固定文件集可移植**三项门禁，全部实跑通过。命令、门禁结果与逐条预测均可独立复现。

## 为什么做这项

10/15 的 D 排班是"根据首轮结论只启动一个必要优化 2–3h；支持镜像/加载 1–2h"，当天交付要求"可提交 baseline……不能阻塞 baseline 发布"。GPU 不可用使第一项直接阻塞，第二项（支持 A 当天构建 baseline 正式镜像）成为唯一不依赖 GPU 的关键路径工作。

G3（10/15 18:00 可提交 baseline）与 G5（两次冷启动断网、预测哈希一致）都要求一个当时没有被验证过的性质：**交付的这堆文件在断网、换路径、只带必需文件的条件下，是否还能加载并给出同样的输出**。此前证据只覆盖"在训练机能跑通"，不覆盖"在镜像里能跑通"。

## 交付物

| 文件 | 作用 |
|---|---|
| `scripts/verify_response_artifact.py` | 门禁脚本；SHA-256 `2b3e0f304704e2db42f386277f40f79b3f09f18e65de6555cc53312771cf0141` |
| `tests/test_verify_response_artifact.py` | 18 个单测，覆盖 digest/离线环境/两次运行比对/命令构造/固定文件集拷贝；SHA-256 `de0eb15d72ca634057c430e97aa60c72c906fdb023067ebc985ae4970df3dfaa` |
| `configs/response_artifact.json` | 基座与 adapter 的字节指纹（目录 digest + 文件数 + 字节数 + adapter 单体 SHA-256）；SHA-256 `54c74c0511fa484d76b59b01773ddb84246b21335e4d264248bc71f384d6e2e2` |
| `reports/d_day3_artifact_verification.json` | 机器可读门禁结果（含三次运行的完整命令与性能报告） |
| `results/response_gen/artifact_verify_day3/run{1,2,3}.jsonl` + `*_report.json` | 三次运行的逐条预测与性能报告，可独立复核 |

## 门禁设计与实测结果

脚本对同一批 8 条冻结 B1 val 样本跑**三次**推理（复用正式入口 `scripts/run_response_inference.py`，不复制生成逻辑），并在三次运行前强制离线环境变量：

| 门禁 | 方法 | 结果 |
|---|---|---|
| 字节指纹一致 | 目录 digest（非隐藏文件的相对路径 + 逐文件 SHA-256）比对 `configs/response_artifact.json` | **PASS**：基座 7 文件 / 3,098,955,668 字节，digest `3dad5ec4…`；adapter 10 文件 / 52,866,512 字节，digest `af4f0e38…`，两侧均与固定值一致 |
| 离线自包含 | 运行前设 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`（三者运行前均未设置，已由报告 `offline_env._overridden` 记录），推理入口本身用 `local_files_only=True` | **PASS**：三次运行全部加载成功，无网络访问 |
| 冷启动确定性 | 同一路径连续两次独立进程运行，逐条预测必须字节相同 | **PASS**：8/8 完全一致（G5 门禁） |
| 固定文件集可移植 | 把 digest 覆盖的**非隐藏文件**拷到全新路径（`/tmp/…/staged_model`、`staged_adapter`，即镜像里的实际形态），第三次运行结果必须与第一次字节相同 | **PASS**：8/8 完全一致 |
| 输出非空 | 空回复计数 | **PASS**：0 空回复 |

诊断项（不作为门禁）：与 10/14 远程 4090 记录的全量 val 预测比对，8 条中 **1 条字节相同**，mean 字符 BLEU-4 **0.4808**、ROUGE-L F1 **0.5811**。语义高度接近但措辞漂移，属预期，见"发现"第 2 条。

三次运行耗时（本机 RTX 4060 8GB，cuda，bfloat16，max_new_tokens 256）：

| 运行 | 模型路径 | 总耗时 | 秒/条 | tok/s |
|---|---|---|---:|---:|
| 1 | `weights/hf/Qwen2.5-1.5B-Instruct` | 57.594s | 7.1992 | 17.61 |
| 2 | 同上（冷启动重复） | 66.431s | 8.3039 | 15.26 |
| 3 | `/tmp/…/staged_model`（仅固定文件集） | 82.029s | 10.2536 | 12.36 |

对照：10/14 远程 4090 全量 val 为 3.757 秒/条、22.4 tok/s。本机 4060 三次运行 7.20–10.25 秒/条（均值 8.59），约为远程 4090 的 1/1.9–1/2.7 速度。运行 3 的模型路径是脚本内的临时目录（运行结束即删除），复现时按上文命令执行即可，脚本会自动重建该目录。

## 复现命令

```bash
python -m unittest discover -s tests -v          # 139/139 PASS
python scripts/ci_smoke.py                       # PASS（入口/契约/可重复性）

python scripts/verify_response_artifact.py \
  --model weights/hf/Qwen2.5-1.5B-Instruct \
  --adapter weights/response_sft_d2_1p5b \
  --samples 8 --max-new-tokens 256 --device auto \
  --report reports/d_day3_artifact_verification.json \
  --keep-runs results/response_gen/artifact_verify_day3
```

运行代码即本 PR 新增的脚本/单测/配置，以本文上方三个 SHA-256 为身份（提交前后逐字节一致，已核验）；本 PR 的 head commit 见 PR 页面。测试环境：Python 3.12、torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2；`python -m unittest discover -s tests -v` 实测 139/139（新增 18，原 121 未回归）。本机 RTX 4060 8GB 属轻量冒烟，不计平台卡时，平台 GPU 卡时消耗 0。

## 发现

1. **adapter 的 `base_model_name_or_path` 指向远程训练机路径**（`/root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct`），本机与镜像中都不存在。已实测**无害**：`PeftModel.from_pretrained(model, adapter)` 以显式传入的 model 为准，第三次运行从全新路径加载并给出字节相同输出即为证据。但该字段被计入目录 digest，因此 A 侧副本若改写这个路径，必须重新固定 `configs/response_artifact.json` 的 adapter digest，否则门禁会报 mismatch（这是门禁的预期行为，不是缺陷）。
2. **跨 GPU 不保证字节一致。** 本机 4060 与远程 4090 的输出共享一段公共前缀后开始漂移（样例：`pp:["val_conv_000001",1,1]` 前 57 字符相同后分叉），bf16 贪心解码在不同 GPU 架构上会因近似持平 logit 的 argmax 翻转而放大差异。**含义**：10/14 记录的全量 val 诊断指标（BLEU-4 0.1075 / ROUGE-L 0.2542）是在 4090 上测得的估计值，镜像在平台硬件上跑出的字节不会与之相同；G5 要求的是同一环境内两次冷启动一致，这一条已通过。请勿把"与记录文件字节相同"当作交付标准。
3. **vLLM 路径未验证。** 本机未安装 vLLM（`ModuleNotFoundError: No module named 'vllm'`），大赛专属镜像 `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8` 内的 vLLM 能否直接挂载该 LoRA adapter，需 A 在镜像内实测。transformers 路径已验证可用（本机 5.17.0，远程 4.57.6，两版本下 chat template 渲染与 tokenize 已由 10/14 报告交叉核验一致）。
4. **retrieval 全量 val 预测其实已经可 join，无需补跑。** B2 报告（`reports/B_DAY3_B2_EVAL.md`）把"D retrieval full prediction JSONL"列为尚缺，实际 `results/baseline_val/submission.jsonl` 已有 2,354 行，只是键为官方 schema 的 `id`（`eval_000001_t1`，仅全局行号 + 目标轮）。用 `data/processed/val_eval_input.jsonl` 的 `(conversation_id, target_user_turn_id)` 做桥接即可对上 B1 的 `sample_id`（`pp:["val_conv_000001",1,1]`）。已实测 **2,354/2,354 可 join、0 缺失、0 空回复**，三路对比可完整 replay，不需要 GPU 也不需要重跑。这属于 B 的评测产物，未由 D 代生成文件，桥接方法记录在此供 B 直接使用。

## 权重交接

10/14 A 向 D 索取 LoRA adapter，与 C 的模型一起在云端训练。已交付：adapter 权重（36,981,072 字节，SHA-256 `62a695ea594a1546e217ee5edd691131051a4e64e3ecfcec91af75de7a133fcf`）、val 全量 2,354 条预测、训练报告与日志、交付说明（哈希/基座/运行要素）。本机的 `configs/response_artifact.json` 指纹即按该交付副本固定，A 侧可用同一脚本核验收到的文件未被改动。

## 阻塞与下一步（均需 GPU 或 C 侧产物）

| 事项 | 阻塞原因 | 解除后动作 |
|---|---|---|
| 原模型全量同集臂补跑 | 平台 GPU 今日用尽；10/14 已跑约 87% 被机器移交终止 | 补跑命令见 `reports/D_DAY3_RESPONSE_LORA_SFT.md`；约 2.46 GPU h |
| C2 条件臂复跑（`--condition-jsonl`） | 等 C 的 C2 权重；C 的条件键须与 B1 `sample_id` 对齐 | 键不一致时缺失检查会直接报错拒绝运行，不会静默混合条件/无条件提示 |
| D3 一轮有假设的优化 | 平台 GPU 今日用尽 | 按首轮结论只启动一个必要优化，受卡时预算约束 |
| vLLM 挂载验证 | 本机无 vLLM | A 在专属镜像内实测，D 提供支持 |
