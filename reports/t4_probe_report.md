# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `weights/base_qwen15b`；设备 `cuda`（NVIDIA GeForce RTX 4090，bfloat16）；seed `20261013`。
环境：Python 3.12.13、torch 2.11.0+cu128、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：18653/18653 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 116 / mean 902.0 / max 2048（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：120 个 optimizer steps，其中 117 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；1.6254 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2332 steps/epoch × 2 epoch，估算 7581 秒（+30% 故障余量 9855 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：120 steps；first loss 2.969118；梯度有限非零=True（范数 1.07217383）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 2/2 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），2 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：好，那就先别听建议，也不用分析。  这种感觉很真实：当关系开始走下坡，人就像站在一个摇晃的平台上，每一步都像是在试探着看能不能稳住脚跟。  你不需要立刻决定要不要重新靠近，也不需要马上给这个平台贴个“站不稳”的标签。现在，就让它在那里晃着，…
- `pp:["train_conv_000001",11,2]`：这种害怕，比他态度是否变好，更接近于一种“被边缘化”的恐惧：不是怕他不再理你，而是怕他不再理你之后，你也跟着被边缘化。  这种边缘感，往往不是因为某个人突然变了，而是因为你在某个节点后，发现自己成了那个“谁都不理”的人。这种落差感，比单纯的…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/processed/t4_long_response/response_train.jsonl' \
  --config 'configs/response_sft_t4_longweight.json' \
  --device cuda \
  --output-dir 'weights/t4_probe' \
  --report-json 'reports/t4_probe_report.json' \
  --report-md 'reports/t4_probe_report.md' \
  --train-format b1 \
  --condition-jsonl 'data/processed/b1_full_v1/classification_train.jsonl' \
  --model-dir 'weights/base_qwen15b' \
  --max-steps 120 \
  --eval-prompts 2 \
  --local-files-only
```

运行代码：commit `f9f2fefa7249e86a78c4d20f7ddf77720d2cb528`（dirty=True）；脚本 SHA-256 `9f67ff4143b6f781fb3129b828982cc29e67fab35c6d0f13cbf3e06f43fab5a1`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
