# D 岗位回复 SFT 训练运行报告

**状态：candidate training。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `external/Qwen2.5-1.5B-Instruct`；设备 `cuda`（NVIDIA GeForce RTX 4090，bfloat16）；seed `20261013`。
环境：Python 3.12.12、torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：18653/18653 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 116 / mean 902.0 / max 2048（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：2332 个 optimizer steps，其中 2329 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；1.8094 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2332 steps/epoch × 1 epoch，估算 4219 秒（+30% 故障余量 5485 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：2332 steps；first loss 2.969118；梯度有限非零=True（范数 0.88015127）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：嗯，我不给建议了。  那种别扭感确实很难受：明明只是关系变了，却像是连自己在这群人里的位置都动摇了。它不像具体的事那样能立刻解决，却比事情本身更消耗人。  你就按你的节奏待着就行。…
- `pp:["train_conv_000001",11,2]`：那个“怕被整个圈子排除在外”的念头，比单纯的一次冷淡更扎心。它像是在提醒你：不只是他变了，你也可能因为这次反应，被暂时抛到外面看不清的位置。  现在群里的正常互动，更像是他在试探边界，而不是在确认关系。你担心的更多是自己会不会因此失去连接感…
- `pp:["train_conv_000001",15,3]`：这个决定挺稳的：先按群里现在的节奏走，不专门挑茬，也不再往那个位置靠。空一下也好，至少不用一直盯着自己到底算不算得上。  等你想再聊时再说就好。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'external/response_train.jsonl' \
  --config 'configs/response_sft.json' \
  --device cuda \
  --output-dir 'external/response_sft_d2_1p5b' \
  --report-json 'external/d_day3_sft_full_1p5b.json' \
  --report-md 'external/D_DAY3_SFT_FULL_1P5B.md' \
  --train-format b1 \
  --condition-jsonl 'external/classification_train.jsonl' \
  --model-dir 'external/Qwen2.5-1.5B-Instruct'
```

运行代码：commit `6761204182134153b24581021935ed9323ca60fc`（dirty=False）；脚本 SHA-256 `37a8e567539b4ee09b6060b014fa1279339c92e57bb8ad4726bd91dc777991a8`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
