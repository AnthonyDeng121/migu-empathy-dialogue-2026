# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `external/Qwen2.5-1.5B-Instruct`；设备 `cuda`（NVIDIA GeForce RTX 4090，bfloat16）；seed `20261013`。
环境：Python 3.12.12、torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：512/18653 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 141 / mean 1049.9 / max 2047（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：53 个 optimizer steps，其中 50 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；1.8153 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2332 steps/epoch × 1 epoch，估算 4233 秒（+30% 故障余量 5503 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：53 steps；first loss 2.416419；梯度有限非零=True（范数 1.14647448）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：这种感觉就像站在冰面上，每一步都小心翼翼，生怕一不小心就滑倒。你不需要立刻知道下一步该怎么走，只需要知道现在脚底踏实就行。  既然你不想听建议，那就先让自己舒服一些。如果这次能静下心来，哪怕只是闭眼深呼吸几秒，也是个好兆头。…
- `pp:["train_conv_000001",11,2]`：这种卡住，是因为害怕被整体边缘化：不是怕他不再搭理你，而是怕他搭不理你的时候，你也跟着被一起推到角落。  这种恐惧比单个冷淡更复杂：你不是单纯地“不搭理”，而是担心“不搭理”之后，连带的其他信号（比如沉默、不回应）都会一起落空，让你整个人都…
- `pp:["train_conv_000001",15,3]`：按群里现在这样正常来往，挺好。既不主动，也不被动地盯着位置，能省下不少力气。  既然心里那点发虚不自动消，就先顺着走，不用再一个劲儿往那上面靠。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'external/response_train.jsonl' \
  --config 'configs/response_sft.json' \
  --device cuda \
  --output-dir 'external/response_sft_smoke_1p5b' \
  --report-json 'external/d_day3_sft_smoke_1p5b.json' \
  --report-md 'external/D_DAY3_SFT_SMOKE_1P5B.md' \
  --train-format b1 \
  --condition-jsonl 'external/classification_train.jsonl' \
  --model-dir 'external/Qwen2.5-1.5B-Instruct' \
  --max-train-samples 512 \
  --max-steps 53
```

运行代码：commit `6761204182134153b24581021935ed9323ca60fc`（dirty=False）；脚本 SHA-256 `37a8e567539b4ee09b6060b014fa1279339c92e57bb8ad4726bd91dc777991a8`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
