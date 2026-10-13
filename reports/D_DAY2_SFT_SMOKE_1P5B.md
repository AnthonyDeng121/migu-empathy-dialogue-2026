# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `weights/hf/Qwen2.5-1.5B-Instruct`；设备 `cuda`（NVIDIA GeForce RTX 4060 Laptop GPU，bfloat16）；seed `20261013`。
环境：Python 3.12.3、torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：512/18657 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20。
token 长度（prompt+回复）：min 141 / mean 1049.9 / max 2047（max_length 2048）。
回复 turn 与目标一致性：18657/18657 与转写正文一致。

测速：53 个 optimizer steps，其中 50 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；12.1061 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2333 steps/epoch × 1 epoch，估算 28243 秒（+30% 故障余量 36717 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：53 steps；first loss 2.411758；梯度有限非零=True（范数 1.1483463）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `train_conv_000001_t9`：这种感觉就像站在冰面上，每一步都小心翼翼，生怕一不小心就滑倒。你不需要立刻知道下一步该怎么走，只需要知道现在脚底踏实就行。  既然你不想听建议，那就先让自己舒服一些。如果这次能暂时不去想“冷淡”背后的具体问题，哪怕只是“我在这圈子里有点站不…
- `train_conv_000001_t11`：这种怕，比他俩的关系缓和，更像是一种“被边缘化”的预感。  “他态度没再冷”不代表他愿意多留，可能只是暂时没意识到“现在不说话就太冷了”。可你心里的“被排挤”感，却像是在等他这句话：“我们俩的关系缓和了，但圈子还在，你呢？”如果他没说，你就…
- `train_conv_000001_t15`：按群里现在这样正常来往，挺好。既不主动，也不被动地盯着那个位置，能省下不少力气。  既然心里那点发虚不自动消，就先顺其自然，不强求自己总能在群里找到一个明确位置。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/raw/训练-验证-数据集/train/train_public.jsonl' \
  --config 'configs/response_sft.json' \
  --device cuda \
  --output-dir 'weights/response_sft_smoke_1p5b_v2' \
  --report-json 'reports/d_day2_sft_smoke_1p5b.json' \
  --report-md 'reports/D_DAY2_SFT_SMOKE_1P5B.md' \
  --model-dir 'weights/hf/Qwen2.5-1.5B-Instruct' \
  --max-train-samples 512 \
  --max-steps 53
```

运行代码：commit `8af6b9322e1e92996188653c5e587c806bbf4d0e`（dirty=True）；脚本 SHA-256 `c5dfd750567e1b399021c5b9766189e1017dab5cfff989943d0540d014a6196a`。

权重只保存在本机忽略目录，不提交 Git；本机运行不计平台 GPU 卡时。
