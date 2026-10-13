# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，revision `7ae557604adf67be50417f59c2c2f167def9a775`，本次加载自 `weights/hf/Qwen2.5-0.5B-Instruct`；设备 `cuda`（NVIDIA GeForce RTX 4060 Laptop GPU，bfloat16）；seed `20261013`。
环境：Python 3.12.3、torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 4,399,104 （0.88% of 498,431,872）。
样本：512/18657 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20。
token 长度（prompt+回复）：min 141 / mean 1049.9 / max 2047（max_length 2048）。
回复 turn 与目标一致性：18657/18657 与转写正文一致。

测速：53 个 optimizer steps，其中 50 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；7.7031 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2333 steps/epoch × 1 epoch，估算 17971 秒（+30% 故障余量 23363 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：4.59 GiB。

检查：53 steps；first loss 2.876953；梯度有限非零=True（范数 2.38356972）；LoRA 权重变化=True（336 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 336/336 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `train_conv_000001_t9`：好，那就先放一下。现在最需要的是：  1. **不急**：给自己留个缓冲区，不强迫自己马上做出决定。 2. **不问结果**：暂时不去想“如果冷下来，我会变成什么样”。专注于当下，感受那份别扭。 3. **不强求**：给身体足够的时间来适应…
- `train_conv_000001_t11`：被排除在外，确实比冷淡更刺眼。但冷淡背后，可能还有其他原因：可能是对方突然变了，不再像以前那样亲热；可能是你自己的情绪波动，导致别人对你不够敏感；也可能是某种潜意识的“我太累了，需要有人陪护”……这些都不好说，也不一定能立即解决。  现在最…
- `train_conv_000001_t15`：按群里现在这么正常，你先顺着他走就行了。圈子里到底站哪儿，你先不问，不靠。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/raw/训练-验证-数据集/train/train_public.jsonl' \
  --config 'configs/response_sft_0p5b.json' \
  --device cuda \
  --output-dir 'weights/response_sft_smoke_0p5b_v2' \
  --report-json 'reports/d_day2_sft_smoke_0p5b.json' \
  --report-md 'reports/D_DAY2_SFT_SMOKE_0P5B.md' \
  --model-dir 'weights/hf/Qwen2.5-0.5B-Instruct' \
  --max-train-samples 512 \
  --max-steps 53
```

运行代码：commit `8af6b9322e1e92996188653c5e587c806bbf4d0e`（dirty=True）；脚本 SHA-256 `c5dfd750567e1b399021c5b9766189e1017dab5cfff989943d0540d014a6196a`。

权重只保存在本机忽略目录，不提交 Git；本机运行不计平台 GPU 卡时。
