# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，revision `7ae557604adf67be50417f59c2c2f167def9a775`，本次加载自 `weights/hf/Qwen2.5-0.5B-Instruct`；设备 `cpu`（cpu，float32）；seed `20261013`。
环境：Python 3.12.3、torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 4,399,104 （0.88% of 498,431,872）。
样本：16/18657 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 4。
token 长度（prompt+回复）：min 176 / mean 427.3 / max 512（max_length 512）。
回复 turn 与目标一致性：18657/18657 与转写正文一致。

测速：16 个 optimizer steps，其中 13 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 2），计时两端同步设备；7.7890 秒/step（mean full gradient-accumulation window after warm-up）。
全量 9329 steps/epoch × 2 epoch，估算 145327 秒（+30% 故障余量 188925 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。

检查：16 steps；first loss 3.579902；梯度有限非零=True（范数 2.66280484）；LoRA 权重变化=True（336 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 336/336 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `train_conv_000001_t9`：你这么说，我倒觉得有点道理。毕竟，冷淡和僵局都是人生的一部分，它们都有其存在的价值。  现在不妨先放一下，…
- `train_conv_000001_t11`：理解你的感受，但别太执着。  你确实觉得害怕的是，自己跟别人的关系可能真的已经变得不那么重要了。不过，别忘了…
- `train_conv_000001_t15`：按群里的现状正常来往，心里那块发虚也自然能自消。不过，你要是真的觉得圈里太空，那就别…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/raw/训练-验证-数据集/train/train_public.jsonl' \
  --config 'configs/response_sft_tiny_cpu.json' \
  --device cpu \
  --output-dir 'weights/response_sft_epochloop_cpu' \
  --report-json 'reports/d_day2_epochloop_cpu.json' \
  --report-md 'reports/D_DAY2_EPOCHLOOP_CPU.md' \
  --model-dir 'weights/hf/Qwen2.5-0.5B-Instruct' \
  --max-train-samples 16
```

运行代码：commit `bcba020dcd49c234b3deab8024df95ceb6670a5c`（dirty=True）；脚本 SHA-256 `c5dfd750567e1b399021c5b9766189e1017dab5cfff989943d0540d014a6196a`。

权重只保存在本机忽略目录，不提交 Git；本机运行不计平台 GPU 卡时。
