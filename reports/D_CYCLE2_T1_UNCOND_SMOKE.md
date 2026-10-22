# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `weights/hf/Qwen2.5-1.5B-Instruct`；设备 `cuda`（NVIDIA GeForce RTX 4060 Laptop GPU，bfloat16）；seed `20261013`。
环境：Python 3.12.3、torch 2.5.1+cu121、transformers 5.17.0、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：8/18653 个 SFT 样本（超长跳过 0）；条件注入 `none`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 456 / mean 1075.4 / max 2042（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：2 个 optimizer steps，其中 0 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；12.0927 秒/step（wall-clock fallback: run shorter than the warm-up skip）。
全量 2332 steps/epoch × 2 epoch，估算 56401 秒（+30% 故障余量 73321 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.71 GiB。

检查：2 steps；first loss 3.120735；梯度有限非零=True（范数 1.31778014）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：嗯，这种感觉挺真实的：当别人对你冷淡，你会不由自主地怀疑自己在这个圈子中的位置，甚至担心自己会不会也被甩走。  这种感觉的确很不舒服，因为你在乎的不只是自己，还有周围的人怎么看自己。但换个角度看，也是个提醒：你在这里，别人也在，彼此之间都有…
- `pp:["train_conv_000001",11,2]`：真怕被整个圈子排除在外，那才是最可怕的。因为这不仅是个人情绪的问题，也是社交网络的复杂性：一个人的情绪波动，可能很快就被放大到影响整个群体。  现在他俩的关系似乎有了些缓和，但你还是担心自己会被边缘化，这种恐惧感就像一把刀，每次触碰到都会疼…
- `pp:["train_conv_000001",15,3]`：明白了，那就顺其自然吧。  不管最后结果如何，至少现在能给自己一个选择：既不纠结于“到底算不算得上”，也不太在乎“具体站在哪儿”。只要能让自己舒服，就挺好。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/processed/b1_full_v1/response_train.jsonl' \
  --config 'configs/response_sft_unconditional_2ep.json' \
  --device cuda \
  --output-dir 'results/tmp_uncond_smoke' \
  --report-json 'reports/d_cycle2_t1_uncond_smoke.json' \
  --report-md 'reports/D_CYCLE2_T1_UNCOND_SMOKE.md' \
  --train-format b1 \
  --model-dir 'weights/hf/Qwen2.5-1.5B-Instruct' \
  --max-train-samples 8 \
  --max-steps 2
```

运行代码：commit `6c44c689952743ecbc78439d233a9527fea1b1cc`（dirty=True）；脚本 SHA-256 `bb501a20e27cc75f612e97ce40d5d6899dfd310b26e75378de8e1cc954929d4c`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
