# D 岗位回复 SFT 训练运行报告

**状态：candidate training。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `external/Qwen2.5-1.5B-Instruct`；设备 `cuda`（NVIDIA GeForce RTX 4090，bfloat16）；seed `20261013`。
环境：Python 3.12.12、torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：18653/18653 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 116 / mean 902.0 / max 2048（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：4664 个 optimizer steps，其中 4661 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；1.7514 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2332 steps/epoch × 2 epoch，估算 8168 秒（+30% 故障余量 10619 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：4664 steps；first loss 2.969118；梯度有限非零=True（范数 0.87996262）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：嗯，那就先不分析。  那种别扭感确实挺扎心的：明明只是一个人的事，却像是连带着整个圈子都晃了一下。别人可能还在热闹里，你脑子里却已经在替他们打分了。  不用急着把它理顺，也不用强迫自己立刻好起来。这里安静一会儿就好。…
- `pp:["train_conv_000001",11,2]`：那个“怕被整个圈子排除”的念头，比他态度稍微缓过来一点点更扎人。因为面子通常不会只针对一个人，它像是把大家的目光都集中到你身上，让你觉得自己的位置可能随时会被重新打量一遍。  群里的正常互动，往往只是暂时没有直接冲突，不代表关系还在走回头路…
- `pp:["train_conv_000001",15,3]`：那就先按群里现在这样正常来往，不用再给自己加一层“必须谈清楚”的任务。  空落落的那部分先放着，不靠紧也不靠远，等你想回头时再说。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'external/response_train.jsonl' \
  --config 'configs/response_sft_2epoch.json' \
  --device cuda \
  --output-dir 'external/response_sft_d4_2ep' \
  --report-json 'external/d_day4_sft_full_2ep.json' \
  --report-md 'external/D_DAY4_SFT_FULL_2EP.md' \
  --train-format b1 \
  --condition-jsonl 'external/classification_train.jsonl' \
  --model-dir 'external/Qwen2.5-1.5B-Instruct'
```

运行代码：commit `None`（dirty=None）；脚本 SHA-256 `bb501a20e27cc75f612e97ce40d5d6899dfd310b26e75378de8e1cc954929d4c`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
