# D 岗位回复 SFT 训练运行报告

**状态：candidate training。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `base_qwen15b`；设备 `cuda`（NVIDIA GeForce RTX 4090，bfloat16）；seed `20261013`。
环境：Python 3.11.15、torch 2.12.0+cu132、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：18653/18653 个 SFT 样本（超长跳过 0）；条件注入 `none`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 90 / mean 876.9 / max 2048（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：4664 个 optimizer steps，其中 4661 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；2.3622 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2332 steps/epoch × 2 epoch，估算 11018 秒（+30% 故障余量 14323 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：4664 steps；first loss 2.99967；梯度有限非零=True（范数 1.05975997）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：嗯，那就先不分析。  那种别扭感确实很具体：冷淡像是个锚，一落下去，周围那些原本能踩实的东西，好像也都晃了一下。人在圈子里面，本来就容易把自己和别人一起掂量一遍，冷淡一下，连这点安全感都会隐隐发虚。  你就这么待着也好，不用现在就把下一步定…
- `pp:["train_conv_000001",11,2]`：那个“怕被整个圈子排除”的念头，比他态度稍微缓过来一点点更扎人。因为面子通常不会只针对一个人，它像是把大家的目光都集中到你身上，让你觉得自己的位置可能正在往下滑。  群里的正常互动，往往只是暂时没有直接冲突，不代表关系还在走回头路。可人在孤…
- `pp:["train_conv_000001",15,3]`：那就先按群里现在这样正常来往，不用再给自己加一层“必须谈清楚”的任务。  空落落的那部分先放着，不靠紧也不靠远，等你想回头时再说。…
周期存档：每 250 optimizer step 一次，本次运行共保存 18 个（weights/t3_unconditional_2ep/checkpoints/）。

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/processed/b1_full_v1/response_train.jsonl' \
  --config 'configs/response_sft_unconditional_2ep.json' \
  --device cuda \
  --output-dir 'weights/t3_unconditional_2ep' \
  --report-json 'reports/t3_uncond_2ep_report.json' \
  --report-md 'reports/t3_uncond_2ep_report.md' \
  --train-format b1 \
  --model-dir 'base_qwen15b' \
  --checkpoint-every 250 \
  --local-files-only
```

运行代码：commit `None`（dirty=None）；脚本 SHA-256 `4fd9b7c7e2371269f939ca9e5547c908947bdd63ba18d09bfc1cf90668b38e1c`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
