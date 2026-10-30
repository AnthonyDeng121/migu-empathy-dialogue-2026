# D 岗位回复 SFT 训练运行报告

**状态：candidate training。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `weights/base_qwen15b`；设备 `cuda`（NVIDIA GeForce RTX 4090，bfloat16）；seed `20261013`。
环境：Python 3.12.13、torch 2.11.0+cu128、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：18653/18653 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 116 / mean 902.0 / max 2048（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：4664 个 optimizer steps，其中 4661 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；1.6039 秒/step（mean full gradient-accumulation window after warm-up）。
全量 2332 steps/epoch × 2 epoch，估算 7481 秒（+30% 故障余量 9725 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。
显存峰值：6.72 GiB。

检查：4664 steps；first loss 2.969118；梯度有限非零=True（范数 1.07209659）；LoRA 权重变化=True（392 个张量）；离线重载一致=True；生成探针 3/3 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），3 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：嗯，那就先不分析。  那种别扭感确实挺扎心的：明明只是一个人的事，却像是连带着整个圈子都晃了一下。别人可能还在热闹里，你脑子里却已经在替他们打分了。  不用急着把它理顺，也不用现在就想好怎么办。就让它在那里待一会儿就行。…
- `pp:["train_conv_000001",11,2]`：那个“怕被整个圈子排除”的念头，比他态度稍微缓过来更扎人。因为面子通常不会只针对一个人，它像是把别人的情绪也一起扣到了你头上。  现在群里的正常互动，可能只是暂时没碰触到最核心的那个害怕——你担心的不是某一次冷淡，而是“如果我不再主动，我就…
- `pp:["train_conv_000001",15,3]`：那就先按群里现在这样正常来往，不用再专门给这个位置下结论。  越盯越空的时候，先顺着过比硬扛着“必须看清”更省力。圈子里的位置感本来就不是靠一个人表态能定下来的，等你心里不再只盯着这一条线时，那些念头自然也就轻了。…
周期存档：每 250 optimizer step 一次，本次运行共保存 18 个（weights/t4_longweight_2ep/checkpoints/）。
每个 checkpoint 附带 manifest.json（实验身份：训练数据 SHA-256 `8ef2a5dcfe7c…`、条件文件 `4d4e949813ac`、配置、基座 7 个文件的内容指纹、预算 epochs=2/steps=4664）；恢复时逐项比对，任何不一致即拒绝续跑。

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'data/processed/t4_long_response/response_train.jsonl' \
  --config 'configs/response_sft_t4_longweight.json' \
  --device cuda \
  --output-dir 'weights/t4_longweight_2ep' \
  --report-json 'reports/t4_longweight_2ep_report.json' \
  --report-md 'reports/t4_longweight_2ep_report.md' \
  --train-format b1 \
  --condition-jsonl 'data/processed/b1_full_v1/classification_train.jsonl' \
  --model-dir 'weights/base_qwen15b' \
  --checkpoint-every 250 \
  --local-files-only
```

运行代码：commit `07a2f2e0beff4a57a9730d8019a0f75072309872`（dirty=True）；脚本 SHA-256 `9f67ff4143b6f781fb3129b828982cc29e67fab35c6d0f13cbf3e06f43fab5a1`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
