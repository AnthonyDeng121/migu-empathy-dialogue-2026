# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本次加载自 `external/Qwen2.5-1.5B-Instruct`；设备 `cpu`（cpu，float32）；seed `20261013`。
环境：Python 3.12.12、torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：8/18653 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20；数据格式 `b1`。
token 长度（prompt+回复）：min 485 / mean 1046.6 / max 2039（max_length 2048）。
回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为邻接/上下文问题，无金标不一致）。

测速：1 个 optimizer steps，其中 0 个计入平均（前 3 个 warm-up step 不计）；每步为一个完整梯度累积窗口（batch 1 × 累积 8），计时两端同步设备；404.4706 秒/step（wall-clock fallback: run shorter than the warm-up skip）。
全量 2332 steps/epoch × 1 epoch，估算 943225 秒（+30% 故障余量 1226193 秒）；small-sample extrapolation, budget reference only; workplan requires a +30% failure margin on top of the raw estimate。

检查：1 steps；first loss 3.076147；梯度有限非零=True（范数 1.27453041）；LoRA 权重变化=False（392 个张量）；离线重载一致=True；生成探针 1/1 非空。
重载方式：offline PeftModel.from_pretrained on the saved adapter; all LoRA tensors compared and greedy generations reproduced；比对 392/392 个 LoRA 张量（weights_match=True），1 条确定性生成复现（generation_match=True）。
- `pp:["train_conv_000001",9,1]`：“站不稳”这个词很贴切：当关系变了，人就像站在摇晃的船上，明明知道船在晃，却总忍不住想找个地方坐稳，哪怕只是暂时的。  这种感觉，的确像是在提醒你：“别太在意这个变化，别把自己看得太重。”可事实呢，当你发现周围人都在重新调整位置，你也得跟着…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train 'external/response_train.jsonl' \
  --config 'configs/response_sft.json' \
  --device cpu \
  --output-dir 'external/response_sft_cpu_smoke' \
  --report-json 'external/d_day3_cpu_smoke.json' \
  --report-md 'external/D_DAY3_CPU_SMOKE.md' \
  --train-format b1 \
  --condition-jsonl 'external/classification_train.jsonl' \
  --model-dir 'external/Qwen2.5-1.5B-Instruct' \
  --max-train-samples 8 \
  --max-steps 2 \
  --eval-prompts 1
```

运行代码：commit `6d187bceac004a84d1f8db5ad651e1dedfe5a89f`（dirty=False）；脚本 SHA-256 `37a8e567539b4ee09b6060b014fa1279339c92e57bb8ad4726bd91dc777991a8`。

权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录（本报告不声称运行是否计入平台卡时，以平台账单为准）。
