# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；设备 `cuda`（bfloat16）；seed `20261013`。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 9,232,384 （0.59% of 1,552,946,688）。
样本：512/18657 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20。
token 长度（prompt+回复）：min 141 / mean 1049.9 / max 2047（max_length 2048）。
回复 turn 与目标一致性：18657/18657 与转写正文一致。

测速：50 optimizer steps，1.1756 秒/step；全量 2333 steps/epoch，估算 2743 秒/epoch（未含 workplan 要求的 +30% 故障余量）。
显存峰值：6.72 GiB。

冒烟检查：50 steps；first loss 3.075108；梯度有限非零=True；LoRA 权重变化=True；adapter 重载一致=True；生成探针 3/3 非空。
- `train_conv_000001_t9`：好，那就先别听建议，直接把那个别扭感拿回去。  “站不稳”是一种情绪，比具体行动更实在。现在不需要讨论策略，只需要确认一下：这种“站不稳”是你最真实的感受，而不是因为别人说了什么、做了什么。  等你觉着舒服了，再来问下一步。…
- `train_conv_000001_t11`：这种卡住，是因为害怕被整体边缘化：不是怕他不再理你，而是怕你被这个圈子里的其他人一起疏远。  你之前说过的那个“站不稳”，不只是因为“被冷淡”，还有“被排挤”。这种恐惧比直接的冷淡更复杂，因为它不是某个人的决定，而是你被集体看走眼。  现在…
- `train_conv_000001_t15`：好，按群里现在这样正常来往就行。  既不指望能一下子解决，也不太想再把注意力往那个方向拽。先顺着走，等时机合适再说。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py --train 'data/raw/训练-验证-数据集/train/train_public.jsonl' \
  --config configs/response_sft.json --device cuda \
  --output-dir weights/response_sft_smoke_1p5b \
  --report-json reports/response_sft_run.json --report-md reports/RESPONSE_SFT_RUN.md
```

权重只保存在本机忽略目录，不提交 Git；本机运行不计平台 GPU 卡时。
