# D 岗位回复 SFT 训练运行报告

**状态：smoke。**

模型：[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；设备 `cuda`（bfloat16）；seed `20261013`。
LoRA：r=8 alpha=16 dropout=0.05；可训练参数 4,399,104 （0.88% of 498,431,872）。
样本：512/18657 个 SFT 样本（超长跳过 0）；条件注入 `gold`；最大历史轮数 20。
token 长度（prompt+回复）：min 141 / mean 1049.9 / max 2047（max_length 2048）。
回复 turn 与目标一致性：18657/18657 与转写正文一致。

测速：50 optimizer steps，0.9259 秒/step；全量 2333 steps/epoch，估算 2160 秒/epoch（未含 workplan 要求的 +30% 故障余量）。
显存峰值：4.59 GiB。

冒烟检查：50 steps；first loss 3.124727；梯度有限非零=True；LoRA 权重变化=True；adapter 重载一致=True；生成探针 3/3 非空。
- `train_conv_000001_t9`：好，那就先放一下。现在最需要的是：  1. **暂时不去管**：既然觉得冷下来后容易觉得自己站不稳，就先放一放。 2. **不急着做决定**：如果觉得冷下来后能让自己站稳，就先不急着去找对方，等心情好了再说。 3. **不强求结果**：就算…
- `train_conv_000001_t11`：你这么说，我理解。现在他说话时，你没有冷到，但周围的人可能开始疏远你，你害怕自己站在边缘，因为没人能帮你重新站稳。  这种被排斥感，比单纯冷淡更重。你不想再主动找他，是因为你怕自己站不住，而不是因为他变了。你怕自己被完全孤立，而不是因为别人…
- `train_conv_000001_t15`：按群里现在这样正常来往，圈子里到底站哪儿这种事越盯越空，你先顺着他走，圈子里就先不空了。…

复现命令：

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py --train 'data/raw/训练-验证-数据集/train/train_public.jsonl' \
  --config configs/response_sft.json --device cuda \
  --output-dir weights/response_sft_smoke_0p5b \
  --report-json reports/response_sft_run.json --report-md reports/RESPONSE_SFT_RUN.md
```

权重只保存在本机忽略目录，不提交 Git；本机运行不计平台 GPU 卡时。
