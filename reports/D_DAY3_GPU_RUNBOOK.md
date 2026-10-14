# D2 首轮 LoRA/SFT GPU 运行手册（GPU 访问恢复后执行）

**状态（2026-10-14 12:50 +08）：远程实例 GPU 设备访问被平台收回（`/dev/nvidia0` 打开 EPERM、NVML `Unknown Error`、`torch.cuda.device_count()=0`），首轮训练未启动。本手册是恢复后的一键执行序列；环境、数据、代码均已就绪并验证。**

## 环境现状（已验证）

- 远程：`ssh root@183.222.230.12 -p 59249`，实例 2026-10-14 11:41 +08 创建（容器启动时 GPU 可用，11:50 起被收回）。
- 仓库：`/root/migu-d2/repo` @ commit `6d187bc`（feature/d-plan-sync-1014，含 .git，`git status` 干净）；后续 commit 用 `git fetch` + `git reset --hard origin/feature/d-plan-sync-1014` 同步（.git 已 scp 过去，origin 指向 GitHub）。
- 依赖：`/opt/conda/envs/conda_env/bin/python`（Python 3.12.12；torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2、accelerate 1.15.0、safetensors 0.7.0）。平台预置环境，勿升级。
- 数据：`/root/migu-d2/data/b1_full_v1/{response_train,response_val,classification_train}.jsonl`，SHA-256 与本地 B1 v1 一致（response_train `a70ca036…`、response_val `ddf76588…`、classification_train `4d4e9498…`）。
- 模型：`/root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct`（revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，经 hf-mirror 下载，3.09GB）。
- CPU 端到端冒烟已通过：`reports/d_day3_response_sft_cpu_smoke.json`（B1 数据路径、训练、保存、离线重载、报告均验证；`lora_weights_changed=False` 是 1 步 warm-up 系数为 0 的真实结果，非缺陷）。
- GPU 看守日志：`/root/migu-d2/gpu_watch.log`（每 5 分钟记录一次设备可开性）。

## 步骤 1：GPU 恢复确认

```bash
ssh -p 59249 root@183.222.230.12 "/opt/conda/envs/conda_env/bin/python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
```

必须输出 `True NVIDIA GeForce RTX 4090` 才继续；仍为 False 则不上传任何东西、不开始训练。

## 步骤 2：≥50 step 短跑复测 ETA（工作plan口径：warm-up 后测 ≥50 step、两端同步、+30% 余量）

```bash
ssh -p 59249 root@183.222.230.12 "cd /root/migu-d2/repo && git fetch origin && git reset --hard origin/feature/d-plan-sync-1014 && /opt/conda/envs/conda_env/bin/python scripts/train_response_sft.py \
  --train /root/migu-d2/data/b1_full_v1/response_train.jsonl \
  --train-format b1 \
  --condition-jsonl /root/migu-d2/data/b1_full_v1/classification_train.jsonl \
  --config configs/response_sft.json \
  --device cuda \
  --model-dir /root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct \
  --max-train-samples 512 --max-steps 53 --eval-prompts 3 \
  --output-dir /root/migu-d2/weights/response_sft_smoke_1p5b \
  --report-json /root/migu-d2/reports/d_day3_sft_smoke_1p5b.json \
  --report-md /root/migu-d2/reports/D_DAY3_SFT_SMOKE_1P5B.md"
```

验收门槛（与 D1 相同）：first loss 有限、首步梯度有限非零、LoRA 权重变化=True、离线重载一致（全部张量+确定性生成复现）、生成探针非空、显存峰值 <24GB。

## 步骤 3：选型与首轮决策

- 4090 秒/step 实测 × 2,332 步（18,653 样本、batch 1×累积 8）得裸 ETA，+30% 余量后须 ≤ 当日剩余卡时/使用时限；1.5B 超预算时按 TRAINING_WORKPLAN 第 6 条依序缩减：缩上下文（`max_history_turns`/`max_length`）→ 减 epoch → 换 `configs/response_sft_0p5b.json`，每次重测并记录改变。
- 1.5B 与 0.5B 的质量差距（D1 同切片 BLEU-4 0.0377 vs 0.0259、ROUGE-L 0.1565 vs 0.1377）是主要权衡依据；不超预算时首选 1.5B。

## 步骤 4：首轮全量训练（正式模式，不带任何 cap）

```bash
ssh -p 59249 root@183.222.230.12 "cd /root/migu-d2/repo && /opt/conda/envs/conda_env/bin/python scripts/train_response_sft.py \
  --train /root/migu-d2/data/b1_full_v1/response_train.jsonl \
  --train-format b1 \
  --condition-jsonl /root/migu-d2/data/b1_full_v1/classification_train.jsonl \
  --config configs/response_sft.json \
  --device cuda \
  --model-dir /root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct \
  --output-dir /root/migu-d2/weights/response_sft_d2_1p5b \
  --report-json /root/migu-d2/reports/d_day3_sft_full_1p5b.json \
  --report-md /root/migu-d2/reports/D_DAY3_SFT_FULL_1P5B.md"
```

用 `nohup ... &` 或 tmux 无人值守运行；训练日志另存 `/root/migu-d2/logs/`。

## 步骤 5：评测与交付

1. adapter SHA-256、全量报告取回本地（scp），权重哈希入 `docs/EXECUTION_PLAN.md` D2 行。
2. 三路同集评测（保底检索 / 原模型 / 微调）走 `scripts/run_response_inference.py`：B1 val 行（`sample_id`/`context_turns`）已支持直接作为 `--input`；`--condition-jsonl` 接入 C 的预测条件（缺一条即报错）。
3. PR #10 追加 D2 交付（报告、日志、计划 D 行、权重哈希），指派 A 审。

## 失败处理

- GPU 再次不可用：记录 `gpu_watch.log` 证据，上报用户/A，不把未开始训练写成完成。
- 短跑不过门槛：按门槛逐项修（先查数据路径与依赖版本），修好后重跑，不用正式模式赋 True。
