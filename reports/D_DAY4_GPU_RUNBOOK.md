# D4 2-epoch 候选运行手册（2026-10-16，远程双 4090 实例）

**候选定义**：与 D2 首轮（`configs/response_sft.json`，epochs=1）唯一变量是 `epochs=2`（`configs/response_sft_2epoch.json`）。依据：D2 训练 loss 2.969→2.159 在 epoch 末仍单调下降（欠拟合）；错误分析显示弱项在长回复内容针对性。预期耗时：训练约 2×4,220s ≈ 2.35h + 全量 val 推理约 2.5h（cap 保持 256——cap512 探针已证伪放长）。

## 步骤 0：实例与文件状态确认（实例若被重建，先做步骤 0.5）

```bash
ssh -o ConnectTimeout=10 root@183.222.230.87 -p 58346 "
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader;
ls /root/migu-d2/ 2>/dev/null;
ls /root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct/config.json /root/migu-d2/weights/response_sft_d2_1p5b/adapter_model.safetensors 2>/dev/null
"
```

预期：双 4090 可见；`/root/migu-d2/` 含 `data/ weights/ repo/ reports/ logs/`。若权重/代码丢失（实例被重建），执行步骤 0.5 从本机重传（基座 3.09GB + adapter 53MB；B1 三件套在 `data/processed/b1_full_v1/`，约 60MB）：

```bash
# 本机执行（SSH 口令认证见 ~/.zcode 记忆：SSH_ASKPASS + setsid）
scp -P 58346 -r weights/hf/Qwen2.5-1.5B-Instruct root@183.222.230.87:/root/migu-d2/weights/hf/
scp -P 58346 -r weights/response_sft_d2_1p5b root@183.222.230.87:/root/migu-d2/weights/
scp -P 58346 -r data/processed/b1_full_v1 root@183.222.230.87:/root/migu-d2/data/
```

## 步骤 1：同步当日分支代码（远程 git 不通 GitHub 主站，codeload 通）

```bash
ssh root@183.222.230.87 -p 58346 "
mkdir -p /root/migu-d2/repo && cd /root/migu-d2/repo &&
curl -sL https://codeload.github.com/AnthonyDeng121/migu-empathy-dialogue-2026/tar.gz/refs/heads/feature/d-day4-response-optimization | tar xz --strip-components=1 &&
git init -q 2>/dev/null; echo HEAD_CODE_COMMIT=\$(sha256sum configs/response_sft_2epoch.json scripts/train_response_sft.py | md5sum | cut -c1-12)
"
```

记录 `HEAD_CODE_COMMIT` 与两个文件 SHA-256，写进训练报告的 reproduction 字段。

## 步骤 2：2-epoch 全量训练（GPU 0，正式模式无 cap）

```bash
ssh root@183.222.230.87 -p 58346 "
cd /root/migu-d2/repo && mkdir -p /root/migu-d2/logs &&
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/conda_env/bin/python scripts/train_response_sft.py \
  --train /root/migu-d2/data/b1_full_v1/response_train.jsonl \
  --train-format b1 \
  --condition-jsonl /root/migu-d2/data/b1_full_v1/classification_train.jsonl \
  --config configs/response_sft_2epoch.json \
  --device cuda \
  --model-dir /root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct \
  --output-dir /root/migu-d2/weights/response_sft_d4_2ep \
  --report-json /root/migu-d2/reports/d_day4_sft_full_2ep.json \
  --report-md /root/migu-d2/reports/D_DAY4_SFT_FULL_2EP.md \
  2>&1 | tee /root/migu-d2/logs/D_DAY4_SFT_FULL_2EP.txt
"
```

验收门槛（与 D2 首轮相同）：first loss 有限、首步梯度有限非零、LoRA 权重变化=True、离线重载一致、显存峰值 <24GB。预期 4,664 步（2×2,332）、约 2.35h。

## 步骤 3：新 adapter 全量 val 推理（cap=256，与 D2 首轮完全同路径）

```bash
ssh root@183.222.230.87 -p 58346 "
cd /root/migu-d2/repo &&
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/conda_env/bin/python scripts/run_response_inference.py \
  --input /root/migu-d2/data/b1_full_v1/response_val.jsonl \
  --model /root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct \
  --adapter /root/migu-d2/weights/response_sft_d4_2ep \
  --output /root/migu-d2/results/d4_2ep_val_full.jsonl \
  --performance-report /root/migu-d2/results/d4_2ep_val_full_report.json \
  --max-new-tokens 256 --device cuda \
  2>&1 | tee /root/migu-d2/logs/d4_2ep_val_full.txt
"
```

输入用 D2 首轮同文件 `response_val.jsonl`（远程已有，pp 系 id，与 D2 臂逐行同序，对比免桥接）。

## 步骤 4：取回本机并核验哈希

```bash
scp -P 58346 root@183.222.230.87:/root/migu-d2/results/d4_2ep_val_full.jsonl results/response_gen/
scp -P 58346 root@183.222.230.87:/root/migu-d2/results/d4_2ep_val_full_report.json results/response_gen/
scp -P 58346 root@183.222.230.87:/root/migu-d2/reports/d_day4_sft_full_2ep.json reports/
scp -P 58346 root@183.222.230.87:/root/migu-d2/reports/D_DAY4_SFT_FULL_2EP.md reports/
scp -P 58346 root@183.222.230.87:/root/migu-d2/logs/D_DAY4_SFT_FULL_2EP.txt logs/
scp -P 58346 root@183.222.230.87:/root/migu-d2/logs/d4_2ep_val_full.txt logs/
```

两侧 `sha256sum` 比对预测文件与报告；adapter 目录树 digest 用 `scripts/verify_response_artifact.py` 同法计算并记录。

## 步骤 5：同集对比（本地 CPU）

```bash
python scripts/analyze_response_errors.py \
  --gold data/processed/b1_full_v1/response_val.jsonl \
  --arm d4_2ep=results/response_gen/d4_2ep_val_full.jsonl \
  --arm d2=results/response_gen/d2_1p5b_val_full.jsonl \
  --bridge data/processed/val_eval_input.jsonl \
  --output reports/d_day4_2ep_comparison.json
```

两臂均为 pp 系 id 可直接 join（`--bridge` 仅为脚本兼容 eval 系 id 保留）。head_to_head 以第一个臂（d4_2ep）为主、d2 为对照，报告里注明对照关系。
