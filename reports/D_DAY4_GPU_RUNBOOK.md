# D4 2-epoch 候选运行手册（2026-10-16，远程 4090 实例）

**候选定义**：与 D2 首轮（`configs/response_sft.json`，epochs=1）唯一变量是 `epochs=2`（`configs/response_sft_2epoch.json`）。依据：D2 训练 loss 2.969→2.159 在 epoch 末仍单调下降（欠拟合）；错误分析显示弱项在长回复内容针对性。预期耗时：训练约 2×4,220s ≈ 2.35h + 全量 val 推理约 2.5h（cap 保持 256——cap512 探针已证伪放长）。

**实例（2026-10-16 20:26 用户新开，替代已关机的同日 58346 端口实例）**：`ssh root@183.222.230.87 -p 58425`，单张 RTX 4090 24GB，`/` overlay 47G 可用，`/root` 初始为空（无 `/mnt/storage`，无历史 staging）。**远程到 GitHub(codeload) 约 1.3MB/s 可达；hf-mirror/hf.co/modelscope 均不可达（<2KB/s）**，模型只能从本机 scp（实测约 1MB/s，2.45GB ≈ 40min）或经 GitHub Release 中转。SSH 口令认证：本机无 sshpass，用 `SSH_ASKPASS` + `setsid ssh`。

## 步骤 0：环境确认（新实例首次）

```bash
ssh -p 58425 root@183.222.230.87 "/opt/conda/envs/conda_env/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
ssh -p 58425 root@183.222.230.87 "/opt/conda/envs/conda_env/bin/pip install peft==0.21.2"
```

必须输出 `True NVIDIA GeForce RTX 4090`。transformers 4.57.6 为镜像预置；peft 0.21.2 + accelerate 1.15.0 需安装（与 10/14/10/15 训练环境同版本）。

## 步骤 0.5：代码与数据上传（新实例从零开始）

```bash
# 远程拉代码（codeload）
ssh -p 58425 root@183.222.230.87 "mkdir -p /root/migu-d2/repo && cd /root/migu-d2/repo && \
  curl -sL https://codeload.github.com/AnthonyDeng121/migu-empathy-dialogue-2026/tar.gz/refs/heads/feature/d-day4-response-optimization | tar xz --strip-components=1 && \
  sha256sum configs/response_sft_2epoch.json scripts/train_response_sft.py"
# 本机打包并上传基座 + B1 数据（scp 实测约 1MB/s，后台跑）
tar czf /tmp/qwen15b_base.tar.gz -C weights/hf Qwen2.5-1.5B-Instruct
tar czf /tmp/b1_data.tar.gz -C data/processed b1_full_v1
scp -P 58425 /tmp/qwen15b_base.tar.gz /tmp/b1_data.tar.gz root@183.222.230.87:/root/
# 远程解压 + 哈希核验
ssh -p 58425 root@183.222.230.87 "mkdir -p /root/migu-d2/data /root/migu-d2/weights /root/migu-d2/logs /root/migu-d2/results && \
  cd /root/migu-d2/data && tar xzf /root/b1_data.tar.gz && cd b1_full_v1 && sha256sum -c SHA256SUMS && \
  cd /root/migu-d2/weights && tar xzf /root/qwen15b_base.tar.gz"
```

核验口径：B1 三件套以 `data/processed/b1_full_v1/SHA256SUMS` 为准（classification_train `4d4e9498…`、response_train `a70ca036…`、response_val `ddf76588…`）；代码以本地 `sha256sum` 比对（config `082b70e6…`、train 脚本 `bb501a20…`）；基座目录 digest 落训练报告时用 `scripts/verify_response_artifact.py` 同法计算。

## 步骤 1：2-epoch 全量训练（单卡，正式模式无 cap）

```bash
ssh -p 58425 root@183.222.230.87 "
cd /root/migu-d2/repo &&
/opt/conda/envs/conda_env/bin/python scripts/train_response_sft.py \
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

## 步骤 2：新 adapter 全量 val 推理（cap=256，与 D2 首轮完全同路径）

```bash
ssh -p 58425 root@183.222.230.87 "
cd /root/migu-d2/repo &&
/opt/conda/envs/conda_env/bin/python scripts/run_response_inference.py \
  --input /root/migu-d2/data/b1_full_v1/response_val.jsonl \
  --model /root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct \
  --adapter /root/migu-d2/weights/response_sft_d4_2ep \
  --output /root/migu-d2/results/d4_2ep_val_full.jsonl \
  --performance-report /root/migu-d2/results/d4_2ep_val_full_report.json \
  --max-new-tokens 256 --device cuda \
  2>&1 | tee /root/migu-d2/logs/d4_2ep_val_full.txt
"
```

输入用 D2 首轮同文件 `response_val.jsonl`（pp 系 id，与 D2 臂逐行同序，对比免桥接）。预期约 2.5h。

## 步骤 3：取回本机并核验哈希

```bash
scp -P 58425 root@183.222.230.87:/root/migu-d2/results/d4_2ep_val_full.jsonl results/response_gen/
scp -P 58425 root@183.222.230.87:/root/migu-d2/results/d4_2ep_val_full_report.json results/response_gen/
scp -P 58425 root@183.222.230.87:/root/migu-d2/reports/d_day4_sft_full_2ep.json reports/
scp -P 58425 root@183.222.230.87:/root/migu-d2/reports/D_DAY4_SFT_FULL_2EP.md reports/
scp -P 58425 root@183.222.230.87:/root/migu-d2/logs/D_DAY4_SFT_FULL_2EP.txt logs/
scp -P 58425 root@183.222.230.87:/root/migu-d2/logs/d4_2ep_val_full.txt logs/
```

两侧 `sha256sum` 比对预测文件与报告；adapter 目录树 digest 用 `scripts/verify_response_artifact.py` 同法计算并记录。

## 步骤 4：同集对比（本地 CPU）

```bash
python scripts/analyze_response_errors.py \
  --gold data/processed/b1_full_v1/response_val.jsonl \
  --arm d4_2ep=results/response_gen/d4_2ep_val_full.jsonl \
  --arm d2=results/response_gen/d2_1p5b_val_full.jsonl \
  --bridge data/processed/val_eval_input.jsonl \
  --main d4_2ep --base d2 \
  --output reports/d_day4_2ep_comparison.json
```

两臂均为 pp 系 id 可直接 join。判定门槛：长金标桶（101-200、>200）ROUGE-L 相对 D2 首轮提升且短桶不退化 → 候选 v2；否则明确结论"保留 D2 首轮"。
