# D 岗位 Day 3：D2 首轮 LoRA/SFT（冻结 B1 训练集）

**状态：候选已产出（Integration Gate）——adapter、训练日志、复现命令、冒烟/全量报告齐全；val 全量推理进行中；待 B2 同集三路评测与 A review 后定采用。** 平台侧事故与恢复过程见文末记录。

## 候选与配置

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`（Apache-2.0；赛事对预训练模型的使用许可仍待 A 按官方规则核实） |
| 训练数据 | 冻结 B1 `response_train.jsonl` 18,653 条（SHA-256 `a70ca036f2ad28265bbbd3ce6882006f81e7dd7abb39a50dbf0f16cb713e2ff6`），排除 4 条异常（邻接/上下文问题，无金标不一致） |
| 条件注入 | `condition_mode: gold`——金标情感/画像自 B1 `classification_train.jsonl`（SHA-256 `4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5`）按 `sample_id` join，18,653/18,653 全覆盖；缺一条即报错，不静默回退无条件提示 |
| LoRA | r=8、alpha=16、dropout=0.05，target q/k/v/o/gate/up/down；可训练 9,232,384（0.59%） |
| 批次/长度 | batch 1 × 梯度累积 8；max_length 2048；max_history_turns 20；epochs 1；lr 1e-4；warmup 3%；seed `20261013` |
| 设备/环境 | 远程 RTX 4090 24GB（bfloat16）；Python 3.12.12、torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2、accelerate 1.15.0（平台 vLLM 大赛专属镜像预置环境） |

数据防泄漏在 B1 转换（`prepare_training_data.py --mode full`，`context_after_target=false`、`target_assistant_in_context=false`）之上，由本 PR 的 B1 加载器再次复验：context turn_id 严格升序、终于目标用户轮、且全部 ≤ 目标轮（目标助手回复为 target+1，结构上不可能进入上下文）。

## 冒烟与测速（工作plan口径）

53 步冒烟（3 warm-up + 50 计时步，512 样本）：**1.8153 秒/step**（完整梯度累积窗口、两端同步）；first loss 2.416、首步梯度范数 1.146、392 个 LoRA 张量全部变化、离线重载 392/392 张量一致 + 3/3 确定性生成复现、显存峰值 6.72GiB。外推全量 1 epoch 4,233s（+30% 余量 5,503s）。

## 首轮全量训练（正式模式，无任何 cap）

- **2,332 optimizer steps**（18,653 样本、batch 1×累积 8），实测 **1.8094 秒/step**，总墙钟 **4,219.7s ≈ 1.17h**——冒烟外推 4,233s 与实际偏差 0.3%，ETA 口径可信。
- mean train loss 2.969（first）→ 2.159（末）；18,653/18,653 样本全部使用，超长跳过 0；token 长度 min 116 / mean 902 / max 2048。
- 检查全部真实通过：first loss 有限、首步梯度有限非零（范数 0.880）、LoRA 权重变化=True（392 张量）、离线重载一致（392/392 张量 + 3/3 生成复现）、生成探针 3/3 非空。
- **adapter**：36,981,072 字节，SHA-256 `62a695ea594a1546e217ee5edd691131051a4e64e3ecfcec91af75de7a133fcf`（落盘文件独立 sha256sum 核验一致）；目录自包含 tokenizer，`local_files_only` 可离线加载。
- 证据：`reports/d_day3_sft_smoke_1p5b.json` / `reports/D_DAY3_SFT_SMOKE_1P5B.md`、`reports/d_day3_sft_full_1p5b.json` / `reports/D_DAY3_SFT_FULL_1P5B.md`、`logs/D_DAY3_SFT_FULL_1P5B.txt`；运行代码 commit `6761204182134153b245810219935ed9323ca60fc`（dirty=False）。**报告口径修正记录**：三份 MD 末行原为冒烟模板文案"本机运行不计平台 GPU 卡时"，对本次平台租卡训练不成立；已由渲染器修正为中性表述并从运行 JSON 重新渲染（JSON 权威记录与运行时代码哈希未动），修正逻辑有单测锁定（117/117）。

## 显存/吞吐预算（4090 vs 4060）

| 项 | 本机 4060 8GB（D1 冒烟） | 远程 4090 24GB（本次） |
|---|---|---|
| 秒/step | 12.1061 | 1.8094（6.7×） |
| 全量 1 epoch | 28,243s（7.85h） | 4,219.7s（1.17h） |
| 显存峰值 | 6.72GiB | 6.72GiB（24GB 卡余量充足） |
| 原模型推理 | 241.4s/100 条（35.7 tok/s） | val 全量 2,354 条推理中，ETA 见推理报告 |

卡时口径：首轮训练约 1.17 GPU h + 冒烟约 0.2 GPU h + val 推理约 1.6 GPU h（进行中），均发生在平台租用实例上，按平台账单记录。

## 评测与采用（待办）

1. **无条件臂（已完成）**：`run_response_inference.py` 对 B1 val 2,354 条全量推理（`--max-new-tokens 256`，无 condition 注入）：**8,844.5s（3.757 秒/条，p50 2.545s，22.4 tok/s，197,717 token）**。本地字符级诊断（`src/evaluation.metrics` 已测函数，按 `(conversation_id, target_user_turn_id)` 键对齐官方 raw val 金标，同代码路径重算各臂）：

| 臂 | BLEU-4（字符） | ROUGE-L（字符 F1） | 空输出 | 平均长度 |
|---|---|---|---|---|
| 检索 baseline（全量 2,354） | 0.0294（D1 记录 0.0267） | 0.1439（D1 记录 0.1375） | 0 | — |
| **微调 1.5B + D2 adapter（全量 2,354）** | **0.1075** | **0.2542** | **0** | 135.9（金标 158.2） |
| 原模型 1.5B 未微调（D1 100 条同切片参考） | 0.0377 | 0.1565 | — | — |
| 原模型 1.5B 未微调（全量 2,354） | 未跑完（见下） | 未跑完 | — | — |

   微调臂相对检索 baseline：BLEU-4 约 3.7×、ROUGE-L 约 1.8×，合法率 100%，长度校准合理（p50 90 vs 金标 99）；相对 D1 原模型 100 条切片约 2.8×/1.6×（切片不同，仅供参照）。字符级指标仅本地诊断，非官方口径。
   **原模型全量同集臂说明**：远程 4090 于 16:45 第二次被平台抢占回收后，该臂改在本机 4060 启动；因机器需要移交下一位使用者，运行至 111 分钟（约 2,050/2,354 条）时按用户要求终止，未产出全量数。指标驱动已用 D1 记录的 100 条切片精确复现验证（0.0377/0.1565 分毫不差），B2 或后续获得 GPU 时可用同一命令补跑：`scripts/run_response_inference.py --input data/processed/b1_full_v1/response_val.jsonl --model <1.5B 快照> --output results/response_gen/original_1p5b_val_full.jsonl --performance-report results/response_gen/original_1p5b_val_full_report.json --full-set-size 2354 --max-new-tokens 256 --device cuda`。
2. **预测条件臂**：待 C 的 C2 权重产出预测条件后，用 `--condition-jsonl` 复跑同集（缺一条即报错，拒绝混合条件/无条件提示）；"无条件 vs 预测条件注入"是 D/C 待回答问题，**不得把验证真值标签用于最终候选冒充部署结果**。
3. 采用决定按执行计划 10/14 22:00 门槛；B2 的同环境全量指标为准，本地字符级指标仅诊断。

## 平台事故与恢复记录（2026-10-14）

- 11:41:09 远程实例创建（控制台实例 ID `a5503077-6d2c-4600-8bbb-a5b3fd456430`），11:47 GPU 可用；11:50 起 `/dev/nvidia0` 打开 EPERM、NVML `Unknown Error`、`torch.cuda.device_count()=0`。根因：控制台显示该实例**调度方式为"抢占"**，GPU 被平台回收；容器为 k8s burstable pod、无 `CAP_SYS_ADMIN`、seccomp 过滤，内网无法自愈。
- 用户重启实例后 GPU 恢复，冒烟/训练/微调臂推理全部完成（13:12–16:45）；**16:45 后 GPU 第二次被回收**（原模型同集臂启动即报 CUDA unavailable）。原模型臂改在本机 RTX 4060 上跑（D1 原模型推理同环境；诊断评测、非训练）。
- 跨环境一致性核验（保证三路对比可比）：同一 B1 val 样本在远程（transformers 4.57.6）与本机（5.17.0）的 chat template 渲染 + tokenize 结果逐 token 一致（样本 1：94 tokens，样本 1000：168 tokens，哈希相同）；tokenizer 快照同 revision。
- 阻塞期间完成：B1 适配代码与单测、本地/远程真实数据预检、远程环境搭建（数据上传哈希核验、依赖确认、模型经 hf-mirror 下载）、CPU 端到端冒烟、staging 迁移持久盘 `/mnt/storage/migu-d2/`（200G NFS）、`reports/D_DAY3_GPU_RUNBOOK.md` 执行手册。

## 复现

```bash
python -m pip install -r requirements-response.txt
python scripts/train_response_sft.py \
  --train <b1_full_v1>/response_train.jsonl \
  --train-format b1 \
  --condition-jsonl <b1_full_v1>/classification_train.jsonl \
  --config configs/response_sft.json \
  --device cuda \
  --model-dir <Qwen2.5-1.5B-Instruct 快照> \
  --output-dir weights/response_sft_d2_1p5b \
  --report-json reports/d_day3_sft_full_1p5b.json \
  --report-md reports/D_DAY3_SFT_FULL_1P5B.md
```

远程绝对路径版本（含数据/模型位置）见 `reports/D_DAY3_GPU_RUNBOOK.md`。
