# D 岗位 Day 3：原模型全量同集臂补跑（三路对比补齐）

**状态：Done（本轮推理与指标已完成并入档）。** 10/14 因远程 GPU 被抢占改本机 4060 跑、机器移交前终止于约 87% 的**原模型全量同集臂**，10/15 用新开远程双 4090 跑完 2,354/2,354。至此"检索 / 原模型 / 微调模型"三路同集对比第一次全部落在同一冻结集、同一代码路径上。

## 运行配置与证据

| 项 | 值 |
|---|---|
| 远程实例 | `ai-traininfer-c8096171-2bd12a93-6ddb877685-v68zc`（`ssh root@183.222.230.87 -p 58346`），RTX 4090 24GB，双卡（本任务单卡） |
| 起止 | 2026-10-15 15:02:44 → 16:05:02 +08，墙钟 **3,737.778s ≈ 62.3 min** |
| 模型 | `/root/base_qwen15b` = `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，**不挂 adapter**（报告 `adapter_path: null`） |
| 数据 | 冻结 B1 `response_val.jsonl` 2,354 行，SHA-256 `ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2`（与 B2 记录一致） |
| 环境 | Python 3.10.12、torch 2.5.1+cu124、transformers 5.17.0、peft 0.21.2、accelerate 1.15.0（`requirements-response.txt` 钉版本）；`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |
| 生成参数 | greedy（`do_sample=False`）、`max_new_tokens=256`、`use_cache=True`、bfloat16、`condition_mode: none`——与 D2 臂完全一致 |
| 吞吐 | 1.5878 秒/条、p50 1.1869s、198,835 tokens、53.2 tok/s、**0 空输出** |
| 基座一致性 | 远程 `model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`，与本机 `weights/hf/Qwen2.5-1.5B-Instruct/` 逐字节相同（后者目录 digest `3dad5ec4…` 已在 `configs/response_artifact.json` 固定） |

产物（本机 `results/response_gen/`、`logs/`，回传后哈希与远程逐一致）：

| 文件 | SHA-256 |
|---|---|
| `original_1p5b_val_full.jsonl` | `022dd88977e6e1c34aeefc04ffaab12cb4356b40e473699e19106ada5125a121` |
| `original_1p5b_val_full_report.json` | `3491020721dbb962bdd7195d1a610a355f1ad7b051cddf1156496ba3ab49c6b7` |
| `logs/original_1p5b_val_full.txt` | `7fdeb0596c34a08e8d3fc8ce7f4b6344f3e05c2354410a8c73f12e7bc32b882c` |

复现命令（远程已按此执行；脚本即仓库 `scripts/run_response_inference.py`，无新增代码路径）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_response_inference.py \
  --input data/processed/b1_full_v1/response_val.jsonl \
  --model /root/base_qwen15b \
  --output results/response_gen/original_1p5b_val_full.jsonl \
  --performance-report results/response_gen/original_1p5b_val_full_report.json \
  --device cuda --max-new-tokens 256 --full-set-size 2354
```

**ETA 口径**：开跑前 2 条冒烟外推全量 2,729.7s，实测 3,737.8s，**实测为外推的 1.37 倍**（低估）。与 D2 全量推理"外推约 1.6h、实测 2.46h（1.54 倍）"同向——小样本外推系统性低估，正式预算必须按实测修正，不能引用外推值。

## 覆盖与对齐校验

- 2,354 行、2,354 个唯一 id、重复 0、缺失 0、额外 0；**行序与 B1 `response_val.jsonl` 完全一致**（逐行比对 `True`）。
- 金标取官方 raw val 的 `(conversation_id, target_user_turn_id)`，共 2,354 个 prediction point；B1 每行 `response_text` 与 raw val 对应金标逐条相同（**0 处不一致**，独立复核过一遍）。
- 指标由 `src.evaluation.metrics` 的 `bleu4` / `rouge_l` 计算，与 B2 独立复核、D2 报告同一条代码路径。

## 三路同集对比（同一冻结 B1 val、同一代码路径）

| 臂 | 覆盖 | 字符 BLEU-4 | ROUGE-L F1 | 空输出 | 来源 |
|---|---|---:|---:|---:|---|
| 检索 baseline | 2,354 | 0.029367 | 0.143870 | 0 | 既有 `reports/baseline_val_full_metrics.json` |
| **原模型 Qwen 1.5B（无 adapter）** | 2,354 | **0.051974** | **0.175606** | 0 | 本次新增，全量实跑 |
| D2 LoRA/SFT | 2,354 | **0.107468** | **0.254184** | 0 | PR #10 已合并；本次用同代码路径重算，四舍五入与记录值（0.1075 / 0.2542）一致 |

- D2 相对原模型：BLEU-4 **2.07×**、ROUGE-L **1.45×**。
- 原模型相对检索：BLEU-4 1.77×、ROUGE-L 1.22×。
- D2 相对检索：BLEU-4 3.66×、ROUGE-L 1.77×。
- **逐条胜负**：D2 在 **1,979/2,354（84.1%）** 个 prediction point 上 ROUGE-L 高于原模型，原模型仅在 375 个点上更高。提升是普遍的，不是被少数样本拉高的。

## 关键发现：两臂的失败模式正好互补

| 项 | 原模型 | D2 |
|---|---|---|
| 回复长度（非空白字符） | mean 150.9 / median 113 / min 7 / max 515 | mean 135.9 / median 90 / min 2 / max 470 |
| 近空回复（≤6 字符） | **0 条（0.0%）** | **32 条（1.4%）** |

原模型的低分切片几乎全是**过度泛化的长篇安抚**撞上**极短金标**：金标是"我在。"、"我听着。"、"好，我听着。"、甚至"[不回应]"，原模型却输出"理解你的感受，这种情况很常见。很多人在经历创伤后都会有一种恐惧感……"这类长段。D2 明显学到了短确认这个语域，所以在这类点上大幅胜出；代价是偶尔过切成近空（32 条）。

**对采用判断的意义**：D2 的收益主要来自学会短确认语域，而不是整体语言质量提升；同时它引入了原模型没有的近空失败模式。这两个事实都要进候选结论——不能只报"BLEU-4 翻倍"。若要继续优化（D3），近空回复是最明确的靶子，且需要约束"最短生成长度"而不是继续压长度。

## 与既有记录的关系

- B2 报告（`reports/B_DAY3_B2_EVAL.md`）中"原模型仅 100 条切片、NOT FULL SET、不能与 D2 同级排名"的限制，随本次全量跑完**解除**；该表现在可以按全量同级重排。表由 B 维护，D 不代改，数值以本报告为准。
- `docs/EXECUTION_PLAN.md` 的 D2 Day 3 行"原模型全量同集臂…补跑命令在 D2 报告"已同步为已完成，证据指向本报告。
- 10/14 记录的约 87% 中断产物未落盘（脚本跑完才写文件），本次为全量重跑，非续跑。
