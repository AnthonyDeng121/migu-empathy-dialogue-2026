# D 岗位 Day 6（10/18，G5 冻结日）：候选 v2 权重/tokenizer/adapter 核对与冷启动门禁，A Day6 集成分支 D 侧技术核验

**状态：Integration Gate。** 执行计划 10/18 D 行为"权重/tokenizer/adapter 核对 1–2h；故障与 review 支持 1–2h"，当天交付要求"两次预测哈希一致，性能报告完整；22:00 代码及模型冻结"（G5）。核对主体于 **10/17 16:00–17:00 提前完成**（纯 CPU/本机 4060，GPU 卡时 0），本报告 10/18 补写并对接 A 的 Day6 集成 PR（#22）。五项核对全部通过；另以**公共接口属主身份**对 A Day6 分支做了 D 侧技术核验（该 PR 改动了 D 的 `verify_response_artifact.py` 与 `configs/response_artifact.json`，依据执行计划"相关公共接口追加模块负责人复核"），意见已提交 PR #22 供 A 修复与 B 的正式审查参考——**本 PR 的正式审查归 B（闭环 A→B→C→D→A 即"被下一环审"），D 不持有 approve 权，该定位已于 10/18 在 PR #22 评论中更正**。命令、门禁结果与逐条预测均可独立复现。

## 为什么做这项

G5 要求"权重/配置/依赖/commit 固定；两次冷启动断网、预测哈希一致；材料齐全"。候选 v2（2-epoch adapter）的字节分布在四个地方：本机 `weights/response_sft_d4_2ep/`、GitHub Release `d-day4-response-sft-2ep-20261017`、A 侧集成配置（`configs/response_artifact.json` / `configs/integrated_artifacts.json`）、以及 10/17 远程 4090 上的全量 val 预测记录。冻结前必须证明：

1. 四处字节一致，且 A 配置里钉的指纹与本地产物逐位相同（A 已按其 PR #20 合并前的协调项完成 v2 digest 同步，但未经 D 侧复算验证）；
2. A 的 Day6 分支修改了 `scripts/verify_response_artifact.py` 的 `artifact_digest()` 排序实现（D 的脚本）——排序语义变化不能让已钉 digest 失效；
3. adapter 包内的 tokenizer 文件与基座目录存在字节差异（10/17 打 Release 时发现），必须定性：是否影响推理、是否影响预测复现；
4. 交付候选在断网、冷启动、换路径条件下仍给出字节相同的预测（G5 硬门禁的 D 侧证据，10/15 只对 D2 首轮跑过）。

## 交付物

| 文件 | 作用 |
|---|---|
| `reports/d_day6_artifact_verification.json` | 机器可读门禁结果（三次运行完整命令、digest、determinism/portability/recorded 比对） |
| `results/response_gen/artifact_verify_d6/run{1,2,3}.jsonl` + `*_report.json` | 三次运行逐条预测与性能报告，可独立复核 |
| `logs/D_DAY6_ARTIFACT_VERIFY.txt` | 门禁 stdout（PASS 摘要 + 退出码） |
| `reports/D_DAY6_ARTIFACT_VERIFICATION.md` | 本报告 |
| PR #22 review 评论 | 对 A Day6 集成分支的正式审查（1 P0 + 4 P1 + P2 批） |

## 核对结果汇总

| # | 核对项 | 方法 | 结果 |
|---|---|---|---|
| 1 | digest 一致性 + A 算法改动中性性 | main 版与 A 版 `artifact_digest()` 双向计算；合成嵌套目录对照 | **PASS**：v2 adapter `b3e9017e…`（10 文件/52,866,512 字节）、基座 `3dad5ec4…`（7 文件/3,098,955,668 字节），两种实现逐位相同 |
| 2 | 逐文件 SHA-256 | 本机直算 vs A 的 `configs/integrated_artifacts.json` 钉值 | **PASS**：adapter `adapter_model.safetensors` `6b75245a…`、基座 `model.safetensors` `dd924a11…`，逐位一致 |
| 3 | tokenizer 一致性 | 字节/结构/功能三层比对 + 推理加载路径代码审读 | **PASS（差异为序列化约定，功能等价，推理零影响）** |
| 4 | Release 包复验 | API 资产端点下载（避开 CDN 缓存）+ 包内清单 + 逐字节比对 | **PASS**：zip `e82edaf0…`、包内 11 项 0 失败、10 文件与本地逐字节一致 |
| 5 | 冷启动确定性 + 可移植性门禁 | `scripts/verify_response_artifact.py` 3 次运行（run1/run2 冷启动 + run3 钉文件换路径） | **PASS**：8/8 字节一致 ×2、0 空回复、digest 匹配 |

## 核对 1：digest 一致性与 A 算法改动的中性性实证

A 的 Day6 分支将 `artifact_digest()` 的文件枚举从 `sorted(paths)`（Path 默认比较）改为显式 `key=lambda p: p.relative_to(directory).as_posix()`（相对路径 posix 字符串排序）。该函数是 D 在 10/15 建立的交付指纹算法，`configs/response_artifact.json` 的钉值由它产生。

**实证方法**：从两个分支分别提取该脚本，在同一进程中对同一目录双向计算，并用合成嵌套目录构造排序分歧场景。

| 目录 | main 版实现 | A 的 Day6 版实现 | 是否相同 |
|---|---|---|---|
| `weights/response_sft_d4_2ep`（v2 adapter，扁平 10 文件） | `b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb` | 同左 | **相同** |
| `weights/hf/Qwen2.5-1.5B-Instruct`（基座，扁平 7 文件） | `3dad5ec47fba5d325dd25df184c4fea7d92e913356943f0ef9a6b6f5e0a0b574` | 同左 | **相同** |
| 合成嵌套目录（`a/b.json` 与 `a.json` 竞争排序） | `d121e517…` | `4cc00cf4…` | **不同** |

**结论**：对所有实际交付物（两个目录均为扁平结构）A 的改动**digest 中性**，`configs/response_artifact.json` 里钉的 v2 值继续有效；分歧只出现在嵌套目录中存在"点号文件名 vs 同名子目录"竞争时（旧排序按路径 parts、新排序按字符串：`a/b.json` 排前 vs `a.json` 排前）。**给 A 的含义**（已写入 PR #22 review）：改动方向合理（消除不同 Python 版本间 Path 比较语义差异），但属对 D 脚本的行为变更——建议在 docstring 注明排序语义，且将来若有嵌套结构产物入签，必须用新代码重算；`tests/test_verify_response_artifact.py` 的新用例只覆盖了平目录，未覆盖嵌套分歧场景。

## 核对 2：逐文件 SHA-256

| 文件 | 本机实测 | 期望值（A `configs/integrated_artifacts.json`） | 结果 |
|---|---|---|---|
| `weights/response_sft_d4_2ep/adapter_model.safetensors` | `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e` | 同左 | 一致 |
| `weights/hf/Qwen2.5-1.5B-Instruct/model.safetensors` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` | 同左 | 一致 |

## 核对 3：tokenizer 一致性（adapter 包内 vs 基座目录）

adapter 包为自包含而内嵌了一份 tokenizer（Release 包的 10 个文件之一），与基座目录的对应文件存在字节差异。逐层定性：

| 文件 | 差异 | 定性 |
|---|---|---|
| `vocab.json` | 无（逐字节相同） | — |
| `merges.txt` | adapter 侧多一行 `#version: 0.2` 文件头；合并规则 151,387 条全同（151,388 vs 151,387 行） | 保存格式差异 |
| `tokenizer.json` | 结构全同（BPE、vocab 151,643、merges 151,387、added_tokens 22、version 1.0），但 `model.merges` 存法不同：基座为旧版字符串 `"Ġ Ġ"`，adapter 包为新版二元组 `["Ġ","Ġ"]`；`ignore_merges` 为 `null` vs `false`（语义等价） | 新版 tokenizers 库重存导致 |
| `tokenizer_config.json` | 基座把 `chat_template` 内嵌；adapter 侧拆为独立 `chat_template.jinja` | transformers 新旧版存法差异 |

**功能等价实证**（transformers 5.17.0 实载两份 tokenizer）：三组中英文样词（"今天心情很难过，想找人说说话。"、"我听着，你慢慢说，不着急。"、"Hello world! 128 + 45 = ?"）分词 input_ids 完全一致；`apply_chat_template` 渲染一致。**推理加载路径代码审读**：`scripts/run_response_inference.py:125`（D 的全量 val 路径）与 A 的 `scripts/run_integrated.py:106`（集成路径）均从 `response_model_dir`（基座目录）加载 tokenizer，adapter 内嵌份不参与推理。**结论：字节差异为序列化约定，功能等价，对预测零影响；差异被钉在 adapter digest 内，属预期（digest 钉的就是交付包的精确字节）。**

## 核对 4：Release 包复验（API 端点，避开 CDN 缓存）

按 10/15 与 10/17 的教训（替换资产后 browser_download_url 有 CDN 缓存），用 API 资产端点下载 `d4_2ep_adapter.zip`（asset id 649803995）：

| 项 | 实测 | 期望 | 结果 |
|---|---|---|---|
| zip 大小 | 38,194,384 字节 | 同左 | 一致 |
| zip SHA-256 | `e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3` | 同左（A 的 `integrated_artifacts.json` 亦钉此值） | 一致 |
| 包内 SHA256SUMS | 11 项全验，0 失败（无 10/15 那种自引用条目 bug） | — | PASS |
| 包内 10 模型/tokenizer 文件 vs 本地 `weights/response_sft_d4_2ep` | 逐字节比对 | — | **全部一致** |
| 包内剔除 HANDOFF.md/SHA256SUMS 后目录 digest | `b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb`（10 文件/52,866,512 字节） | 同左 | 一致（整包 12 文件 digest `5e1eb92f…`，文档字节差 2,138 已对上） |

## 核对 5：v2 冷启动确定性与可移植性门禁（G5 硬门禁的 D 侧证据）

`scripts/verify_response_artifact.py`（main 版，SHA-256 `2b3e0f304704e2db42f386277f40f79b3f09f18e65de6555cc53312771cf0141`）对 8 条冻结 B1 val 样本跑三次推理，运行前强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`（三项运行前均未设置，由报告 `offline_env._overridden` 记录）。本次用 **A 的 Day6 分支版 `configs/response_artifact.json`**（即交付时 A 侧实际会用的指纹配置）作为 `--artifact-config`，使 digest 校验同时验证"A 钉的值 vs 本地产物字节"。

| 门禁 | 结果 |
|---|---|
| 字节指纹一致（基座 `3dad5ec4…` 7 文件/3,098,955,668 字节；adapter `b3e9017e…` 10 文件/52,866,512 字节） | **PASS**（status=ok ×2） |
| 冷启动确定性（run1 vs run2，两次独立进程） | **PASS**：8/8 字节一致 |
| 固定文件集可移植（run3 只含 digest 钉住文件、换路径重建） | **PASS**：8/8 与 run1 字节一致 |
| 输出非空 | **PASS**：0 空回复 |
| 与记录文件重叠 | **PASS**：8/8 与 `results/response_gen/d4_2ep_val_full.jsonl` 有共同 id |

诊断项（非门禁）：与 10/17 远程 4090 全量 val 记录比对，8 条中 2 条字节相同（rate 0.25）、mean 字符 BLEU-4 0.6451 / ROUGE-L 0.7073。语义高度接近、措辞漂移属预期——跨 GPU bf16 贪心解码在近似持平 logit 上的 argmax 翻转（10/15 对 D2 的同一诊断为 1/8、0.4808/0.5811，v2 与记录更贴合）。**G5 要求的是同一环境内两次冷启动一致，已通过；请勿把"与远程记录字节相同"当作交付标准。**

三次运行耗时（本机 RTX 4060 8GB，cuda，bfloat16，max_new_tokens 256，samples=8）：

| 运行 | 模型路径 | 总耗时 | 秒/条 | tok/s |
|---|---|---|---:|---:|
| 1 | `weights/hf/Qwen2.5-1.5B-Instruct` + v2 adapter | 76.324s | 9.5405 | 11.80 |
| 2 | 同上（冷启动重复） | 74.229s | 9.2786 | 12.14 |
| 3 | 临时路径（仅固定文件集重建） | 74.373s | 9.2966 | 12.11 |

对照：10/17 远程 4090 全量 2,354 条为 3.3176 秒/条、24.95 tok/s；本机 4060 约为其 1/2.8 速度，量级吻合（D2 在 10/15 的同一对照为 1/1.9–1/2.7）。全量外推约 22,000s ≈ 6.1h（4060），仅作预算参考。

## 对 A Day6 集成分支的 D 侧技术核验（输入给 B 的正式 review；D 不持有 approve 权）

A 的分支 `feature/a-day6-latest-model-integration`（tip `c6d1323`）于 10/17 夜开为 PR #22。审查方式：分支 diff 全量 + 两个并行子智能体分头实跑（分支 177/177 单测、ci_smoke、bash -n、git check-ignore）+ D 本人本地复算全部哈希。完整意见见 PR #22 评论（含 10/18 的角色定位更正），摘要：

**D 侧接线全部正确**（均有本机独立复算证据）：`configs/integrated.json`（v2 adapter/基座/256/无条件/sequence 分类/空画像）符合 D 全部要求；`configs/response_artifact.json` 的 v2 digest 同步正确且与本地复算逐位一致（D 此前 flag 的跨组事项已闭环）；`configs/integrated_artifacts.json` 五哈希三方一致；`run_integrated.py` 生成参数/schema/ID/memory_refs 与 D 已验证路径逐项相同；专家材料打包清单对 D 在 PR #21 §4 给的增量 21/21 全纳入；177/177 单测实跑 + ci_smoke PASS；`artifact_digest()` 排序改动对交付物中性（见核对 1）。

**必修 P0（1 项）**：`Dockerfile.integrated:22,24` COPY 目标仍是 Day4 旧目录名（`weights/emotion_profile/`、`weights/response_sft_d2_1p5b/`），与 `configs/integrated.json`（`weights/emotion_longtail075_b1_20261016`、`weights/response_sft_d4_2ep`）矛盾——构建期 `cloud_preflight.py --files-only --verify-hashes`（Dockerfile:26 → `run_integrated.py:42` 的 `check_files`）必抛 FileNotFoundError；若绕过 preflight，镜像装入的是 D2 首轮旧 adapter 而非锁定的 v2 候选。

**P1（4 项）**：`docs/CLOUD_QUICKSTART.md:9,11,17` Day4 遗留段与 Day6 段自相矛盾；Day6 sequence 新路径零单测零仓库内实跑证据（AGENTS 自查第 3 条，已请 A 附 preflight/cloud_smoke 实际输出或补测）；`docs/EXECUTION_PLAN.md:80` 引用本分支删除的文件成悬空指针；删除 5 个已提交报告（含 B 的两个）+ `**/*VERIFICATION*` 等通配 gitignore 会罩住仍被跟踪的 D 门禁证据 `reports/D_DAY3_ARTIFACT_VERIFICATION.md`——需团队显式背书或收窄模式。P2 一批（response_artifact description 陈旧文案、candidate 字段建议带 digest、`parse_failure_count` 硬编码、`cloud_runtime.json` 不在仓库、新排序用例未覆盖嵌套、EXECUTION_PLAN 缺 Day6 行、合并顺序建议 #21 先合）。

## 复现命令

```bash
# 环境：Python 3.12.3、torch 2.5.1+cu121、transformers 5.17.0；本机 RTX 4060 8GB，GPU 卡时 0
python -m unittest discover -s tests -v          # 172/172 PASS（本分支基线）

# 冷启动门禁（--artifact-config 用 A 的 Day6 分支版指纹配置，使 digest 校验覆盖 A 侧钉值）
python scripts/verify_response_artifact.py \
  --model weights/hf/Qwen2.5-1.5B-Instruct \
  --adapter weights/response_sft_d4_2ep \
  --recorded results/response_gen/d4_2ep_val_full.jsonl \
  --artifact-config /tmp/response_artifact_day6.json \
  --samples 8 --max-new-tokens 256 --device cuda \
  --report reports/d_day6_artifact_verification.json \
  --keep-runs results/response_gen/artifact_verify_d6/
# （/tmp/response_artifact_day6.json 提取命令：
#  git show <A 分支>:configs/response_artifact.json > /tmp/response_artifact_day6.json）

# Release 复验：gh api repos/AnthonyDeng121/migu-empathy-dialogue-2026/releases/assets/649803995 \
#   -H "Accept: application/octet-stream" > d4_2ep_adapter.zip && sha256sum d4_2ep_adapter.zip
```

门禁报告 `reports/d_day6_artifact_verification.json` 含三次运行完整命令（`commands` 字段）与各项门禁布尔量；逐条预测在 `results/response_gen/artifact_verify_d6/`。

## 阻塞与下一步

| 事项 | 状态 | 动作 |
|---|---|---|
| PR #22 的 P0 修复 | 待 A 修改 | A 回复后 D 按 review 逐条复核，通过即 approve |
| PR #21（D5 材料检查） | 待 A 审/合并 | 合并后本分支同步 `logs/README.md` Day6 登记行与 EXECUTION_PLAN（已在本分支先行落地，合并时若有同文件冲突按双方内容并集解决） |
| G5 冻结窗口 | 10/18 18:00 候选全量回归、22:00 代码及权重冻结 | D 待命：A 侧容器两次冷启动回归若与本地门禁出现差异，D 支持诊断（已知跨 GPU bf16 漂移非门禁项）；冻结前终检 main 上 `configs/response_artifact.json` 指向 v2 |
| B2 同集指标复核纳入 v2 臂 | 开放（不归 D 收尾） | 预测已在 main `results/response_gen/d4_2ep_val_full.jsonl`；B 需要时 D 提供口径支持 |
