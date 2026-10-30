# D 岗位 CYCLE2 T7：一次性交付前候选冻结核对

日期：2026-10-30（Asia/Shanghai）。分支：`feature/d-day7-deadline-standby`（核对执行时 HEAD `a969f9d`；随 PR 入库前已合并 main `4dbe135`）。执行性质：**只读核对 + 只写本报告**；未训练、未跑 GPU、未跑推理、未打包、未做任何 git 写操作、未发 Release。

权威冻结来源：`docs/CYCLE_CLOSEOUT_20261021.md`（D 权重 SHA-256、基座 revision、部署配置三项）、`.hermes-memory/weights-via-github-release.md`（Release 资产与目录 digest 口径）、`configs/response_artifact.json`（钉扎 digest 与字节数）。

---

## 一、结论摘要

**冻结候选 = D4 v2**（两轮 SFT LoRA adapter；条件关闭、画像为空、生成上限 256、memory_refs 恒 `[]`），与 CYCLE_CLOSEOUT 固定基线及 CYCLE2 T3/T4/T5 实验结论一致，**未发现任何需要改换候选的证据**。

逐项结果：**9 项核对中 8 项 PASS、1 项 FAIL**。

| # | 核对项 | 结果 |
|---|---|---|
| 1 | adapter 目录文件完整性（10 文件） | PASS |
| 2 | adapter_model.safetensors SHA-256 / 字节数 | PASS |
| 3 | adapter 配置（基座指向 + peft 参数） | PASS（附 1 条说明） |
| 4 | GitHub Release 资产 / zip 哈希 / 目录 digest 口径 | PASS |
| 5 | condition_response=false 与 profile_mode=empty 代码路径 | PASS |
| 6 | 解码参数（max_new_tokens=256、do_sample=false） | PASS |
| 7 | memory_refs 恒输出 [] | PASS |
| 8 | 分类器引用与本地存在性 | **FAIL**（本地部署路径缺失，见 §2.8） |
| 9 | T3/T4 训练实验结论与计划表一致性 | PASS |

**FAIL 详情（唯一，且不影响权重本身）**：`configs/integrated.json` 的 `classifier_dir` 指向 `weights/emotion_longtail075_b1_20261016`，**该目录在本机不存在**；同一份 C 分类器权重实际位于 `weights/classification/c_day5_longtail075`（`model.safetensors` SHA-256 `aedbacb1…` 与冻结值一致、4 文件合计 409,584,858 字节与 C 报告 `checkpoint_bytes` 逐位一致）。即**权重字节是对的、部署引用路径是错的**。按 `scripts/run_integrated.py:69-75` 的 `check_files`，用 `configs/integrated.json` 直接启动会抛 `FileNotFoundError`。该配置属部署入口（A/C 共用），本次只记录不改动（见 §五下一步）。

**D4 v2 权重与发布链路本身 0 FAIL**：本地 adapter 字节、adapter 目录 digest、基座目录 digest、Release 资产字节与 zip 哈希、边车哈希、包内 10 文件逐字节一致性全部对上。

---

## 二、逐项核对表

### 2.1 文件完整性 — PASS

| 项 | 冻结/预期 | 实测 | 结果 |
|---|---|---|---|
| adapter 目录清单 | 10 个文件（含 adapter_config.json、adapter_model.safetensors、tokenizer.json、tokenizer_config.json、special_tokens_map.json、vocab.json、merges.txt、added_tokens.json、chat_template.jinja、README.md） | 10 个文件，逐一在列，无多余、无缺失 | PASS |

证据：
```
$ ls -la weights/response_sft_d4_2ep/
-rwxrwxrwx 1 cubesaint cubesaint     5246 Oct 17 14:47 README.md
-rwxrwxrwx 1 cubesaint cubesaint     1201 Oct 17 14:47 adapter_config.json
-rwxrwxrwx 1 cubesaint cubesaint 36981072 Oct 17 14:47 adapter_model.safetensors
-rwxrwxrwx 1 cubesaint cubesaint      605 Oct 17 14:47 added_tokens.json
-rwxrwxrwx 1 cubesaint cubesaint     2507 Oct 17 14:47 chat_template.jinja
-rwxrwxrwx 1 cubesaint cubesaint  1671853 Oct 17 14:47 merges.txt
-rwxrwxrwx 1 cubesaint cubesaint      613 Oct 17 14:47 special_tokens_map.json
-rwxrwxrwx 1 cubesaint cubesaint 11421896 Oct 17 14:47 tokenizer.json
-rwxrwxrwx 1 cubesaint cubesaint     4686 Oct 17 14:47 tokenizer_config.json
-rwxrwxrwx 1 cubesaint cubesaint  2776833 Oct 17 14:47 vocab.json
```

### 2.2 adapter 哈希对账 — PASS

| 项 | 冻结值 | 实测值 | 结果 |
|---|---|---|---|
| adapter_model.safetensors SHA-256 | `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e` | `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e` | PASS |
| 字节数 | 36,981,072 B（CYCLE_CLOSEOUT/记忆口径，Release 资产同源） | 36,981,072 B | PASS |

证据：
```
$ sha256sum weights/response_sft_d4_2ep/adapter_model.safetensors
6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e  weights/response_sft_d4_2ep/adapter_model.safetensors
```

### 2.3 adapter 配置 — PASS（附 1 条说明）

| 参数 | adapter_config.json 实测 | `configs/response_sft_2epoch.json` | 一致 |
|---|---|---|---|
| base_model_name_or_path | `/root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct` | `model_name: Qwen/Qwen2.5-1.5B-Instruct` | 指向同一基座（见说明） |
| r | 8 | `lora_r: 8` | 是 |
| lora_alpha | 16 | `lora_alpha: 16` | 是 |
| lora_dropout | 0.05 | `lora_dropout: 0.05` | 是 |
| target_modules | q/k/v/o/gate/up/down_proj（7 个） | 同集合 | 是 |
| task_type / peft_type | CAUSAL_LM / LORA | — | 与 LoRA SFT 相符 |
| use_dora / use_rslora | false / false | 未启用 | 相符 |

**说明（非 FAIL）**：`base_model_name_or_path` 记录的是**远程训练机绝对路径** `/root/migu-d2/weights/hf/Qwen2.5-1.5B-Instruct`，不是仓库相对路径；这一点 `configs/response_artifact.json` 的 `adapter.notes` 已明确记载（"A 侧副本若改写该路径需重新固定 digest"）。运行时 `PeftModel.from_pretrained` 只用它做元数据，实际加载的基座来自 `configs/integrated.json` 的 `response_model_dir`（`weights/hf/Qwen2.5-1.5B-Instruct`，见 2.6/2.10 核对）。**该字段参与目录 digest，故不能顺手改写。**

### 2.4 GitHub Release 对账 — PASS

`gh auth status`（只读，先做）：
```
github.com
  ✓ Logged in to github.com account smallrespon
  - Active account: true
  - Token scopes: 'admin:public_key', 'gist', 'read:org', 'repo'
```

Release 视图（只读）：
```
$ gh release view d-day4-response-sft-2ep-20261017 --repo AnthonyDeng121/migu-empathy-dialogue-2026
title:   D Day4 response candidate v2: Qwen2.5-1.5B-Instruct + LoRA (2 epoch)
tag:     d-day4-response-sft-2ep-20261017
prerelease: true
created: 2026-10-17T04:33:51Z   published: 2026-10-17T06:54:24Z
asset: d4_2ep_adapter.zip
asset: d4_2ep_adapter.zip.sha256
```

| 项 | 冻结/预期 | 实测 | 结果 |
|---|---|---|---|
| Release tag / 预发布状态 | `d-day4-response-sft-2ep-20261017`，Pre-release | 一致 | PASS |
| 资产名 | `d4_2ep_adapter.zip` + `.sha256` 边车 | 两个资产均在列 | PASS |
| zip 字节数 | 38,194,384 B | API 资产 `size=38194384` | PASS |
| zip SHA-256 | 边车 `e82edaf0…` | GitHub 资产 digest `sha256:e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3`；API 端点下载实算同值 | PASS |
| 包内结构 | 顶层目录 + adapter/tokenizer 10 文件 + HANDOFF + SHA256SUMS | `d4_2ep_adapter/` 下 12 条目，10 个模型文件与本地**逐字节一致**（diff 仅多出 HANDOFF.md/SHA256SUMS 两个文档） | PASS |
| 包内载荷 digest（去文档） | `b3e9017e…`（10 文件 / 52,866,512 B） | 解包去文档后实算 `b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb` | PASS |

证据（边车与 zip 均经 **API 资产端点**下载，规避 CDN 缓存旧文件，落 `/tmp`、不入库）：
```
$ gh api repos/AnthonyDeng121/migu-empathy-dialogue-2026/releases/tags/d-day4-response-sft-2ep-20261017 \
    --jq '.assets[] | {name, size, id, digest}'
{"name":"d4_2ep_adapter.zip","size":38194384,"id":649803995,
 "digest":"sha256:e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3"}
{"name":"d4_2ep_adapter.zip.sha256","size":85,"id":649804201,
 "digest":"sha256:7cb508551c13ca2b673439981be5cfdc0ebb0e0f00908c95f90f8d27fb31cfc0"}

$ gh api repos/.../releases/assets/649804201 -H "Accept: application/octet-stream" > /tmp/d4_2ep_adapter.zip.sha256
$ cat /tmp/d4_2ep_adapter.zip.sha256
e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3  d4_2ep_adapter.zip

$ gh api repos/.../releases/assets/649803995 -H "Accept: application/octet-stream" > /tmp/d4_2ep_adapter.zip
$ ls -l /tmp/d4_2ep_adapter.zip
-rw-r--r-- 1 cubesaint cubesaint 38194384 Oct 30 17:59 /tmp/d4_2ep_adapter.zip
$ sha256sum /tmp/d4_2ep_adapter.zip
e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3  /tmp/d4_2ep_adapter.zip

# 包内 10 个模型文件与本地逐字节一致（diff 仅列出手工文档）
$ (cd /tmp/d4_zip_check/d4_2ep_adapter && sha256sum * | sort -k2) > /tmp/zip_sums.txt
$ (cd weights/response_sft_d4_2ep && sha256sum * | sort -k2) > /tmp/local_sums.txt
$ diff /tmp/zip_sums.txt /tmp/local_sums.txt
1d0
< 3c9601b63886bdec0483409b952613c7b0c730521b70d22d3832c008a68989fb  HANDOFF.md
3d1
< 1685eb037f23f40f40803386444e52df68c7909bea2c15dbdcdff7ab46bb5ba0  SHA256SUMS
```

**Release 正文记载**（`gh release view` 输出摘录）与本地实测一致：*"Adapter directory digest b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb (10 files / 52866512 bytes). zip sha256 e82edaf0… (matches the .sha256 sidecar, the GitHub asset digest, configs/integrated_artifacts.json response_package_sha256 and weights/README.md; the previously published value 3e59e009... was stale)"*，并明确 *"D4 v2 is the frozen delivery candidate … maintained through the T7 freeze check (2026-10-30)"*。

**"adapter 目录 digest `b3e9017e…`"口径已查清**：它是 `scripts/verify_response_artifact.py` 的 `artifact_digest()`（第 64–87 行）对**目录内全部非隐藏文件**计算的复合摘要，不是任何单个文件的 SHA-256，也不是 zip 的哈希。算法：按相对 POSIX 路径排序后，对每个文件依次 `digest.update(相对路径 UTF-8)` + `digest.update(该文件 sha256 十六进制字符串)`，同时累加字节数；隐藏条目（如 HF `.cache`）排除。发布时是对**包内 adapter 载荷（不含 HANDOFF.md/SHA256SUMS）**算的，故本地 `weights/response_sft_d4_2ep/`（10 文件）与包内去文档载荷算出同一值（上面两条证据均已实算复现）。

### 2.5 条件来源（代码路径） — PASS

| 项 | 位置 | 实测 |
|---|---|---|
| condition_response=false 时不注入条件 | `scripts/run_integrated.py:55-56`（`response_condition`：`if not config["condition_response"]: return None`） | 关闭时返回 `None`，第 173 行调用、第 174 行 `build_eval_sample(row, condition=None)`，prompt 中无情感/画像条件块 |
| profile_mode=empty 时画像为空数组 | `scripts/run_integrated.py:148-149`（`apply_profile`：`if mode == "empty": classification["user_profile"] = {key: [] for key in PROFILE_LABELS}`） | 三维全部置 `[]`；sequence 分类器路径第 166 行亦以空画像初始化 |
| 配置侧取值 | `configs/integrated.json:8` `condition_response: false`；第 4 行 `profile_mode: "empty"` | 与代码路径一致 |

补充：`load_config`（第 31–34 行）强制 sequence 分类器只允许 `profile_mode ∈ {empty, independent}`，`empty` 是合法组合；第 41–42 行要求 `condition_response` 为布尔。

### 2.6 解码参数 — PASS

| 项 | 位置 | 实测 |
|---|---|---|
| max_new_tokens | `configs/integrated.json:9` | 256 |
| do_sample | `configs/inference_config.json:7` `"generation": {"max_new_tokens": 256, "do_sample": false}` | false |
| 运行时实际生效值 | `scripts/run_integrated.py:178-179` | `max_new_tokens=self.config["max_new_tokens"], do_sample=False`（贪心解码硬编码，温度/采样参数不参与） |

`configs/inference_config.json` 第 10 行回指 `configs/integrated.json` 为集成配置，两处口径一致（256 / 贪心）。

### 2.7 memory_refs 恒输出 [] — PASS

| 项 | 位置 | 实测 |
|---|---|---|
| 输出点 | `scripts/run_integrated.py:181` | `return {**classification, "response_text": text, "memory_refs": []}` —— 每条预测硬编码空数组 |
| 字段校验 | `src/data/schema.py:66, 74-77` | 预测字段必须恰为四元组；`memory_refs` 须为无重复字符串列表（空列表合法） |
| 训练/提示词侧 | `src/response/sft_data.py:13` | memory_refs 不进 prompt（无记忆库输入字段） |

### 2.8 分类器引用与本地存在性 — **FAIL**

| 子项 | 冻结记载 | 实测 | 结果 |
|---|---|---|---|
| 配置引用 | `classifier_dir=weights/emotion_longtail075_b1_20261016`、`classifier_format=sequence`（C 冻结：`c-day5-emotion-longtail075-20261016`，sequence） | `configs/integrated.json:2-3` 与 C 冻结记载**逐字一致** | PASS |
| 配置路径本地存在性 | 路径应可被 `check_files` 打开 | `weights/emotion_longtail075_b1_20261016` **不存在**（`find` 全仓无此目录） | **FAIL** |
| 分类器权重本体 | 模型 SHA-256 `aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37` | 实际位于 `weights/classification/c_day5_longtail075/model.safetensors`，SHA-256 **完全一致**；该目录 4 文件合计 409,584,858 B，与 C 报告 `artifacts.checkpoint_bytes` 逐位一致 | PASS（字节） |
| 分类器格式 | sequence | 本地副本为 `BertForSequenceClassification` 4 文件（config.json / model.safetensors / tokenizer.json / tokenizer_config.json），`check_files` sequence 分支所需三件（config.json、model.safetensors、tokenizer_config.json）齐备 | PASS |

**差异详情（FAIL 的具体内容）**：
1. `configs/integrated.json`（以及 C 自己的 `reports/c_day5_emotion_longtail075_metrics.json` `artifacts.checkpoint_path`）写的是 `weights/emotion_longtail075_b1_20261016`；本机实际解落位置是 `weights/classification/c_day5_longtail075`（D 在 T1 下载 Release 后按此路径解落，见 `reports/D_CYCLE2_T1_PREP.md:30`）。
2. 权重字节无差异（模型 SHA-256 与目录总字节双核对一致），**纯粹是路径不一致**。
3. 后果：`scripts/run_integrated.py:69-75` 的 `check_files` 会在启动时抛 `FileNotFoundError`，即**按当前 `configs/integrated.json` 无法在本机直接冷启动**；T7 打包前必须由 A/C 统一该路径（或把副本放到配置指向的位置）。
4. 本次只记录、不改配置、不移动文件（配置属部署入口，且 D 纪律禁止替 A/C 改他们负责的文件）。

证据：
```
$ ls -la weights/emotion_longtail075_b1_20261016/
ls: cannot access 'weights/emotion_longtail075_b1_20261016/': No such file or directory
$ find . -maxdepth 4 -name "*longtail075*" -not -path "./.git/*"
./reports/c_day5_emotion_longtail075_metrics.json
./weights/classification/c_day5_longtail075
$ sha256sum weights/classification/c_day5_longtail075/model.safetensors
aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37  weights/classification/c_day5_longtail075/model.safetensors
$ du -sb weights/classification/c_day5_longtail075
409584858  weights/classification/c_day5_longtail075
```
C 侧目录树 SHA-256 `258c43a7…`（`reports/c_day5_emotion_longtail075_metrics.json` `artifacts.directory_tree_sha256`）：该口径的算法未在仓库中记录，**本次未重新推导，如实记为未核对**；模型文件级 SHA-256 与总字节已双核对一致。

### 2.9 两项训练实验结论与计划表一致性 — PASS

**T3（`reports/D_CYCLE2_T3_UNCONDITIONAL.md`）结论句摘录**：
- 「**160 条筛选集上未观察到无条件训练的优势**……差 −0.0018 / −0.0024；逐条 65 胜 / 65 负 / 30 平……结论限定为『未观察到无条件训练的收益』」。
- 「**按计划维持 condition 关闭，暂保留 D4 v2，不据本轮结果改换候选**……候选改换留待 T6 完整评测」。
- 证据边界：「该诊断集规模不足以证明等价，也不支持『失配代价 ≈ 0』的断言」。

**T4（`reports/D_CYCLE2_T4_LONG_RESPONSE.md`）结论句摘录**：
- 「### 5.1 T4 主判定：**NO-GO**（长样本加权未产生可测收益）」：>200 字桶 ROUGE-L 0.1956 vs D4 v2 0.1950，**Δ=+0.0006，判定线 ≥+0.01 未达**；全集 0.2605 vs 0.2603（Δ=+0.0002）；逐条 1038/1016/300。
- 「按 2.2 节先定后跑的判定线记录负结果，不盲搜第二组权重；T4 adapter 留档本机 `weights/t4_longweight_2ep/`，**不发 Release**；交付候选维持 **D4 v2** 不变」。
- T5 条件矩阵同卡 NO-GO（最好一臂 D4v2 仅情感 Δ=+0.0016，低于预注册线 Δ≥0.005 且 margin≥20），「**维持部署默认：条件关闭**」。

**与 `docs/EXECUTION_PLAN_CYCLE2.md` 计划表核对**：

| 计划行 | 计划原文要点 | 报告/证据 | 一致 |
|---|---|---|---|
| T3 / D（第 128 行） | 「完成完全无条件的两轮 SFT 对照……检验原金标条件训练与部署不注入的失配」 | 报告 §一/§三/§六；T3 BLEU-4 0.1034/ROUGE-L 0.2502 vs D4v2 0.1052/0.2526，逐条 65/65/30，**160 条筛选集未观察到无条件训练收益，不足以证明等价，暂保留 D4 v2** | 一致 |
| T4 / D（第 132 行） | 「选**一个**长回复改进变量……短实验信号成立再完成候选，不盲增生成长度」 | 报告 §五；**>200 字桶 ROUGE-L Δ=+0.0006，判定线 ≥+0.01 未达，NO-GO**，交付候选维持 D4 v2 | 一致 |

两条结论与任务给定的预期结论（T3 未观察到收益、暂保留 D4 v2；T4 Δ=+0.0006 未达 ≥+0.01 判定线、NO-GO）**逐项吻合**，计划表 D 行记载亦无冲突。T3/T4 的负结果均留档（T3 adapter 已发 Pre-release `d-cycle2-t3-unconditional-2ep-20261027`；T4 adapter `bc6ca023…` 未发 Release），**均不构成候选改换理由**。

### 2.10 附带核对（T7 D 单元格"tokenizer/模板"项，超出清单但有交付价值）

| 子项 | 实测 | 结果 |
|---|---|---|
| 基座目录 digest（钉扎值 `3dad5ec4…`，7 文件 / 3,098,955,668 B） | 实算 `3dad5ec47fba5d325dd25df184c4fea7d92e913356943f0ef9a6b6f5e0a0b574`，7 文件 / 3,098,955,668 B | PASS |
| 基座 revision | `configs/response_artifact.json` / `configs/response_sft_2epoch.json` 记 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；本地 `weights/hf/Qwen2.5-1.5B-Instruct` 目录 digest 与钉扎值一致（字节级） | PASS（digest 层面） |
| tokenizer 一致性 | adapter 与基座 `vocab.json` 哈希相同（`ca10d7e9…`）；两者 `tokenizer.json` 均为 BPE / vocab 151,643 / merges 151,387 / added_tokens 22（体积差为序列化差异，非内容差异） | PASS |
| chat 模板 | 部署实际从 `response_model_dir` 取模板（`run_integrated.py:175`）：基座 `tokenizer_config.json` 内 chat_template 与 adapter `chat_template.jinja` **逐字节相同**（均 2507 B，sha256 `cd8e9439…`），未发生模板替换 | PASS |
| 基座 merges.txt 头部差异（观察，非 FAIL） | 基座 `merges.txt` 缺少 `#version: 0.2` 首行（14 B），其余与 adapter 版逐字节一致；该字节状态即钉扎 digest 覆盖的冻结状态 | 记录在案 |

---

## 三、哈希口径说明（三个"哈希"分别是什么）

| 口径 | 对象 | 算法 | 本次实测值 |
|---|---|---|---|
| **adapter safetensors SHA-256** | 单个文件 `weights/response_sft_d4_2ep/adapter_model.safetensors` | `sha256sum`（标准单文件 SHA-256） | `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e`（36,981,072 B） |
| **zip SHA-256** | Release 主资产 `d4_2ep_adapter.zip` 整个压缩包 | `sha256sum`（单文件）；边车 `.sha256` 与 GitHub 资产 digest 三者同值 | `e82edaf05330ba221cdfc89697df2ecf246dee42a379211a831cc94c09b05ba3`（38,194,384 B） |
| **adapter 目录 digest** | 目录 `weights/response_sft_d4_2ep` 内**全部非隐藏文件**（10 个，52,866,512 B；发布时对包内 adapter 载荷计算，**不含** HANDOFF.md / SHA256SUMS） | `scripts/verify_response_artifact.py:64-87` `artifact_digest()`：按相对 POSIX 路径排序，对每个文件依次把「相对路径 UTF-8 字节」和「该文件 sha256 的十六进制字符串」喂入同一个 SHA-256，同时累加字节数；隐藏条目（如 HF `.cache`）排除 | `b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb`（10 文件 / 52,866,512 B） |

复算命令（只读，脚本落 `/tmp`，未写入仓库）：
```python
# /tmp/t7_digest.py —— 与 scripts/verify_response_artifact.py artifact_digest() 同算法
import hashlib, sys
from pathlib import Path
def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as h:
        for chunk in iter(lambda: h.read(1024*1024), b""): d.update(chunk)
    return d.hexdigest()
def artifact_digest(directory):
    files = sorted((p for p in directory.rglob("*") if p.is_file()
                    and not any(part.startswith(".") for part in p.relative_to(directory).parts)),
                   key=lambda p: p.relative_to(directory).as_posix())
    dg = hashlib.sha256(); total = 0
    for p in files:
        dg.update(p.relative_to(directory).as_posix().encode("utf-8"))
        dg.update(sha256_file(p).encode("utf-8")); total += p.stat().st_size
    return {"digest": dg.hexdigest(), "files": len(files), "bytes": total}
```
```
$ python3 /tmp/t7_digest.py weights/response_sft_d4_2ep weights/hf/Qwen2.5-1.5B-Instruct /tmp/d4_payload
weights/response_sft_d4_2ep {'digest': 'b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb', 'files': 10, 'bytes': 52866512}
weights/hf/Qwen2.5-1.5B-Instruct {'digest': '3dad5ec47fba5d325dd25df184c4fea7d92e913356943f0ef9a6b6f5e0a0b574', 'files': 7, 'bytes': 3098955668}
/tmp/d4_payload {'digest': 'b3e9017e79bea947338730975b7490d9f432dea94c32e848ad66af1bb98f3ecb', 'files': 10, 'bytes': 52866512}
```
（`/tmp/d4_payload` = Release zip 解包后去掉 HANDOFF.md/SHA256SUMS 的 adapter 载荷，与本地目录同 digest。）

**关系**：单文件 SHA-256 管"这个文件没被换"；目录 digest 管"这个目录的文件集合、文件名与逐文件内容都没被换"（防半拷贝/多文件/改名）；zip SHA-256 管"发布出去的压缩包没被换"。三者互不替代，`configs/response_artifact.json` 同时钉扎后两者（外加基座目录 digest）。另注意 Release 正文原载旧值 `3e59e009…`（与边车/GitHub digest/仓库配置均不符），**已于 2026-10-30 由 D 修正为 `e82edaf0…`**（`gh release edit --notes-file`，仅改正文、资产未动；本次下载实算确认现行有效 zip 哈希为 `e82edaf0…`，未采信 CDN/browser 链接）。

---

## 四、局限声明

1. **本地 BLEU-4/ROUGE-L 只是字符级诊断实现**（`src/evaluation/metrics.py`），官方指标未公布；本报告引用的 T3/T4/T5 数值均为该字符级诊断口径，**不冒充官方分数，亦不构成"官方达标"表述**。
2. **本次未跑 GPU、未重训、未跑推理、未打包、未做冷启动验证**；本报告是字节级/配置级/代码路径级核对，不产生任何新的生成质量证据。
3. **分类器归 C 负责**，D 只核对其部署引用与存在性；分类器质量、校准、阈值不在本次范围。分类器目录树 SHA-256 口径未记录于仓库，本次未推导（模型文件 SHA-256 与总字节已核对一致）。
4. `gh auth status` 显示当前登录账号为 `smallrespon`，Release 查询使用该账号的只读 token；本次只做了 `gh auth status` / `gh release view` / `gh api`（GET）三类只读操作，**未创建、未上传、未删除任何 Release 资产**。
5. 边车与 zip 下载仅用于哈希核验，落 `/tmp`（`/tmp/d4_2ep_adapter.zip`、`.sha256`、`/tmp/d4_zip_check/`、`/tmp/d4_payload/`、`/tmp/t7_digest.py`），**未写入仓库、未 `git add`**。
6. 本地 git 状态：分支 `feature/d-day7-deadline-standby` HEAD `a969f9d`；工作区仅有既存未跟踪项（`.hermes-memory/`、`AGENTS.md`、`results/response_gen/*_progress/writer.lock` 等），本报告为唯一新增跟踪外文件，未 commit/push/stash。

---

## 五、下一步

1. **T7 交付窗口 11/03–11/04**：按计划 D 单元格为"核对冻结回复模型、条件来源、解码和 tokenizer，整理两项训练实验结论；不临时换模板或权重"——本报告即该项交付物，结论是**候选冻结维持 D4 v2，无需改换**。
2. **必须处理的 FAIL（T7 打包前）**：分类器部署路径不一致（`configs/integrated.json:2` 的 `weights/emotion_longtail075_b1_20261016` vs 实际 `weights/classification/c_day5_longtail075`）。建议由 A/C 定夺统一口径（改配置指向，或把副本落到配置路径）；D 不代改。改路径后需复跑 `scripts/run_integrated.py` 的 `check_files` 冒烟（CPU 轻量即可，本机 4060 可做）。
3. **PR 状态更正**：任务背景写"PR #39 待 A 审"，实际 `gh pr view 39` 显示 **PR #39 已 MERGED**（2026-10-30T03:10:20Z，merge commit `b4c07f73c56c390245831128e8971b378230b5f6`，标题"CYCLE2 T4：长样本加权 SFT 完成 + 无人值守评测管线"）。本地 `origin/main` 快照仍停留在 `9b104b1`（PR #38 合并），尚不含该 merge commit；**T7 前需 fetch 同步 main 并合入 D 分支**（fetch/merge 属 git 写操作，本次未执行）。
4. **每日审查职责提醒（本任务范围外）**：`gh pr list --state open` 显示 C 的 **PR #41**（`docs(classification): record C T5 stability review`，head `feature/c-day5-classification-optimization-20261016`）处于 OPEN 且无 reviewer，按 D 的每日固定职责需安排审查；本次冻结核对未涉及该 PR 内容。
5. **T6/D 收尾材料**可直接复用本报告 §三（哈希口径）与 §2.4/§2.10（Release 与 tokenizer/模板核对）作为 T7 交接附件；`docs/EXECUTION_PLAN_CYCLE2.md` 目前尚无 T7/D 行（表中 D 行止于 T6/D），按 `[[update-plan-d-rows]]` 规矩需在计划表补 T7/D 行（状态 Done + 本报告为证据）——该文件为团队共用计划，补写动作留待与 A 协同，本次未改。
