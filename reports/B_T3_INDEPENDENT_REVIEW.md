# B T3：C 画像 v1 与 D 条件实验独立复核

日期：2026-10-26（Asia/Shanghai）。范围：第二周期 T3-B；本地独立复算完成，待既定审查，非团队 Done。

**画像 v1：CONDITIONAL GO，仅保留为独立画像研究候选；emotion-only：NO-GO，保留条件关闭。当前没有证据支持 T5 使用画像第三臂。**

画像通过 T1 必要主指标门槛，但并未证明整套非空画像可靠。条件允许继续检查 checkpoint 和开展后续预先固定的研究，不代表同意部署或把 v1 三维画像直接注入回复。C T3 的 v2 对照不属于本次 T2 产物复核，不据其数字作本次选型。

## 开工与复算范围

开工工作区干净，固定分支 `feature/b-eval-tooling`，HEAD `1d2a4f10e29fc6945828036f7aa62f650fdcaf20`。开工日志如下；本次不切分支、不训练、不推理、不使用 GPU、不修改 A/C/D 实现、不 add/commit/push/merge。采用 karpathy-guidelines 的最小改动原则，复用既有工具，正式仅新增本报告和 JSON。

```text
1d2a4f1 Merge remote-tracking branch origin/main into feature/b-eval-tooling
9d1cc05 Merge pull request #32 from AnthonyDeng121/feature/c-day5-classification-optimization-20261016
16ca7ca fix(classification): sanitize profile training output paths
ef52fb3 fix(classification): keep profile metadata paths portable
b3cc9e1 feat(classification): train profile-only encoder for Cycle2
```

阅读执行计划、T1/T2 报告及 T2 评测代码后，使用本地已有输入重新计算。画像复用 `scripts/b_profile_eval.py`，回复复用 `scripts/compare_t2_arms.py` 和 `src/evaluation/metrics.py`；长度问题复用 `src/response/error_analysis.py`。本次没有新增正式代码。临时编排脚本 `.codex_tmp/acceptance/cycle2/t3/recheck.py`、完整会话明细和日志均在忽略目录，不作为正式源码提交。

## 真实产物与证据边界

C 的真实逐条证据为 `reports/c_cycle2_profile_v1_predictions.jsonl`：2354 行，含 ID、conversation_id、三维画像及每个固定标签的概率。标签映射可从概率字典与官方 schema 逐项核对；阈值来自 C 机器报告：personality_traits=0.64、interests=0.81、style=0.56。三个维度共 7062 次概率解码检查全部一致，数值有限且在 [0,1] 内。

C 自报文件为 `reports/C_CYCLE2_T1_T3_PROFILE.md`、`reports/c_cycle2_profile_v1_20261026.json`、`reports/c_cycle2_profile_v1_b_eval_summary.json`。本次没有照抄这些数字：所有评测来自真实预测重算，随后与其主指标和会话配对交叉核对一致。

C 声明模型为 `weights/profile_encoder_v1_b1_20261026`，通用 MacBERT revision `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1` 起步，encoder 可训练，仅画像 BCE；checkpoint 树 SHA 为 `a8dfb90341c4a5d8ad72a6fd38f5741489114b24efd4898805d0f63ca0542bf7`。**这些训练/权重事实仍为 C 自报**：本机没有该 checkpoint、独立 checkpoint metadata、校准概率和交接 ZIP，B 未重载权重或复算阈值最优性，不能将其写成 B 验证通过。

B 已独立验证训练/验证数据和冻结 split manifest 的哈希与 C 登记一致，并调用既有 B1 adapter / `partition_by_manifest` 重建划分：fit 15944 条/7080 会话、calibration 2709 条/1249 会话；fit/calibration 及 train/validation 会话交集均 0。训练代码先在 calibration 概率上选 exact 阈值再生成验证结果，与声明相符，但缺校准概率，仍不能完整重放实际阈值选择。

D 真实证据为 `t2_off_local4060.jsonl`、`t2_emotion_only.jsonl`、历史抽取 `t2_condition_off.jsonl`，均在 `results/response_gen/`；另有筛选输入/manifest、C Day5 条件文件、两臂报告和 D 比较报告。以下采用同机 off 与 emotion-only 比较，历史 4090 off 仅作设备对照，不混入主比较。

## C 完整性与 SHA-256

2354 行、2354 unique sample ID、900 unique conversation；duplicates/missing/extra 均 0，顺序与 B1 reference 相同，conversation 映射及三维 schema 全通过。personality 的正式字段名为 `personality_traits`。完整结果随机重排 reference、baseline、candidate 后完全相同。

| 文件表示 | 原始字节 SHA-256 |
|---|---|
| 本机 CRLF prediction | `25d06cff8e53705ca59ec29a2b4907ea46b4b79039fb432284e1d72771834ccc` |
| Git LF blob / C 登记 | `4327bcb3fcb796b8e414b5c40931fe0b6fd5eb274a5b8141a65df471583eecb8` |

两种哈希不同仅因 Windows checkout 将 2354 个 LF 换为 CRLF，逐字节规范化后与 Git blob 完全一致。没有把规范化哈希冒充本机原始字节哈希。reference 的 SHA 为 `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201`。所有实际输入/代码字节哈希见 JSON `sources`。

## C 总体与非空指标

集合 exact 忽略标签顺序；full exact 要求三维全部相等。macro exact 是三维 exact 等权平均；每维标签 macro PRF 在该维全部固定枚举上平均（含零支持，零分母=0）；dimension macro 分别平均三维 micro PRF 和三维标签 macro PRF，不能混称。support 为金标正标签出现数，samples 才是行数。

空基线 macro exact 为 **0.6282922684791844**，v1 为 **0.6468422543188899**，差 **+0.018549985839705463**，与 T1/T2 同口径；full exact 仅从 0.3717077315 升至 0.3734069669（+4/2354）。2354 集历史上反复使用，不是新盲测，本地数值不等于官方分数。

| 维度 | 空 exact | v1 exact | micro P / R / F1 | 标签 macro P / R / F1 | support | gold / pred 非空率 |
|---|---:|---:|---|---|---:|---|
| interests | 0.748938 | 0.819456 | 0.815884 / 0.351477 / 0.491304 | 0.552235 / 0.254957 / 0.315201 | 643 | 0.251062 / 0.117672 |
| personality_traits | 0.618522 | 0.607052 | 0.376923 / 0.042870 / 0.076984 | 0.257619 / 0.024453 / 0.040832 | 1143 | 0.381478 / 0.047154 |
| style | 0.517417 | 0.514019 | 0.418367 / 0.092273 / 0.151199 | 0.093523 / 0.023513 / 0.035845 | 1333 | 0.482583 / 0.121071 |

global micro P/R/F1 = 0.567760 / 0.127605 / 0.208377；dimension macro micro P/R/F1 = 0.537058 / 0.162207 / 0.239829；dimension macro 标签 macro P/R/F1 = 0.301125 / 0.100974 / 0.130626。

收益由 interests 提供（exact +0.070518），personality_traits −0.011470、style −0.003398。预测任一非空率 0.248513，低于 gold 0.628292，没有总体非空过度预测；但低召回和错误标签仍突出。

### 空/非空行为

下表四格顺序固定为金标空/预测空、金标空/预测非空、金标非空/预测空、金标非空/预测非空。非空检测 precision 仅表示“预测非空时 gold 也非空”，不检查标签是否正确；不能将 90.43% 检测精度当成画像标签可靠性。

| 维度 | 空空 | 空非空 | 非空空 | 非空非空 | 检测 P / R | 空误报率 | 非空预测整组 exact precision |
|---|---:|---:|---:|---:|---|---:|---:|
| any_profile | 819 | 56 | 950 | 529 | 0.904274 / 0.357674 | 0.064000 | 0.102564 |
| interests | 1720 | 43 | 357 | 234 | 0.844765 / 0.395939 | 0.024390 | 0.754513 |
| personality_traits | 1412 | 44 | 831 | 67 | 0.603604 / 0.074610 | 0.030220 | 0.153153 |
| style | 1118 | 100 | 951 | 185 | 0.649123 / 0.162852 | 0.082102 | 0.322807 |

personality/style 的非空整组 exact precision 仅 0.153153 / 0.322807；interests 为 0.754513。整套任一非空预测中 full exact 为 60/585=0.102564（要求三维同时正确，较严格）；同时 global 标签 precision 仅 0.567760。不能只拿 interests 或二元非空检测有利值证明整套画像适合注入。

## C 切片

历史消息包含目标 user，复用 T1：1–5 / 6–15 / ≥16。所有切片分别重算对应空基线。

| history | support | macro exact | 相对空基线 Δ | global micro F1 | gold / pred 任一非空率 |
|---|---:|---:|---:|---:|---|
| history:long_16_plus | 353 | 0.556185 | 0.007554 | 0.134897 | 0.713881 / 0.237960 |
| history:medium_6_15 | 1098 | 0.612022 | 0.020340 | 0.205493 | 0.670310 / 0.270492 |
| history:short_1_5 | 903 | 0.724622 | 0.020672 | 0.255973 | 0.543743 / 0.225914 |

三档主指标均有正差，但长历史增益较弱，micro F1 最低。

| gold emotion | support | macro exact | 相对空基线 Δ | 任一非空检测 P / R |
|---|---:|---:|---:|---|
| anger | 115 | 0.678261 | 0.023188 | 0.925926 / 0.362319 |
| anxiety | 751 | 0.655570 | 0.015979 | 0.878453 / 0.347921 |
| care | 11 | 0.666667 | 0.000000 | 0.750000 / 0.600000 |
| disgust | 21 | 0.698413 | 0.000000 | 0.833333 / 0.454545 |
| fear | 124 | 0.736559 | 0.005376 | 1.000000 / 0.275862 |
| gratitude | 118 | 0.624294 | 0.025424 | 0.896552 / 0.325000 |
| helplessness | 142 | 0.718310 | 0.002347 | 0.937500 / 0.405405 |
| joy | 20 | 0.683333 | 0.083333 | 1.000000 / 0.466667 |
| loneliness | 115 | 0.715942 | 0.017391 | 0.857143 / 0.285714 |
| mixed | 177 | 0.638418 | 0.005650 | 0.860465 / 0.316239 |
| neutral | 53 | 0.591195 | 0.012579 | 1.000000 / 0.428571 |
| pride | 40 | 0.441667 | -0.025000 | 0.916667 / 0.354839 |
| relaxed | 409 | 0.596577 | 0.027710 | 0.918519 / 0.435088 |
| sadness | 131 | 0.681934 | 0.015267 | 0.869565 / 0.266667 |
| shame | 118 | 0.581921 | 0.050847 | 0.962963 / 0.295455 |
| surprise | 9 | 0.370370 | 0.148148 | 1.000000 / 0.444444 |

mixed n=177，macro exact 仅 +0.005650；care n=11，主指标持平，不能声称改善。pride n=40 为 −0.025000；高支持 anxiety n=751、relaxed n=409 没有整体主指标退步，但这不消除逐维低精度。joy/surprise 小样本正差不作稳定收益证明。

gold 全空 n=875：56 条被报非空，误报率 0.064；full exact 0.936，macro exact 0.974476（比空基线低 0.025524）。gold 任一非空 n=1479：529 条检测到非空、950 条仍全空；full exact 60/1479=0.040568，macro exact 0.453009（较空基线 +0.044625），全局标签 micro F1 0.212380。完整逐维和所有切片 PRF 在 JSON。

## C 样本与会话配对

会话分数 = 该会话三维 exact 命中总数/(3×预测点数)，整数命中数判断 win/tie/loss；会话只计一次，delta 对会话等权，不以预测点充当独立会话。

900 会话：151 win / 639 tie / 110 loss；win rate 0.167778、loss rate 0.122222；平均 paired delta **+0.015356848912404467**。随机输入重排结果一致。

样本按主指标三维命中数：298 改善、161 退步、1895 持平。按 full exact：baseline wrong→candidate correct 60，反向 56；两者正确 819、两者错误 1419；画像相同 1769、变化 585。每维改善/退步分别 personality 17/44、interests 209/43、style 92/100。完整会话明细仅保存在忽略目录 `profile_compare_full.json`，正式 JSON 保留总体及切片统计。

## D 两臂可比性与限制

两臂均 160 行、160 unique ID、160 会话，duplicate/missing/extra=0；ID 顺序与 selection 完全一致，schema 合法且全非空。selection 每行与 B1 response gold 原行相等，历史与 classification reference 一致。两臂及 selection 反序后比较结果相同。

D 使用自身 T1 冻结清单（seed 20261022，160 会话、12 格）；与 B T1 160 条/130 会话只重叠 **11** 个 ID，不能冒称同一筛选集。D 原分层历史桶是 ≤4 / 5–12 / >12；本报告统一重新用 B T1 ≤5 / 6–15 / ≥16 切片。未按结果挑换样本，未要求重跑。

两臂报告一致登记模型 `weights/hf/Qwen2.5-1.5B-Instruct`、adapter `weights/response_sft_d4_2ep`、cuda/bfloat16、相同输入、max_new_tokens=256、min_new_tokens=0。代码为 greedy（do_sample=false、use_cache=true）；条件路径的唯一代码差异为 system prompt 注入预测情感。逐个构造 160 对 prompt，历史消息均相同，emotion-only 不携带 user_profile 或验证回复。

但两份生成报告没有逐臂不可变权重/tokenizer digest、GPU 标识或完整环境指纹。D 声明同一 4060 和相同权重，B 可核对路径/参数声明与代码，**不能据此独立证明实跑唯一变量**；所以把差异视作有来源限制的诊断关联，不能做强因果或显著性结论。

C Day5 条件文件有 2354 个完整 ID，仅 id/emotion_label；B 重算情感 Accuracy=0.716653、Macro-F1=0.584843，与文件生成报告逐项一致，未使用 gold 条件。筛选 mixed 7 条的预测实际为 anxiety 3、anger 1、relaxed 1、mixed 1、neutral 1；care 2 条全被预测为 anxiety。因此这些切片并不等于“注入正确 mixed/care 标签”的实验。

## D 总体回复比较

34 条相同、126 条改变。BLEU-4 和 ROUGE-L 均复用现有字符级逐样本诊断，再取平均，不是官方分数。长度均值/中位数按非空白字符；near-empty 复用 ≤6 stripped 字符，long 复用 >200 stripped 字符。

| 指标 | off | emotion-only | Δ |
|---|---:|---:|---:|
| bleu4_char | 0.102842 | 0.099968 | -0.002874 |
| empty_count | 0.000000 | 0.000000 | 0.000000 |
| length_mean_nonwhitespace | 124.725000 | 125.425000 | 0.700000 |
| length_median_nonwhitespace | 84.000000 | 87.500000 | 3.500000 |
| long_count | 32.000000 | 33.000000 | 1.000000 |
| long_rate | 0.200000 | 0.206250 | 0.006250 |
| near_empty_count | 3.000000 | 3.000000 | 0.000000 |
| rouge_l_char_f1 | 0.249654 | 0.249040 | -0.000614 |

ROUGE gold-overlap 逐条增加 63、减少 60、持平 37（容差 1e-9），不称作质量胜负。长度增加/减少/不变 60/62/38，平均 +0.7。B 重算与 D 聚合一致，逐条 BLEU 最大跨环境浮点差仅 6.938893903907228e-18。D Markdown 写 ROUGE −0.0007；精确重算为 **−0.0006139607907582334**，以本次原始精度为准。

历史 4090 off 与本地 off 的 46/160 回复相同可重现，ROUGE 分别 0.253276/0.249654。它是一组历史设备对照，不足以建立重复试验噪声分布或“条件效应小于设备噪声一个量级”的普遍结论；本报告不沿用该强推断。

## D 切片

| history | support | Δ BLEU | Δ ROUGE | overlap 增/平/减 |
|---|---:|---:|---:|---|
| history:long_16_plus | 26 | -0.001116 | -0.008451 | 6/8/12 |
| history:medium_6_15 | 69 | -0.002093 | 0.000476 | 28/19/22 |
| history:short_1_5 | 65 | -0.004406 | 0.001364 | 29/10/26 |

| gold emotion | support | Δ BLEU | Δ ROUGE |
|---|---:|---:|---:|
| anger | 9 | -0.001120 | 0.012722 |
| anxiety | 51 | -0.007220 | -0.000032 |
| care | 2 | -0.003556 | -0.029731 |
| disgust | 4 | -0.006869 | -0.028333 |
| fear | 6 | -0.003276 | -0.018478 |
| gratitude | 8 | 0.002429 | -0.010302 |
| helplessness | 9 | -0.005202 | -0.005105 |
| loneliness | 7 | -0.009811 | 0.006530 |
| mixed | 7 | -0.003014 | 0.017897 |
| neutral | 1 | 0.000000 | 0.000000 |
| pride | 3 | -0.012080 | 0.004619 |
| relaxed | 36 | 0.003028 | 0.000445 |
| sadness | 7 | 0.017239 | 0.014845 |
| shame | 10 | -0.010356 | -0.010385 |
| joy | 0 | 不可估计 | 不可估计 |
| surprise | 0 | 不可估计 | 不可估计 |

sadness n=7、relaxed n=36 两个诊断指标均正；anger/mixed 等仅 ROUGE 正、BLEU 负。care n=2、fear n=6、disgust n=4、shame n=10 有负信号，样本不足以稳定判定。anxiety n=51 的 ROUGE 几乎持平而 BLEU 下降；长历史 n=26 两个指标均退步，短/中历史 ROUGE 微正但 BLEU 均负。整体没有平均改善，也不能宣称主要类别全面稳定。mixed 7 条的冲突指标及 care 2 条不足以支持继续按类别开启。

逐样本正文已省略，仅保留聚合统计。

## 采用建议及 T5 边界

**画像 v1：CONDITIONAL GO。**通过同集 exact 必要门槛且会话平均为正，允许保留为研究候选；收益并非三维一致，personality/style 非空 precision 与 gold-empty 误报不足以支持整套画像可靠性。条件是完成 checkpoint/metadata/校准概率交接复验，并在后续预先固定实验中处理低精度和切片问题；不得用本验证集拟合新阈值或事后挑选维度，不能将此结论写成部署 GO。

**emotion-only：NO-GO。**当前小筛选集未见总体诊断收益，长历史有负信号，且实际单变量证据不完整。保留条件关闭；这不是证明两臂质量等效，也不推广到所有模型。

**当前没有证据支持 T5 使用画像第三臂。**整套 v1 的主指标过线不自动证明适合给 D 做条件；interests-only/门控实验若以后提出，须作为新候选和新预设实验独立评估，本次没有执行或批准该变体。

## 验证、复现与限制

本次仅新增报告与 JSON，无正式代码变动。全仓 `python -B -m unittest discover -s tests -v`：327 项，326 通过、1 项平台跳过；`python -m compileall -q src scripts tests`、`python scripts/ci_smoke.py`、`git diff --check` 均 PASS。已验证两类输入重排不变、完整覆盖、概率解码和参考 join，不启动模型。

本机重放命令：

```powershell
python -B .codex_tmp/acceptance/cycle2/t3/recheck.py
python -B scripts/b_profile_eval.py compare --reference data/processed/b1_full_v1/classification_val.jsonl --baseline .codex_tmp/acceptance/cycle2/t3/empty_baseline.jsonl --candidate reports/c_cycle2_profile_v1_predictions.jsonl --manifest reports/b_t1_screening_manifest.jsonl --output .codex_tmp/acceptance/cycle2/t3/profile_cli.json
python -B scripts/compare_t2_arms.py --off results/response_gen/t2_off_local4060.jsonl --emotion results/response_gen/t2_emotion_only.jsonl --selection results/response_gen/t2_selection_input.jsonl --output .codex_tmp/acceptance/cycle2/t3/response_cli.json
```

JSON `sources` 保留输入与复用代码 SHA；`profile`/`response` 区分重算结果、C/D 声明和不能验证的 provenance。局限包括：缺 C 权重/校准概率、缺 D 逐臂权重和环境身份、非盲测历史验证、筛选集支持数不足且与 B 清单不同；未做置信区间、人工盲评或官方指标评测。上述缺项如实记录，没有补造预测或通过长推理补齐。
