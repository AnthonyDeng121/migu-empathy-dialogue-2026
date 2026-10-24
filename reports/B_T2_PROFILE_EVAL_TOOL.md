# B T2：画像独立汇总与会话级配对比较

日期：2026-10-24，Asia/Shanghai。范围：第二周期 T2 / B 数据评测；本地实现与验证完成，待既定 C 代码审查，未声明团队 Done。

开工 `git status --short --branch`：`## feature/b-eval-tooling...origin/feature/b-eval-tooling`，无文件改动。开工 `git log -5 --oneline`：

```text
13de391 Merge remote-tracking branch 'origin/main' into feature/b-eval-tooling
c748b47 Merge pull request #29 from AnthonyDeng121/feature/a-mlops
5b5727c 合并最新 main，同步 T1 独立画像接入分支
c023cad feat(eval): 完成 T1 数据诊断与固定评测清单
8c26d7b Merge D cycle2 T1 preparation after acceptance
```

未切换分支、未训练/推理、未使用 GPU、未执行 D 回复实验、未等待本期 C/D 输出；未 add、commit、push 或 merge。原有 T1 产物和其他岗位实现未改动。

## 工具与输入

新增 `scripts/b_profile_eval.py`，提供 `summary`、`compare` 两种模式；新增 `tests/test_b_profile_eval.py`。复用 T1 的 `extended_profile_metrics`、`length_bucket`、`ratio`，间接复用 `src.classification.emotion_profile.profile_metrics` 和 `src.evaluation.metrics.profile_dimension_accuracy`；SHA-256 直接使用 `scripts.b_validate_submission.sha256_file`，JSONL 与 profile schema 使用现有公共实现。`b_compare_candidates.py` 是无 gold 的字段差异工具，不能替代本次有 gold 的画像评测，因此保持原文件不变。

reference 支持 B1 `classification_val.jsonl`，每行必须有非空字符串 `sample_id`（或 `id`）、非空字符串 `conversation_id`、`user_profile`。prediction 最小格式：

```json
{"id":"pp:[\"val_conv_000001\",1,1]","user_profile":{"personality_traits":[],"interests":[],"style":[]}}
```

正式维度名为 **personality_traits / interests / style**；不接受另起 `personality` 别名。接受额外预测字段以兼容分类与提交文件，但只评画像；prediction 不要求情感/回复字段。ID 是非空字符串，不猜测或解析其会话含义；若同时提供 `id` 与 `sample_id`，二者必须一致。

在计算前逐行拒绝非法 ID、重复 ID、缺失/NULL 画像、缺维度/多余维度、非法列表、非字符串/非法/重复标签。随后独立报告 missing / extra ID，不能以行数相同代替覆盖检查。gold、baseline、candidate 都先校验再建索引；预测按 ID 对齐，与物理顺序无关。成功 JSON 的 `id_coverage` 包含行数、unique ID 数、missing/extra/duplicate 和 complete。失败文件评测返回 `passed=false`、错误信息及已有输入的原始字节哈希，CLI 返回 1，且不输出部分指标。

reference 的 `conversation_id` 是唯一映射来源，每个 sample ID 只能出现一次；预测若携带会话字段，必须与 reference 一致。会话明细输出 sample ID 列表，可追溯到 reference。所有输入文件（含可选 manifest）输出路径及原始字节 SHA-256；改变换行、空白或物理行序会改变哈希，这是预期的输入指纹行为。**重排不改变任何指标、切片、会话明细和判定；文件包装中的路径/字节指纹可以变化。**

## 指标口径

每维 exact 是整组标签的集合相等，标签列表顺序无影响。全画像 exact 要求三个维度同时正确；macro exact 为三个维度 exact 的等权平均，是 T1 冻结主指标。保留既有六位小数展示字段与 `macro_exact_unrounded`，比较使用未舍入值。

每维输出：exact、micro precision/recall/F1、固定标签 macro precision/recall/F1、逐标签 PRF/support、gold positive count、predicted positive count、gold/predicted nonempty rate，以及 T1 已有 gold empty/nonempty exact 和非空预测整组 exact precision。

- `samples` 是样本数；`support` 是 gold 正标签出现总数，多标签时可以超过样本数；`per_label.*.support` 是该标签 gold 出现数。
- `micro` 在各维度内池化 TP/FP/FN；`global_micro` 将三维全部标签计数池化。
- 每维 `macro` 对其全部固定合法标签取等权平均，包括零支持标签，零分母按 0。
- `dimension_macro.micro` 平均三个维度的 micro PRF；`dimension_macro.macro` 平均三个维度各自的标签 macro PRF。它们与标签 macro 含义不同。
- 三维分别及“任一维非空”分别输出 gold empty/pred empty、gold empty/pred nonempty、gold nonempty/pred empty、gold nonempty/pred nonempty 四格计数。整体报告预测全空率/任一非空率、样本数、unique conversation 数。

工具只报告数值，不内置 `0.628292`、不生成自动采用结论，也不声称这些是官方指标。

## 配对比较的精确定义

样本层面对全画像及每一维分别统计 baseline/candidate 相同、变化、baseline correct/candidate wrong、baseline wrong/candidate correct、两者都正确、两者都错误。这里 correct 对相应维度集合或全画像 exact 定义；两者都错时仍可能有画像变化。

会话 c 有 n 个 reference 预测点，定义模型得分为：

```text
S(model,c) = 会话内三个维度集合 exact 命中总数 / (3 × n)
delta(c) = S(candidate,c) − S(baseline,c)
```

这等价于逐点三维 macro exact 再在会话内平均，直接沿用 T1 主指标。相同 reference 下两个分母相同，因此用整数命中总数比较：candidate 多则 win，相同则 tie，少则 loss，无浮点容差或人为阈值。每个会话只贡献一次 win/tie/loss；win/loss rate 的分母均为总会话数，包含 ties。mean paired delta 对**会话等权**取均值，内部用 `Fraction` 精确汇总后转为浮点展示；它与总体按预测点计算的 macro exact 差值可能不同。

会话明细按 conversation ID 排序，包含排序后的 sample ID、预测点数、两侧得分、delta 和 outcome。输入重排不影响结果。此统计是描述性配对比较，不包含 bootstrap、置信区间或显著性检验。

## 切片与可靠 join

每个切片输出样本数、unique conversation 数、双方完整汇总；compare 同时输出切片内样本配对和会话配对。

- history：复用 T1，包含目标 user 的历史消息条数，1–5 / 6–15 / ≥16，对应 `short_1_5` / `medium_6_15` / `long_16_plus`。
- gold profile：三个维度全空为 empty，否则 nonempty。
- emotion：reference 的 gold emotion，不使用预测 emotion 分层。

历史优先从 reference 的 `context_turns` 或 `history` 计数，缺失时可以用 reference / manifest 的合法 `history_bucket`；emotion 同样从 reference / manifest 取。`--manifest` 以 sample ID join，允许是 reference 子集，不能包含 reference 外 ID；存在的行必须有一致的 conversation ID。冲突的情感、历史桶或 profile mask 会被拒绝。缺少的元数据进入显式 `unavailable` 切片，不猜测。manifest 仅补充/核对元数据，**不隐式筛选 reference 或 prediction**。

切片内会话分数只平均该切片内预测点。同一会话可以进入多个切片，各切片 win/tie/loss 不能相加当成总体；总体始终每个会话只计一次。没有观测行的切片不生成，所有观测切片均保留 support，小样本不能被百分比掩盖。

## 历史空画像重放

真实输入为 B1 历史 2354 条固定验证数据，900 个会话，**不是新盲测集**。原始文件 SHA-256：

```text
30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201
```

与 T1 `b1_file_checks` 一致。本次明确构造全空预测作为基线重放，未伪造旧画像模型预测；旧画像头聚合报告仍不足以复算其逐条误例。本机重放结果：

| 指标 | 结果 |
|---|---:|
| macro exact，未舍入 | 0.6282922684791844 |
| macro exact，六位展示 | 0.628292 |
| personality_traits exact | 0.618522 |
| interests exact | 0.748938 |
| style exact | 0.517417 |
| 全画像 exact | 0.37170773152081565 |
| 预测全空率 / 任一非空率 | 1 / 0 |
| gold 全空且 pred 全空 | 875 |
| gold 任一非空且 pred 全空 | 1479 |
| 各维/全局 micro、macro PRF | 全部为 0 |

未舍入 macro exact 与 T1 JSON 的 `empty_baseline.profile.macro_exact_unrounded` 精确一致。空 baseline 与自身 compare 为 900 ties，0 wins/0 losses，mean delta=0。

用冻结 `reports/b_t1_screening_manifest.jsonl` 显式抽取两侧输入并 join 校验：160 条、130 个会话，空基线 macro exact 为 **0.4708333333333333**。这与全量值差异来自筛选分布不同，不能把全量 `0.628292` 当作筛选集门槛；未修改清单。筛选 manifest SHA-256：`644c9aa6a79ebb72dd462a4543535ec513fad63bb208967dfb52bc177d6274f1`。

## 合成自测与验证

合成 fixture 共 4 个预测点、3 个会话，其中 A 有两个点，含三维全空与部分非空。以下仅证明工具行为，**不是模型提升结果**。

| 情形 | wins / ties / losses | 会话平均 delta |
|---|---|---:|
| baseline == candidate | 0 / 3 / 0 | 0 |
| 人工正确候选 | 2 / 1 / 0 | +0.2222222222222222 |
| 反向退步 | 0 / 1 / 2 | -0.2222222222222222 |
| 混合改善/退步/持平 | 1 / 1 / 1 | 0 |

混合情形样本层全画像相同 1、变化 3；baseline correct/candidate wrong=1、反向=2、两者都正确=0、两者都错误=1。重排后完整核心结果相同。missing、extra、duplicate、非法画像类型、非法标签、缺失维度均正确拒绝，并留存输入哈希。专项还覆盖同样行数但错误 ID 集合、每一侧 duplicate、不把缺字段当空列表、每维统计、非空四格、固定标签 macro、0 support、15/16 分桶边界、元数据冲突、原始 LF/CRLF 字节哈希及 malformed JSON。

已运行：

```powershell
python -B -m unittest tests.test_b_profile_eval -v
python -B -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/ci_smoke.py
git diff --check
```

专项 22/22 通过；全仓运行 296 项，1 项跳过，其余 295 项通过；compileall、ci_smoke、diff check 全通过。CLI 实际检查 summary 成功、compare 成功、缺 ID compare 非零退出。临时输入、完整输出、自测脚本和日志位于 `.codex_tmp/acceptance/cycle2/t2/`（Git 已忽略），不加入版本控制。可用该目录下 `selftest.py` 重跑本机证据。

## T3 使用方式与限制

T3 再由 B 读取 C 在 T2 完整交付的画像 v1 prediction、标签、阈值、权重/预处理指纹和校准来源，独立复核。T2 无需等待这些新输出。首先核对 C 的训练内会话隔离与校准证据，然后在同一 reference 上运行：

```powershell
python -B scripts/b_profile_eval.py summary --reference data/processed/b1_full_v1/classification_val.jsonl --prediction <C_v1_prediction.jsonl> --manifest reports/b_t1_screening_manifest.jsonl --output .codex_tmp/acceptance/cycle2/t3/v1_summary.json
python -B scripts/b_profile_eval.py compare --reference data/processed/b1_full_v1/classification_val.jsonl --baseline .codex_tmp/acceptance/cycle2/t2/historical_empty_baseline.jsonl --candidate <C_v1_prediction.jsonl> --output .codex_tmp/acceptance/cycle2/t3/v1_vs_empty.json
```

尖括号路径须替换成真实文件；工具可直接接受 C 输出的 `id` 或 `sample_id` 格式。单独评固定筛选集时，先按 manifest ID 显式导出 reference/baseline/candidate 三个相同子集，再运行；不会因传了 manifest 而隐式删掉缺失预测。其他旧候选如只有聚合报告，继续标记缺逐条证据，不能补造结果。

本工具校验文件结构、ID 与可用元数据一致性，不证明权重来源、未见性、无未来信息或训练推理一致性；这些仍须 T3 独立审查。已有 2354 集反复使用，不能据此保证隐藏集提升；无 user_id，不能证明用户级隔离。会话胜负是 T1 指标的分组描述，不是新官方指标；T7 的会话不确定性分析尚未实现。本次未改变采用规则，超越空基线也不自动代表可以上线。
