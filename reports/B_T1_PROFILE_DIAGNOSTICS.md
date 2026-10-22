# B T1：历史画像诊断与固定实验清单

日期：2026-10-22（Asia/Shanghai）。范围：第二周期 T1 / B 数据评测。状态：本地交付待 C 审查，非团队 Done。
工作区开工干净，分支 feature/b-eval-tooling，HEAD de8f0da；未找到根目录、父目录或工作区内 AGENTS.md。未训练、下载、使用 GPU 或新推理；未修改 A/C/D 实现、官方数据、历史报告与权重，未提交、推送。

## 结论与证据边界

空画像 macro exact match 重算为 **0.628292**，可在本次同集同协议下作为数值门槛。历史画像头的全量报告均未证明超过它；Day4 不加权 512 exact 校准与它持平，第二周期采用结论仍为 no-go / 保留空画像。旧报告的 Micro-F1 阶段交接通过与采用是不同结论。
2354 条固定验证集曾被历史实验反复使用，不能称为新盲测。Day3/Day5 情感可在本机归档逐条回放，旧画像头只有聚合报告，未补造逐条预测。历史训练的冻结编码器见过训练内校准样本，新的会话划分不能消除既有表示学习暴露。

## 数据与字段

B1 实际路径为 data/processed/b1_full_v1/，原始数据为 data/raw/训练-验证-数据集/{train,val}/{train,val}_public.jsonl。B1 分类字段为 sample_id、conversation_id、target_user_turn_id、source_split、context_turns、context_text、emotion_label、user_profile；history 截止目标 user。所有行通过仓库 B1 adapter 和标签校验；缺字段/NULL 与合法空数组分别处理。无 user_id，不能证明用户级隔离。

| 数据 | raw unique conversation_id | 所有预测点覆盖的 unique conversation_id | usable=true / B1 样本覆盖的 unique conversation_id | 原始预测点 | B1 样本 | ID 重复 | 缺失/NULL 必需字段 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 8338 | 8333 | 8329 | 18657 | 18653 | 0 | 0 |
| val | 900 | 900 | 900 | 2354 | 2354 | 0 | 0 |

8329 的定义是 classification_train.jsonl 中 18653 条 usable 训练样本实际覆盖的 unique conversation_id 数，与 split_manifest.jsonl 中 train 且 usable=true 的会话集合完全一致；它不是 raw train 的会话总数。raw、所有预测点、usable=true 和实际 B1 样本分别从真实文件核算，并逐个核对 (sample_id, conversation_id) 及出现次数。

train 会话差异：8338 → 8333 → 8329。无 prediction point 的会话 5 个：train_conv_000077, train_conv_000284, train_conv_000325, train_conv_000691, train_conv_000767；有预测点但全部 unusable 的会话 4 个：train_conv_000628, train_conv_000817, train_conv_003644, train_conv_003695；其他缺失原因 0 个，usable 会话不在 raw 中 0 个。

- train_conv_000628：1 个预测点，usable 样本 0；pp:["train_conv_000628",15,1]: assistant_target_not_adjacent_future_turn, target_assistant_in_context。
- train_conv_000817：1 个预测点，usable 样本 0；pp:["train_conv_000817",33,1]: assistant_target_not_adjacent_future_turn, target_assistant_in_context。
- train_conv_003644：1 个预测点，usable 样本 0；pp:["train_conv_003644",7,1]: assistant_target_not_adjacent_future_turn, target_assistant_in_context。
- train_conv_003695：1 个预测点，usable 样本 0；pp:["train_conv_003695",9,1]: assistant_target_not_adjacent_future_turn, target_assistant_in_context。

val 会话差异：900 → 900 → 900。无 prediction point 的会话 0 个：无；有预测点但全部 unusable 的会话 0 个：无；其他缺失原因 0 个，usable 会话不在 raw 中 0 个。


训练的 9 个会话差额由 5 个零预测点会话和 4 个全部预测点不可用会话组成：8338 − 5 − 4 = 8329；没有其他差额。4 个不可用会话各自只有 1 个预测点，均因 assistant_target_not_adjacent_future_turn、target_assistant_in_context 被排除，不能将它们概括为没有 prediction point。会话 ID 与逐点原因见 JSON raw.train。

训练/验证 sample_id 交集 0，会话交集 0；fit/calibration 会话交集 0。原始训练的 4 个非相邻目标预测点已被既有 B1 转换排除（ID 见 JSON raw.train.b1_excluded_ids），不得再把原始 18657 点当作 B1 训练数。原始结构检查细节、每个 B1 文件条数与哈希匹配结果见 JSON。

| 集合 / 画像维度 | 空数 | 空率 | 非空数 | 非空率 | 缺失/NULL |
|---|---:|---:|---:|---:|---:|
| train/personality_traits | 10492 | 0.562483 | 8161 | 0.437517 | 0 |
| train/interests | 12918 | 0.692543 | 5735 | 0.307457 | 0 |
| train/style | 8731 | 0.468075 | 9922 | 0.531925 | 0 |
| validation/personality_traits | 1456 | 0.618522 | 898 | 0.381478 | 0 |
| validation/interests | 1763 | 0.748938 | 591 | 0.251062 | 0 |
| validation/style | 1218 | 0.517417 | 1136 | 0.482583 | 0 |

各维度标签支持数（多标签支持数之和不等于非空样本数；包含零支持标签）：

| 维度 | 标签 | Train | Val |
|---|---|---:|---:|
| personality_traits | agreeable | 361 | 53 |
| personality_traits | assertive | 675 | 74 |
| personality_traits | casual | 27 | 4 |
| personality_traits | conservative | 1364 | 253 |
| personality_traits | emotionally_stable | 336 | 9 |
| personality_traits | extroverted | 37 | 2 |
| personality_traits | high_conscientiousness | 2279 | 205 |
| personality_traits | introverted | 717 | 52 |
| personality_traits | open | 59 | 4 |
| personality_traits | sensitive | 5161 | 487 |
| interests | art_design | 341 | 32 |
| interests | career_development | 1472 | 128 |
| interests | film_animation | 108 | 13 |
| interests | games | 704 | 81 |
| interests | music | 133 | 19 |
| interests | pets | 744 | 106 |
| interests | programming_technology | 82 | 1 |
| interests | reading_writing | 316 | 40 |
| interests | social | 764 | 23 |
| interests | sports_fitness | 624 | 84 |
| interests | study_exam | 744 | 95 |
| interests | travel_outdoor | 144 | 21 |
| style | brief | 1166 | 120 |
| style | colloquial | 653 | 54 |
| style | detailed | 1121 | 147 |
| style | direct | 2600 | 281 |
| style | emoji_user | 27 | 0 |
| style | formal | 128 | 22 |
| style | high_emotional_expression | 943 | 57 |
| style | humorous | 81 | 8 |
| style | indirect | 516 | 59 |
| style | low_emotional_expression | 3 | 0 |
| style | rational | 4418 | 585 |

## 历史长度与画像

历史以包含目标 user 的消息条数分桶：短 1–5、中 6–15、长 ≥16；字符数为 content 长度总和，不含格式符号。这里不以字符数猜 tokenizer 截断长度。分桶为 B 本次预设设计，非历史团队阈值。

| 集合/历史 | 样本 | 全空画像数 | personality 非空率 | interests 非空率 | style 非空率 |
|---|---:|---:|---:|---:|---:|
| train/short_1_5 | 6390 | 2596 | 0.325665 | 0.239124 | 0.515180 |
| train/medium_6_15 | 9161 | 3061 | 0.496889 | 0.344504 | 0.558673 |
| train/long_16_plus | 3102 | 1177 | 0.492585 | 0.338814 | 0.487427 |
| validation/short_1_5 | 903 | 412 | 0.244740 | 0.204873 | 0.438538 |
| validation/medium_6_15 | 1098 | 362 | 0.436248 | 0.280510 | 0.508197 |
| validation/long_16_plus | 353 | 101 | 0.560907 | 0.277620 | 0.515581 |

长短历史的画像非空率存在差异，不能把所有空数组视为漏标；上述为关联统计，不能推出因果。训练与验证的历史消息/字符 min、mean、p50、p95、max 见 JSON。

## 指标定义与历史可比性

复用 src.classification.emotion_profile.profile_metrics、emotion_encoder.classification_metrics 和 evaluation.metrics.profile_dimension_accuracy。exact 是每个维度整组标签集合完全相等，macro exact 对三维等权平均。micro 按标签 TP/FP/FN 池化；标签 macro 对该维所有固定枚举（含无支持标签）取均值，零分母为 0。global_micro 合并三维计数，dimension_macro 分别平均三维的 micro 与标签 macro。情感固定 16 类，混淆矩阵行是真值、列是预测。JSON 保留未舍入 macro exact，主要历史展示六位小数。
历史 full-val 画像的 B1 train/val 哈希与本机完全一致；旧 head tensor exact 的 bool 行全等与当前集合 exact 定义等价。256/512 使用相同 speaker 行序列、左截断，但长度和校准目标不同，只能作历史端到端结果描述，不能当作统一预处理的单变量排名。空画像不依赖输入截断，因而可在同一金标上核对。缺预测则不能独立复算画像误例、非空率或长度切片。

| 历史候选（按时间/配置列示，非排名） | macro exact | 各维 exact（personality/interests/style） | 各维 Micro-F1 | 证据 |
|---|---:|---|---|---|
| 空画像 | 0.628292 | 0.618522 / 0.748938 / 0.517417 | 0.000000 / 0.000000 / 0.000000 | 本次完整重算 |
| c_day3_retrained_metrics.json | 0.429340 | 0.404418 / 0.607477 / 0.276126 | 0.287376 / 0.199728 / 0.270270 | 聚合报告；speaker lines, left truncation, max_length=256; training-only micro-F1 calibration |
| c_day4_profile_unweighted_metrics.json | 0.382470 | 0.278250 / 0.531861 / 0.337298 | 0.297101 / 0.192071 / 0.315329 | 聚合报告；speaker lines, left truncation, max_length=256; training-only micro-F1 calibration |
| weighted_512_training | 0.411923 | 0.391249 / 0.612999 / 0.231521 | 0.304127 / 0.213049 / 0.282764 | 聚合报告；speaker lines, left truncation, max_length=512; micro-F1 calibration |
| weighted_512_exact_calibration | 0.626451 | 0.610875 / 0.750212 / 0.518267 | 0.046774 / 0.036090 / 0.021962 | 聚合报告；speaker lines, left truncation, max_length=512; exact calibration |
| unweighted_512_training | 0.417021 | 0.429482 / 0.587935 / 0.233645 | 0.300964 / 0.196721 / 0.336806 | 聚合报告；speaker lines, left truncation, max_length=512; micro-F1 calibration |
| unweighted_512_exact_calibration | 0.628292 | 0.614698 / 0.750637 / 0.519541 | 0.008562 / 0.038981 / 0.011878 | 聚合报告；speaker lines, left truncation, max_length=512; exact calibration |

Day3 dry-run 仅 32 条：macro exact 0、空画像 0.666667、情感 Accuracy 0.3125/Macro-F1 0.029762，不与全量排名。Day3 另有训练内 exact 校准摘要 0.6223，缺分维/逐条结果，单独列在 JSON，不扩大结论。
历史正类预测数显示过度输出：Day3 三维 1780/829/2256，对应金标正类数 1143/643/1333；Day4 256 不加权为 2445/1148/1870。它们是标签个数，不是非空样本数。Day4 512 不加权 exact 校准后仅预测 25/24/14 个标签，macro Micro-F1 0.019807，不能以非空输出说明超过空画像。
空画像每维及全局 precision/recall/micro/macro F1、预测非空率均为 0；金标为空时 exact=1、非空时 exact=0。完整值及历史聚合原字段见 JSON。旧画像头的 precision/recall、标签 macro F1、非空样本率和关键切片缺失，明确不可回放；本任务不启动模型补推理。

## 情感诊断与混淆

| 来源 | Accuracy | Macro-F1 | 范围/可比性 |
|---|---:|---:|---|
| 多数类 anxiety | 0.319031 | 0.030233 | 同一全量金标；本次重算 |
| old_conditions_unknown_provenance | 0.709431 | 0.573752 | full ID-aligned replay, model/weight/preprocessing provenance unconfirmed; do not rank; matches_report=unknown |
| Day3_archive | 0.709431 | 0.573752 | full ID-aligned replay; archived emotion metrics must match C report; profile is deployed empty baseline, not old profile heads; matches_report=True |
| Day5_archive | 0.716653 | 0.584843 | full ID-aligned replay; archived emotion metrics must match C report; profile is deployed empty baseline, not old profile heads; matches_report=True |

本地归档 Day3/Day5 完整 ID 对齐，情感 Accuracy/Macro-F1/逐类指标与 C 报告核对；归档输出的画像为空，不能误称旧画像头预测。旧 conditions 文件虽然可按 ID 重算，但权重/预处理来源不足，单独标为不可排名。Day3/Day5 的训练配置与输入预处理一致性依据 C 报告及脚本；此次未复载权重，不能据回放声称当前代码推理已复验。

| 类别 | Train支持 | Val支持 | Day3 P/R/F1 | Day5 P/R/F1 |
|---|---:|---:|---|---|
| anger | 855 | 115 | 0.716667 / 0.747826 / 0.731915 | 0.741071 / 0.721739 / 0.731278 |
| anxiety | 6340 | 751 | 0.774235 / 0.808256 / 0.790879 | 0.745121 / 0.864181 / 0.800247 |
| care | 79 | 11 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| disgust | 125 | 21 | 0.937500 / 0.714286 / 0.810811 | 0.730769 / 0.904762 / 0.808511 |
| fear | 1388 | 124 | 0.569697 / 0.758065 / 0.650519 | 0.615942 / 0.685484 / 0.648855 |
| gratitude | 694 | 118 | 0.778689 / 0.805085 / 0.791667 | 0.803571 / 0.762712 / 0.782609 |
| helplessness | 1575 | 142 | 0.648276 / 0.661972 / 0.655052 | 0.684211 / 0.640845 / 0.661818 |
| joy | 171 | 20 | 0.500000 / 0.450000 / 0.473684 | 0.458333 / 0.550000 / 0.500000 |
| loneliness | 876 | 115 | 0.767241 / 0.773913 / 0.770563 | 0.789474 / 0.782609 / 0.786026 |
| mixed | 945 | 177 | 0.432099 / 0.197740 / 0.271318 | 0.461538 / 0.101695 / 0.166667 |
| neutral | 654 | 53 | 0.360656 / 0.415094 / 0.385965 | 0.375000 / 0.339623 / 0.356436 |
| pride | 234 | 40 | 0.909091 / 0.250000 / 0.392157 | 0.846154 / 0.275000 / 0.415094 |
| relaxed | 2816 | 409 | 0.752137 / 0.860636 / 0.802737 | 0.733740 / 0.882641 / 0.801332 |
| sadness | 1085 | 131 | 0.632075 / 0.511450 / 0.565401 | 0.688172 / 0.488550 / 0.571429 |
| shame | 752 | 118 | 0.669065 / 0.788136 / 0.723735 | 0.702290 / 0.779661 / 0.738956 |
| surprise | 64 | 9 | 1.000000 / 0.222222 / 0.363636 | 0.625000 / 0.555556 / 0.588235 |

重点混淆（各类支持数作分母，箭头为实际预测）：

- Day3_archive / mixed → anxiety=58, mixed=35, relaxed=30, neutral=12, sadness=10, shame=9, anger=7, helplessness=7, gratitude=3, joy=2, disgust=1, fear=1, loneliness=1, pride=1。
- Day3_archive / care → anxiety=8, anger=1, fear=1, neutral=1。
- Day5_archive / mixed → anxiety=69, relaxed=37, mixed=18, neutral=10, sadness=9, shame=8, anger=7, helplessness=6, joy=4, gratitude=3, surprise=2, disgust=1, fear=1, loneliness=1, pride=1。
- Day5_archive / care → anxiety=9, anger=1, neutral=1。

Day5 相比 Day3 Accuracy +0.007222、Macro-F1 +0.011091，但 mixed F1 从 0.271318 降为 0.166667、recall 从 0.197740 降为 0.101695；care 两者 F1=0。不能只以总体准确率宣称全面改善。完整 16×16 混淆及历史/画像空非空/情感切片见 JSON predictions；这些历史成绩不是新盲测。

## 固定训练内校准与筛选清单

种子 20261015；旧画像既有算法按 sorted(unique conversation_id) 后 Random(seed).shuffle，首 round(n×0.15) 个会话校准，剩余 fit。实际 fit 15944 条/7080 会话，calibration 2709 条/1249 会话，交集 0；验证条目进入训练内清单为 0。清单覆盖每条 B1 train，保留原 sample_id、会话、来源和 subset，不复制正文。后续 C 必须实际消费该清单或验证自身重建结果相同，B 未修改 C 训练程序。

固定筛选 160 条/130 会话。每类目标 floor(160/16)=10；surprise 仅 9 条，缺额按类别支持数降序补给 anxiety（11 条）；care=10、mixed=10，其余每类 10。每类按 history×三维非空 mask 联合分层从稀有到常见轮转选一条，组内先按 ID 排序再用同一 Random(seed) 流依联合分层排序打乱，结果按 ID 排序。选择只读取金标分层、不读取候选成绩；完整规则与种子在 JSON，测试验证输入反序不影响输出。

历史覆盖 {'long_16_plus': 56, 'medium_6_15': 55, 'short_1_5': 49}；三维 mask（personality/interests/style，1=非空）覆盖 {'000': 15, '001': 17, '010': 20, '011': 23, '100': 19, '101': 18, '110': 28, '111': 20}。269 个可用联合分层覆盖 147 个，未覆盖 122 个，逐项列于 JSON screening.uncovered_joint_strata。160 条预算无法覆盖所有联合层，稀有优先导致样本不代表总体分布；不得把小集分数当作全量提升证明。该筛选集与剩余验证集可能共享会话，不用于阈值拟合；训练 fit/calibration 始终会话隔离。

## 冻结的本地采用规则

1. 冻结日期 2026-10-22（Asia/Shanghai），只针对第二周期 B T1；尚无本周期新候选结果。
2. 主指标固定为三个画像维度的集合 exact match 的等权宏平均；同一 B1 验证 ID、完整 2354 条、相同金标、预处理和评测协议下比较空画像、历史头、新候选。数组顺序不影响得分。
3. 本次重算核实空画像基线 0.628292（未舍入值见 JSON）。只在上述同口径下用作门槛；换数据或协议须重算空画像，不搬用此数值。历史报告与新候选若缺少输入/预处理证据，隔离标记，不直接排名。
4. 候选 macro exact match 必须严格超过同集空画像；相等或更低原则上 no-go，保留空画像。比较未舍入值，避免六位小数掩盖持平；非空预测率提高不构成提升。超过空画像只是必要条件，逐维/关键分层退步和稳定性仍需独立复核。
5. 同时报告每维 exact、micro precision/recall/F1、固定标签 macro precision/recall/F1、全局 micro 及跨维宏平均、预测非空率、金标空/非空、长度/情感切片和非空精度。无预测时相关指标为缺失，不能以预测标签个数冒充非空样本率。零分母按 0 计，未支持标签也计入固定标签宏平均。
6. 情感固定报告 Accuracy、16 类 Macro-F1、逐类 precision/recall/F1/support 和混淆矩阵（行金标、列预测），单列 mixed/care；不能仅凭 Accuracy 选型。T1 不另设情感数值硬门槛。
7. 阈值、融合、checkpoint 及早停只能用训练内 conversation-disjoint 校准；验证 2354 条及筛选 160 条均禁止阈值拟合。筛选集只用于固定候选淘汰，不能作为最终提分证明或新盲测。不得事后换主指标、挑样本或按新结果改清单。
8. 建议（B 本次设计值，非既有团队决策）：沿用旧画像 seed=20261015/15% 会话留出；筛选 160 条、每情感约 10 条，按历史长度×三维空非空细分均衡，稀有类不足时按规则补足。依据为复用旧划分与计划规定的 128–200 条预算；不增加其他无依据数值采用阈值。
9. 旧情感编码器已在完整 B1 训练集拟合，旧画像头的校准会话可能被其见过；不能把旧冻结头校准称为全模型未见。新候选应先划分，再只在 fit 学习全部可训练参数；若沿用见过校准的起点，显式披露，公平重训由 C 后续负责。
10. Day4 历史 Micro-F1>0 阶段交接门槛保留为历史事实，不代替本周期 exact 采用规则；官方评分定义未知，本地结果不等于官方分数或隐藏集提升。T2 会话级配对比较留待下一任务。

## 溯源、复现与交接

本报告和 JSON 是 B T1 实验诊断交付，不是独立验收记录；本地测试、重跑哈希证据和差异审查记录放 .codex_tmp/acceptance/cycle2/，不进普通 Git。输入文件逐个真实路径、字节数及 SHA-256 在 JSON sources；B1 原始/处理文件全部与 manifest 核对。源脚本和指标实现也列哈希，两个清单列输出哈希。

```powershell
python -B scripts/b_t1_profile_diagnostics.py --seed 20261015 --screening-size 160
python -B -m unittest discover -s tests -p test_b_t1_profile_diagnostics.py -v
python -B -m unittest discover -s tests -v
python -B scripts/ci_smoke.py
git diff --check
```

生成文件：reports/B_T1_PROFILE_DIAGNOSTICS.md、reports/b_t1_profile_diagnostics.json、reports/b_t1_train_internal_split.jsonl、reports/b_t1_screening_manifest.jsonl。生成前严格核对输入；重跑会更新这四个 B T1 产物，不写原始数据与历史产物。跨机器缺本地归档时明确 missing，不能得到同一历史回放部分；复现需先核对 sources 哈希。
已完成 T1 B 的历史诊断、清单冻结、采用规则。未完成项限于缺失证据：旧画像头逐条预测及其切片；模型推理时真实输入 token/截断比例；官方评分实现/权重；全模型未见校准的旧对照。这些未伪造、未用训练或大推理补齐。下一步由 C 在既定划分上训练并交付概率、阈值和完整 ID 预测；B 按 T2 做配对工具、T3 独立复核；D 使用固定筛选清单生成输入时剔除分层金标，仅取当轮及此前历史。
