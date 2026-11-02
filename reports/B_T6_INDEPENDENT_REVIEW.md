# B-T6：现有完整预测的独立 CPU 评测

日期：2026-11-01（Asia/Shanghai）。分支 `feature/b-eval-tooling`；HEAD `45e60acaae26f72540d9b0bff1f80121d30edb08`。

**预测级状态：`EVALUATION_COMPLETE_WITH_PROVENANCE_CAVEAT`。** Overall、history、emotion、long-response、paired comparison 和 regression analysis 均已实际完成。正式冻结配置的无保留验收仍为 `BLOCKED`；不标团队 full DONE。

固定组合：`KEEP_DAY5 / KEEP_D4V2 / OFF / empty`，`max_new_tokens=256`。推荐 `SELECT_EXISTING_D4V2_CONDITION_OFF`、`NO_NEW_D_CANDIDATE_SELECTED`、`GPU_NOT_NEEDED_REUSE`。历史 T4 longweight 仅用于已否决方案的配对诊断，不重新入围，不与 D4v2 自身作虚假 A/B。

## 1. Prediction integrity

D4v2 prediction 与 validation 均 2354 条、2354 unique IDs，ID 集合与顺序相同；duplicate/missing/extra/empty 均为 0，validation 覆盖 900 会话。T4 历史对照也通过完整 ID、顺序、空输出检查。两臂 journal 的校验链、逐条输出、完成事件、manifest digest 与报告绑定再次通过现有实现核验。

- Prediction raw SHA-256：`a55274d3d3792dc1df28a2fb797a6b10b18e8688ce6c01eb1cfb24a94acf2b34`。
- Prediction LF SHA-256：`1cfd430149c31d89a827dacb7e75e3e0eb6cab05a2abb8863f74871bd7520ce2`。
- Validation SHA-256：`ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2`。

Publication：原始 CRLF checkout 的字节校验仍失败；既存 LF materialization 的官方 validate_publication 再验 PASS，LF 内容与归档输出绑定。没有改写原始 prediction，也没有重新发布。完整性证明归档文本和输入绑定可信，不等同于已独立证明全部生成资产身份。

## 2. Model identity confidence

**`PARTIALLY_CONFIRMED`，完整身份验收 `NOT_VERIFIED`。** 本轮用户明确确认 adapter inventory 已独立核验、adapter 权重 SHA 已匹配、base model.safetensors SHA 已有历史 preflight 记录；据此接受这些事实。仓库配置及历史 D T7 报告提供交叉支持，本机没有权重实体或新交接 preflight 原件，因此不把它们描述为本轮重新哈希或新增本地 PASS。

- Adapter SHA：`6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e`。
- Base weights SHA：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。
- 匹配历史代码：`64be735`，与 journal code fingerprint 一致。

Base 完整目录 inventory/digest 无法重建，不改变固定 prediction/gold 的 BLEU、ROUGE、切片及配对结果。Adapter inventory 与权重 SHA 足以支持候选资产身份的部分确认，但不能自动证明运行时 base tokenizer/template 的每个字节。历史模板等价性核对可作佐证，尚不能无保留写成生成时配置 PASS。不同目录摘要算法不可直接比较。

## 3. Generation reproducibility

**`NOT_FULLY_VERIFIED`。** manifest digest 重算匹配：`7c452024a25a6b69ce79558b35604ff4e15e507722de097c1a84cedc302025b4`。这证明 manifest 的一致性，不能据此声称从所有已验证资产重建了完整历史生成配置。

| Setting | Historical value | Evidence / impact |
| --- | --- | --- |
| do_sample | false | 匹配历史代码显式传入 |
| max_new_tokens | 256 | manifest/report |
| temperature | UNKNOWN | 无显式覆盖；greedy 下采样项非 operative，不阻塞文本评测 |
| top_p | UNKNOWN | 无显式覆盖；greedy 下采样项非 operative，不阻塞文本评测 |
| top_k | UNKNOWN | 无显式覆盖；greedy 下采样项非 operative，不阻塞文本评测 |
| repetition_penalty | UNKNOWN | 可能改变 greedy logits/token 选择；影响历史复现，不改变固定文本的指标 |

进一步核对的默认值路径：历史 AutoModelForCausalLM.from_pretrained(base) → 可选 PeftModel wrapper → model.generate；调用中没有 repetition_penalty 覆盖。其有效值来自所加载 generation config、model config 或库回退。manifest 记录 Transformers 5.17.0，本机未安装该库且缺历史有效配置，无法确认具体路径及数值；不以常见默认值填补 UNKNOWN。即使 do_sample=False，也不能保证跨环境逐字节复现，历史 deterministic_algorithms=false。

计划与规范检查：`docs/EXECUTION_PLAN_CYCLE2.md` 要求复用匹配预测、遵守既定指标和门槛；`docs/INFERENCE_RESUME.md` 的严格指纹用于生成/恢复和发布绑定，没有禁止对完整归档文本作描述性评分。本轮不修改恢复契约、旧 gate、预注册门槛或资产。

## 4. CPU evaluation results

复用 src.evaluation.metrics.bleu4/rouge_l，按 sample ID 对齐后取逐样本算术均值。两指标均使用移除全部空白的字符序列；BLEU 为现有平滑句级 BLEU-4 的均值，不是 corpus BLEU、官方分数或人工质量分。

| Metric | Selected D4v2 | Historical T4 | Δ T4−D4v2 |
| --- | --- | --- | --- |
| bleu4_char | 0.1122403823 | 0.1130609533 | 0.0008205711 |
| rouge_l_char_f1 | 0.2603151693 | 0.2605315545 | 0.0002163853 |

长度定义分别保留，禁止统称同一个 mean length：

| Definition | D4v2 | T4 |
| --- | --- | --- |
| raw len(text) | 134.3742565845 | 137.2782497876 |
| stripped len(text.strip()) | 134.3742565845 | 137.2782497876 |
| non-whitespace len(''.join(text.split())) | 129.7523364486 | 132.4944774851 |

本次 raw 与 stripped 恰好相同，因为生成时 decode 后已 strip；内部空白仍在，因此与 non-whitespace 不同。

### History slices

严格复用 B T1：含目标 user 的 context_turns 条数 ≤5 / 6–15 / ≥16；不采用 D 筛选集分桶。

| Gold slice | N | D4v2 BLEU-4 | D4v2 ROUGE-L | T4 ROUGE-L | Δ T4−D4v2 |
| --- | --- | --- | --- | --- | --- |
| short_1_5 | 903 | 0.1057172389 | 0.2511026872 | 0.2508667433 | -0.0002359440 |
| medium_6_15 | 1098 | 0.1133785254 | 0.2632454279 | 0.2640435063 | 0.0007980784 |
| long_16_plus | 353 | 0.1253868901 | 0.2747668612 | 0.2743309920 | -0.0004358692 |

### Emotion slices

按固定 classification_val gold 标签切片，未按预测标签选样本。mixed/care 显式保留；care=11、surprise=9 等小样本不能用于稳定提升断言。

| Gold slice | N | D4v2 BLEU-4 | D4v2 ROUGE-L | T4 ROUGE-L | Δ T4−D4v2 |
| --- | --- | --- | --- | --- | --- |
| anger | 115 | 0.1059483254 | 0.2519707575 | 0.2545207380 | 0.0025499805 |
| anxiety | 751 | 0.0894756930 | 0.2264855107 | 0.2269060511 | 0.0004205404 |
| care | 11 | 0.0864745651 | 0.2277312435 | 0.2379172640 | 0.0101860204 |
| disgust | 21 | 0.0981922089 | 0.2370288202 | 0.2206419247 | -0.0163868955 |
| fear | 124 | 0.1033324994 | 0.2425292675 | 0.2397445273 | -0.0027847402 |
| gratitude | 118 | 0.1475665995 | 0.3238983073 | 0.3254276087 | 0.0015293015 |
| helplessness | 142 | 0.0984584339 | 0.2458295476 | 0.2512721508 | 0.0054426032 |
| joy | 20 | 0.0765491591 | 0.2279561718 | 0.2271278851 | -0.0008282866 |
| loneliness | 115 | 0.0974233318 | 0.2337563113 | 0.2357170192 | 0.0019607079 |
| mixed | 177 | 0.1311272906 | 0.2801190072 | 0.2755233202 | -0.0045956870 |
| neutral | 53 | 0.1783627211 | 0.3561653554 | 0.3597700055 | 0.0036046500 |
| pride | 40 | 0.1198891352 | 0.2849178109 | 0.2640746339 | -0.0208431770 |
| relaxed | 409 | 0.1457143875 | 0.3125380637 | 0.3139343696 | 0.0013963059 |
| sadness | 131 | 0.1151640472 | 0.2554502816 | 0.2582081937 | 0.0027579121 |
| shame | 118 | 0.1019989516 | 0.2425381367 | 0.2409920690 | -0.0015460677 |
| surprise | 9 | 0.0778418571 | 0.2025982003 | 0.2050024627 | 0.0024042624 |

### Gold response length slices

按 gold len(response_text.strip()) 分桶；>200 support=528，不按预测长度重选。

| Gold slice | N | D4v2 BLEU-4 | D4v2 ROUGE-L | T4 ROUGE-L | Δ T4−D4v2 |
| --- | --- | --- | --- | --- | --- |
| <=6 | 15 | 0.3496122164 | 0.5419432573 | 0.5334710893 | -0.0084721680 |
| 7-20 | 65 | 0.3036907459 | 0.5157584800 | 0.5141290169 | -0.0016294631 |
| 21-50 | 298 | 0.1976534078 | 0.3965123564 | 0.4009562321 | 0.0044438757 |
| 51-100 | 822 | 0.1072004072 | 0.2628705098 | 0.2605185070 | -0.0023520028 |
| 101-200 | 626 | 0.0817838232 | 0.2139010251 | 0.2156143952 | 0.0017133701 |
| >200 | 528 | 0.0776773761 | 0.1950495746 | 0.1955777379 | 0.0005281632 |

冻结 D T4 gate：primary `>200` ROUGE-L delta ≥+0.01，实际 `0.0005281632` → **FAIL**；overall safety delta ≥−0.005，实际 `0.0002163853` → **PASS**。最终采用状态 **`NO_GO`**。不因微小正值改动门槛或换主指标。

### Paired comparison and regression analysis

方向固定为历史 T4 minus D4v2；沿用现有 compare_response_arms，零差值为 tie。

| Metric | Win | Tie | Loss | Mean delta | Median delta |
| --- | --- | --- | --- | --- | --- |
| bleu4 | 1026 | 280 | 1048 | 0.0008205711 | 0.0000000000 |
| rouge_l | 1038 | 300 | 1016 | 0.0002163853 | 0.0000000000 |

ROUGE delta p25=-0.0188556042，p75=0.0184596239；回复逐字相同 268 条。上述为 metric win/tie/loss，不称人工质量胜负。

退步诊断按总体、历史桶、情感桶保存 metric loss/improvement/tie 数、退步/改善均值及格式标签新增/消除计数，全部为汇总，不保存 response 正文。

| Diagnostic | D4v2 | T4 |
| --- | --- | --- |
| empty_response | 0 | 0 |
| near_empty_count | 40 | 36 |
| low_cjk_count | 1 | 3 |
| markdown_artifact | 372 | 381 |
| repetition | 32 | 34 |
| too_short_pred_gold_ratio_lt_0p7_count | 616 | 609 |

格式复用 error_analysis 默认规则：near-empty≤6、low-CJK<0.5、重复句 min6、重复 8gram min3、既存 Markdown patterns。格式标签是诊断标记，不直接等同错误质量或安全事件。各类新增/消除计数见 JSON regression_analysis。

### Fixed classification/profile checks

固定组合逐条核对 response=D4v2、profile=empty；按现有 gold 复算 Day5 Accuracy=0.7166530000、Macro-F1=0.5848430000；empty profile macro exact=0.6282922685。不据此宣称 C 新候选复算完成。C T6 Day5 冻结交接仍 `INTEGRATION_STATUS_PENDING`；C T5 的 0.65/profile_v1 稳定性检查不替代该交接。

## 5. Comparability limitations

两个归档臂的 input、base directory digest、历史 code/environment、显式 decoding config 一致，adapter 不同；可比较这些固定文本。未重建完整目录不会改变共同摘要一致的事实，但各 adapter 下继承 generation defaults/PEFT 行为未完全验证，不能无保留宣称严格单变量因果效应。

Validation 已多次用于历史实验，不是新盲测；小类别及切片结果不证明隐藏集收益。未做人工内容评审。会话级不确定性留在 T7，本轮不新增置信门槛。

资源仅引用既有证据：D4v2 4.2881 s/sample、19.35 tok/s，为历史成功 generate 调用耗时，非冷启动；5,299MB 峰值属于组合 workload 采样范围，不能声称 D4v2 单臂独立峰值。本轮未做资源重测。

## 6. T6 final recommendation

**预测级独立评测可以完成，且本轮已完成：`EVALUATION_COMPLETE_WITH_PROVENANCE_CAVEAT`。** 保持 D4v2/condition OFF 和固定 Day5/empty；`SELECT_EXISTING_D4V2_CONDITION_OFF`、`NO_NEW_D_CANDIDATE_SELECTED`、`GPU_NOT_NEEDED_REUSE`。

可接受为非阻塞 caveat：完整 base 目录重建缺失，以及 greedy 下 temperature/top_p/top_k 数值缺失。Adapter inventory/权重和 base preflight 的已确认事实支持部分身份，不需要重新取得全部权重才允许评分。

**正式冻结配置无保留验收仍 `BLOCKED`，阻塞码 `HISTORICAL_RUNTIME_CONFIG_NOT_FULLY_VERIFIED`。** 它只限制“归档生成精确实现全部冻结配置”的声明：runtime base tokenizer/template 的精确对应及 effective repetition_penalty 仍未完全确认，C T6 冻结交接也单独待确认。只需针对性的已有配置证据或责任方明确接受可复现性边界；不把全目录 inventory 再设为所有评测的绝对门槛。不能将未知配置写为 PASS，也不因此否定已经完成的文本评测。

## 7. Validation and operation boundary

- 相关测试：47/47 PASS（compare_response_arms、response_error_analysis、core、B T1 diagnostics、inference_resume；恢复测试使用 fake model）。
- compileall：PASS；CI smoke：PASS；JSON parse：PASS；Markdown/JSON consistency：PASS；git diff --check：PASS。
- 汇总交叉检查：各切片 support 合计 2354、加权均值回到总体、paired 计数闭合、总体与 T5 独立结果一致（误差 <1e−14）。
- 既有全套 489 tests 中的平台/依赖失败与 skip 记录保留；本轮未重跑全套，不声称全套通过。
- 未训练、未做真实模型推理、未用 GPU、未重生成或修改预测、未 add/commit/push；仅新增获准两份正式报告与忽略目录内验收材料。

既有 identity_gate.json / IDENTITY_GATE.md / CONTINUED_PROVENANCE.md 原样保留。它们的“缺历史身份链即不运行任何指标”的决策由本轮用户授权重新评估；仅预测评分阻塞被解除，历史 UNKNOWN 和复现 caveat 没有被改成 PASS。详细指纹、定义、完整切片与退步汇总见 `b_t6_independent_review.json`；本轮执行材料在 `.codex_tmp/acceptance/cycle2/b_t6_20261101/cpu_review/`。
