# B-T5 独立复核与 T6 readiness gate

日期：2026-11-01（Asia/Shanghai）。依据 `docs/EXECUTION_PLAN_CYCLE2.md`。本轮 B-T5 审核交付已完成；C 候选独立复算因缺逐条证据阻塞。未生成 B-T6 正式报告，未执行 GPU 推理或 Git 写操作。

## 1. Git / PR snapshot

| 项 | 值 |
| --- | --- |
| branch | feature/b-eval-tooling |
| HEAD | 75a8135f6bf424cb63347a68c2f1b00d68c7b69a |
| origin/main | 75a8135f6bf424cb63347a68c2f1b00d68c7b69a |
| origin/pr38 | c584bedf2a1ed0c61c3eb97d06f54e9b6fd2ee32 |
| origin/pr39 | a969f9d5f1dfb319e1e85c37cd6447fd9dd6ed6c |
| pr38_in_main | True |
| pr39_in_main | True |
| initial_status | CLEAN |

初始工作区干净；PR38/PR39 均已进入 main，未 checkout、merge 或 fetch 上游。工作结束仅新增本报告与 JSON。

## 2. C T4 evidence identity

结论：`BLOCKED`，阻塞码 `C_T5_BLOCKED_MISSING_T4_PREDICTION`；采用建议 `KEEP_DAY5`。这是证据不足时保留已冻结基线，不表示候选已完成独立评估。

PR38 的 T4 报告、structured JSON、config、16 类混淆矩阵存在；但 PR38 tree、当前 results/artifacts/weights 和本地 C 快照均无该候选的逐 ID prediction / emotion probabilities。C T5 stability 报告不能充当 T4 候选预测。

候选 checkpoint：`weights/emotion_longtail065_b1_20261028/`，本地不存在。上游声明 model SHA `bc0c7cab2259d2aea5ca77e812caff1363575da48ee94b92c03e84c39709e895`，B 未独立重哈希。唯一变量 exponent 0.65；seed 20261015，1 epoch / 583 steps，batch 32，LR 5e-6，max_length 256。候选 prediction 路径、SHA、行数、unique IDs、duplicate/missing/extra 均 UNKNOWN，不能把缺失文件写成 0 行或 0 错误。

| 文件 | raw SHA-256 | LF SHA-256 |
| --- | --- | --- |
| reports/c_cycle2_t4_emotion_weight065_20261028.json | 0d805d3e85352b9ff6498e2918eeef45b586e0d24302a3670830512f4cb74d14 | b65ae850a3e0e657ae06b1a93dba871a129089ab2f4f76a30b83440654a7b389 |
| data/processed/b1_full_v1/classification_val.jsonl | 30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201 | 30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201 |
| results/response_gen/c5_conditions_val_full.jsonl | e55a59f68e4c669a68f21b665912684cc076a41aaa6c66843c2600f3a5b3a320 | 478e861a344647d655f9c4fbfee78d81812942e7c8c2282789564e39137ce7ea |
| results/response_gen/c5_conditions_val_full_report.json | 88b0d3fbb133330c919ae105d0618cb72aabccd1d475baac47cbaec95ef67432 | 4ca53bb832b5c69eb4de403f7955fbf1ad364882751ad03c4dd30101ea9714ab |

Day5 checkpoint 来源：`weights/classification/c_day5_longtail075`；模型声明 SHA `aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37`，与 `configs/integrated_artifacts.json` 一致。Day5 prediction 为 2354 行 / 2354 unique IDs，duplicate/missing/extra=0，标签均符合 frozen schema，ID 集合和顺序与 gold 一致。

## 3. C overall / mixed / care

B 对已有 Day5 逐条预测调用正式 `src.classification.emotion_encoder.classification_metrics`，结果与原报告完全一致。C T4 不具备逐条证据，candidate 与 delta 留空；下表不把上游聚合值冒充独立复算。

| 指标 | Day5 独立复算 | T4 独立复算 | delta |
| --- | --- | --- | --- |
| Accuracy | 0.7166530000 | BLOCKED | N/A |
| Macro-F1 | 0.5848430000 | BLOCKED | N/A |

| label | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| anger | 0.7410710000 | 0.7217390000 | 0.7312780000 | 115 |
| anxiety | 0.7451210000 | 0.8641810000 | 0.8002470000 | 751 |
| care | 0.0000000000 | 0.0000000000 | 0.0000000000 | 11 |
| disgust | 0.7307690000 | 0.9047620000 | 0.8085110000 | 21 |
| fear | 0.6159420000 | 0.6854840000 | 0.6488550000 | 124 |
| gratitude | 0.8035710000 | 0.7627120000 | 0.7826090000 | 118 |
| helplessness | 0.6842110000 | 0.6408450000 | 0.6618180000 | 142 |
| joy | 0.4583330000 | 0.5500000000 | 0.5000000000 | 20 |
| loneliness | 0.7894740000 | 0.7826090000 | 0.7860260000 | 115 |
| mixed | 0.4615380000 | 0.1016950000 | 0.1666670000 | 177 |
| neutral | 0.3750000000 | 0.3396230000 | 0.3564360000 | 53 |
| pride | 0.8461540000 | 0.2750000000 | 0.4150940000 | 40 |
| relaxed | 0.7337400000 | 0.8826410000 | 0.8013320000 | 409 |
| sadness | 0.6881720000 | 0.4885500000 | 0.5714290000 | 131 |
| shame | 0.7022900000 | 0.7796610000 | 0.7389560000 | 118 |
| surprise | 0.6250000000 | 0.5555560000 | 0.5882350000 | 9 |

上游 T4 自报（未由 B 独立复算）：Accuracy 0.714953、Macro-F1 0.590474、mixed F1 0.212766（support 177）、care F1 0（support 11）。仅作证据索引，不参与独立提升判定。完整标签 schema、Day5 逐类指标及混淆矩阵在 JSON。

## 4. C paired / history / conclusion

baseline correct → candidate wrong、baseline wrong → candidate correct、both correct、both wrong，以及 mixed/care paired changes 全部 BLOCKED。缺少 candidate 的 sample-ID join，不能从混淆矩阵恢复配对结果。

历史长度严格复用 B T1 冻结桶：context_turns（含目标用户轮）1–5 / 6–15 / 16+；不与 D 160 条分层抽样的 ≤4 / 5–12 / >12 桶混用。

| history bucket | support | Day5 Accuracy | Day5 Macro-F1 | T4 |
| --- | --- | --- | --- | --- |
| short_1_5 | 903 | 0.7043190000 | 0.5846440000 | BLOCKED |
| medium_6_15 | 1098 | 0.7313300000 | 0.5667710000 | BLOCKED |
| long_16_plus | 353 | 0.7025500000 | 0.4903590000 | BLOCKED |

C signal=`BLOCKED`；classification=`KEEP_DAY5`。解除阻塞需要固定 checkpoint/config/validation 对应的完整 T4 逐条预测；本轮不重跑模型。

## 5. D T4 evidence / identity

同卡 full validation 与 screening160 两组均存在；不混用历史 bitahub 部分预测和不同设备的旧 baseline。以下四份预测均 input ID 集合与顺序相等，duplicate/missing/extra=0，empty=0。完整 raw/LF 文件、report、publication marker、journal manifest/records/attempts SHA 在 JSON。

| 用途 | 路径 | records / unique IDs | raw SHA-256 | LF SHA-256 |
| --- | --- | --- | --- | --- |
| candidate_identity | results/response_gen/t4_longweight_val_newcard.jsonl | {'path': 'results/response_gen/t4_longweight_val_newcard_progress/records.jsonl', 'raw_sha256': '6fe12ed456989baa4eb1608ffe659eaab680476a38a43dd165379d3f78455244', 'lf_sha256': '9b2d122846954f24c6b8f2c0b7a1db582f963dfd973824073eb756cb5ee9fd36', 'bytes': 1446200} / 2354 | f0d2cbd7426bb7aa42f0e58ac40e3c9675812fcf01ec0b81dcc6e9f65cb377bd | e55c2c17b826b67489e15827ad29f0add4bbd5b823512ef58753f3411bf5d19f |
| baseline_identity | results/response_gen/d4v2_val_newcard.jsonl | {'path': 'results/response_gen/d4v2_val_newcard_progress/records.jsonl', 'raw_sha256': '6d94d7ced1cd75842667e05da32d08aaa9a19a2e0d44b72eca821039f06cadb2', 'lf_sha256': 'a2b1d8602b52f53f0d07efce40142d1a58c12c568aeb415c5c3821ed59d7fab5', 'bytes': 1426639} / 2354 | a55274d3d3792dc1df28a2fb797a6b10b18e8688ce6c01eb1cfb24a94acf2b34 | 1cfd430149c31d89a827dacb7e75e3e0eb6cab05a2abb8863f74871bd7520ce2 |
| screen_candidate_identity | results/response_gen/t4_longweight_selection_newcard.jsonl | {'path': 'results/response_gen/t4_longweight_selection_newcard_progress/records.jsonl', 'raw_sha256': '268e001524df57b8a67a97b6b327cf3cf327498262639f5f85b515d44747b08b', 'lf_sha256': '37e5e7be68776bae8a2016a916ac9fa027be6d2ee2bd446cccb17b56d762a5d6', 'bytes': 95270} / 160 | 9565bb77ded6f5f480d2fb437f784a50dd33599f5ec2fac2ae107af9c2f6732e | 6954c70dfbb5a16ce3d71c81bd5dab15dbd3b59a7dcbb9699a137292c9888830 |
| screen_baseline_identity | results/response_gen/d4v2_selection_newcard.jsonl | {'path': 'results/response_gen/d4v2_selection_newcard_progress/records.jsonl', 'raw_sha256': '1a1b8293852bfb0b9e941fb2519bba20a078b33302773557e4f12a14cc78cf60', 'lf_sha256': 'ad82f2fea4506d4153cef0489618fe8927bb4e239be4cafd7acef26b2c2f0d7d', 'bytes': 93641} / 160 | ae3ab11f7bbc4748398f7117f873b5e925f6d5255eba3ff33b2c6f5895e29c4e | 3d4211f33fdf581d032ecb78bba97bdb08744139601325947a5b4dcd4160f3a0 |

| 输入/gold | raw SHA-256 | LF SHA-256 |
| --- | --- | --- |
| data/processed/b1_full_v1/response_val.jsonl | ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2 | ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2 |
| results/response_gen/t2_selection_input.jsonl | 8d1d487e4e2fb7d66071d72c8d9078744831751bfed7c9fb3103c067d9163959 | 014e52ab7d176c84a91b1c2a830b456c829d062c05dfeba80f670a309bc86a53 |

validation 完整覆盖固定 2354 条 / 900 会话；screening160 为冻结 160 条 / 160 会话，seed 20261022，逐条内容与完整 response_val 中对应记录相等。

candidate adapter 声明 SHA `bc6ca0234eb7222e526e651364326131c7b98f5d56c18a6301ca8c4b193d2f4e`；D4v2 声明 SHA `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e`，与训练报告和 launcher 钉值一致。生成 journal 的 candidate adapter tree=`3783131711634eb273eaff6b060a67c0f30754086fb886ca51e39ad57cd9db91`，baseline tree=`ae06454fee5063905c4eb62fa74318159e19941f8b4ccc797ae20c355c5ddc5c`；full/160 对应同 adapter tree，8 个臂共用 model tree=`4ebff76fb9b70c855dbce11d68db3c08c024d5f33a10c4c6c22035e950122b34` 与代码 tree。

生成配置：condition_mode=none，max_new_tokens=256，min_new_tokens=0，greedy do_sample=False、use_cache=True。T4 训练配置的 max_new_tokens=512 是训练配置项，不冒充实际生成的 256。两 full 臂和所有 selection 臂 journal checksum chain 均通过，逐条 journal prediction 等于输出；每个臂仅一次 start/end，end=completed。

身份边界：上述复算绑定真实 prediction/config/journal；本地 weights 仅有 README，无法独立重哈希模型文件或核对远端上传 manifest。上游报告/launcher 的 adapter SHA 不能升级为本轮已重哈希。tokenizer/chat template 包含在生成 model-tree 摘要内，但缺少可解释到逐文件的运行时 inventory。

## 6. D full-validation independent metrics

复用 `src.evaluation.metrics.bleu4/rouge_l`，每条按 sample ID join 后取均值。长度沿用错误分析口径 `len(text.strip())`；另保存 non-whitespace mean 以兼容 compare_response_arms 的长度口径。

| metric | T4 candidate | D4v2 baseline | delta |
| --- | --- | --- | --- |
| bleu4_char | 0.1130609533 | 0.1122403823 | 0.0008205711 |
| rouge_l_char_f1 | 0.2605315545 | 0.2603151693 | 0.0002163853 |
| length_mean | 137.2782497876 | 134.3742565845 | 2.9039932031 |
| length_median | 93.0000000000 | 92.0000000000 | 1.0000000000 |
| length_min | 3 | 2 | 1 |
| length_max | 480 | 479 | 1 |
| empty_response | 0 | 0 | 0 |
| length_nonws_mean | 132.4944774851 | 129.7523364486 | 2.7421410365 |

格式诊断严格复用 `src/response/error_analysis.py` 默认值：near-empty ≤6、low-CJK <0.5、长回复 >200、重复句 min_length=6、重复 8-gram min_count=3、现有 markdown patterns；未引入阈值。

| diagnostic | T4 | D4v2 |
| --- | --- | --- |
| long_response_gt200 | 548 | 516 |
| repetition | 34 | 32 |
| markdown_artifact | 381 | 372 |
| near_empty | 36 | 40 |
| low_cjk | 3 | 1 |
| repeated_sentence | 8 | 6 |
| repeated_ngram | 27 | 29 |

## 7. D paired comparison

方向按逐条指标 delta 与 0 比较，沿用 compare_response_arms；只称 metric win / tie / loss。完整 2354 条：

| metric | metric win | metric tie | metric loss | mean delta | median delta |
| --- | --- | --- | --- | --- | --- |
| bleu4 | 1026 | 280 | 1048 | 0.0008205711 | 0.0000000000 |
| rouge_l | 1038 | 300 | 1016 | 0.0002163853 | 0.0000000000 |

## 8. D >200 long slice

切片按 gold `len(response_text.strip()) > 200` 冻结定义，support=528；不按 candidate 输出长度重新选样本。

| metric | T4 | D4v2 | delta |
| --- | --- | --- | --- |
| bleu4_char | 0.0796055450 | 0.0776773761 | 0.0019281689 |
| rouge_l_char_f1 | 0.1955777379 | 0.1950495746 | 0.0005281632 |
| length_mean | 299.0170454545 | 294.1268939394 | 4.8901515152 |

预注册采用线：long ROUGE delta ≥+0.01 且 overall ROUGE delta ≥−0.005。实际 long delta=+0.0005281632，未达；overall=+0.0002163853，未触发下降否决但不构成提升。偏短 pred/gold<0.7 从同卡 baseline 200/528 降至 candidate 192/528；历史报告的 210/528 属于旧设备 baseline，不与本次 200/528 混用。

长度均值增加约 4.89 字，未达到长桶主指标门槛；判断为“无明确改变”，不能确立内容质量改善，也不能将微小 metric 正值解释为人工质量收益。其余冻结长度桶在 JSON 中完整保留。

## 9. D 160-row screening recheck

| metric | T4 | D4v2 | delta |
| --- | --- | --- | --- |
| bleu4_char | 0.1028073574 | 0.0995542593 | 0.0032530981 |
| rouge_l_char_f1 | 0.2494494879 | 0.2508232507 | -0.0013737628 |
| length_mean | 130.1937500000 | 127.3187500000 | 2.8750000000 |
| length_median | 84.5000000000 | 81.0000000000 | 3.5000000000 |
| length_min | 4 | 4 | 0 |
| length_max | 446 | 442 | 4 |
| empty_response | 0 | 0 | 0 |

| metric | metric win | metric tie | metric loss | mean delta | median delta |
| --- | --- | --- | --- | --- | --- |
| bleu4 | 76 | 15 | 69 | 0.0032530981 | 0.0000000000 |
| rouge_l | 71 | 16 | 73 | -0.0013737628 | 0.0000000000 |

BLEU 小幅升、ROUGE 下降，与全量微小 ROUGE 正值方向不一致。160 条只作筛选复核，不冒充全量。D signal=`NO_CLEAR_SIGNAL`；response=`KEEP_D4V2`，T4 longweight 按预注册采用线 NO-GO。

## 10. D T5 condition matrix

独立重读 6 个真实 160 条臂（两个 off、四个 conditional），核对各 adapter tree、condition fields、input/condition hashes、逐条 journal 与 publication。both condition 文件逐条等于 C Day5 预测情感与 C v1 预测画像，未用 gold 分类替换条件。画像臂是 v1 实验，不据此宣称可靠画像采用门槛通过。

| adapter | condition | ROUGE delta vs own off | metric win/tie/loss | margin | eligible |
| --- | --- | --- | --- | --- | --- |
| D4v2 | emotion | 0.0015766971 | 69/34/57 | 12 | False |
| D4v2 | both | 0.0004611904 | 65/40/55 | 10 | False |
| T4 | emotion | -0.0024173370 | 51/42/67 | -16 | False |
| T4 | both | -0.0004316881 | 65/38/57 | 8 | False |

预注册触发线 `delta >= 0.005 AND margin >= 20`；四臂均未通过，与已有 t5_conditional_decision.json 一致。condition=`OFF`；不因小正 delta 开启条件。

## 11. Final fixed candidate

唯一固定组合：`KEEP_DAY5 + KEEP_D4V2 + OFF`，`profile_mode=empty`、max_new_tokens=256。C 候选缺证据、D longweight 未达采用门槛、条件矩阵无触发臂，均不足以替换冻结基线。画像 v1 仅内部研究候选，不扩为本轮多个待选组合。

已有离线组合 `results/cycle2_t5_fixed_20261030/baseline.jsonl`：2354 条，raw SHA `9add027ec278415890098e3a925be9a762b316fecac99ab789b8706343b3e6d0`；独立核对每条 Day5 emotion、D4v2 response、三维空画像完全匹配。此检查不等于 A T5 工程交付或真实加载的全面复审。

## 12. T6 readiness only

`T6 readiness=BLOCKED`；`T6 execution mode=BLOCKED`。已有完整预测，因此不写 GPU_REQUIRED，也不安排重复生成。不能在 adapter/model、tokenizer、chat template、generation defaults 与最终冻结资产尚未逐项证实时宣称 READY。

| gate item | result |
| --- | --- |
| condition_off | PASS |
| input_sha | PASS |
| sample_ids | PASS |
| generation_config_recorded | PASS |
| max_new_tokens_256 | PASS |
| output_record_count_2354 | PASS |
| duplicate_missing_extra_zero | PASS |
| journal_checksum_chain_and_output_equality | PASS |
| publication_lf_materialization | PASS |
| adapter_model_sha_matches_final_frozen_assets | NOT_INDEPENDENTLY_VERIFIABLE |
| tokenizer_matches_final_frozen_assets | NOT_INDEPENDENTLY_VERIFIABLE |
| chat_template_matches_final_frozen_assets | NOT_INDEPENDENTLY_VERIFIABLE |
| generation_defaults_matches_final_frozen_assets | NOT_INDEPENDENTLY_VERIFIABLE |

可复用候选路径 `results/response_gen/d4v2_val_newcard.jsonl`；raw SHA `a55274d3d3792dc1df28a2fb797a6b10b18e8688ce6c01eb1cfb24a94acf2b34`；上游 LF prediction SHA `1cfd430149c31d89a827dacb7e75e3e0eb6cab05a2abb8863f74871bd7520ce2`；generation fingerprint `7c452024a25a6b69ce79558b35604ff4e15e507722de097c1a84cedc302025b4`。对应完整 journal `results/response_gen/d4v2_val_newcard_progress/records.jsonl` 与 `results/response_gen/d4v2_val_newcard_progress/manifest.json`。

缺失项：当前没有模型/tokenizer 实体或与 journal model/adapter tree 一一对应的远端逐文件清单。训练报告的七个 base 文件可推得 tree `77f0e36abfa98943ef9619f6ea5a5615349514cd19e3ef829c045b40d000cd7f`，与生成记录 model-tree `4ebff76f…` 不同；可能涉及目录内容差异，但缺 inventory，不能推断已匹配。代码 tree 与最终 HEAD 也不同，不能只用当前脚本断言历史运行全部默认值相同。

解除方式：补充与现有 journal 绑定的冻结模型/tokenizer/chat-template/generation defaults 及逐文件哈希清单，CPU 核对；匹配后直接升级为 `T6_READY_REUSE_EXISTING_FULL_PREDICTIONS / GPU_NOT_NEEDED_REUSE`。不需要重新生成来填证据缺口。未生成 B-T6 正式文件。

## 13. CRLF / publication boundary

本机 CRLF 导致 raw publication validator 拒绝原 checkout；该失败没有被隐藏或改写成 raw PASS。B 在 `.codex_tmp/acceptance/cycle2/b_t5_20261101/lf_publication/` 临时物化两 full 臂 LF 字节，正式 `validate_publication` 均 PASS；report/output/marker SHA 与上游 LF 绑定一致。journal checksum chain 逐条验证通过；未改写任何上游跟踪文件。formal JSON 同时保留 raw/LF SHA。此确认排除换行造成的内容差异，不补足缺失模型身份。

## 14. Tests and limitations

`python -B -m unittest discover -s tests -v`：489 tests，FAIL 26、ERROR 13、SKIP 39（原始运行 exit=1）。另用同套 suite 的 TestResult 记录 PASS=421；FAIL 计数包含 subtest，不能简单用总数相减推算 PASS。结构化结果与日志在 `.codex_tmp/acceptance/cycle2/b_t5_20261101/`。

失败/错误全部位于未修改的 test_remote_scripts.py / test_workload_scripts.py：Windows Bash/Unix 路径、shell 返回错误、GBK/UTF-8 解码及 shell 未产出 pipeline 文件；不归因于本次报告新增。SKIP 的实际依赖/环境原因见 JSON。没有为了全绿修改 C/D 代码或测试。

`python -m compileall -q src scripts tests` PASS；`python scripts/ci_smoke.py` PASS；CPU independent recomputation PASS；两 full LF materialized publication PASS；`git diff --check` PASS。git status 仅新增两正式报告；Git 读取外部 ignore 有 Permission denied 警告，不影响已取得的分支和 SHA。

本地 validation 已被历史实验复用，非盲测，不是官方分数。JSON 只存路径、哈希、计数、汇总指标、切片与结论，不存对话、gold/candidate/baseline response。全部模型、数据、阈值未改；未 add/commit/push/merge。
