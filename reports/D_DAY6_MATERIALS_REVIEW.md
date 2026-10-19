# D 岗位 G6 材料复核报告（2026-10-19）

对应执行计划 10/19 D 行「仅复现确定性回复问题、材料复核 1–2h」。本报告为冻结后（10/18 22:00 代码及权重冻结之后）的 D 侧材料完整性复核与确定性缺陷排查，全部结论有本机执行证据；不新增任何实验。复核基线：main `6778aa5`（10/19 14:5x 拉取）；对照分支：A 的 Day6 集成分支 tip `d2a49df8`（PR #22）、本岗位于建 PR #21（D5 材料检查）与 #23（D6 权重核对）。

## 一、结论

**D 侧无阻塞性缺陷，无待修确定性问题。** 发现的材料缺口全部源于「#21/#23 尚未合并」这一合并进度问题，而非材料本身缺失或损坏；合并后即到位。需 A 决策的事项两项（见第四节）：专家材料清单补入 D6 报告、合并后终检 main 的 artifact 钉值。

## 二、D 侧材料清单盘点（对照 main）

| 类别 | main `6778aa5` 现状 | 缺口 | 缺口所在 |
|---|---|---|---|
| 报告链 `reports/D_*` | D2 8 个（4 报告+4 json）、D3 13 个（9 报告+4 json）、D4 7 个（3 报告+4 证据），共 28 个 D 侧文件 | D5 材料检查报告、D6 权重核对报告 | 分别在 PR #21、#23（均未合并） |
| 日志 `logs/D_*` | D2/D3/D4 各轮齐全（negate 放行清单见 `logs/README.md`） | D6 门禁日志 `D_DAY6_ARTIFACT_VERIFY.txt` | PR #23 |
| 门禁证据 `results/response_gen/` | Day3 三轮 6 文件在库 | `artifact_verify_d6/` 6 文件（冷启动门禁逐条预测+报告） | PR #23 |
| 配置 `configs/response_artifact.json` | **仍钉 D2 首轮**（`response_d2_1p5b`，adapter digest `af4f0e38…`，路径 `weights/response_sft_d2_1p5b`） | v2 钉值（`b3e9017e…`，路径 `weights/response_sft_d4_2ep`） | PR #22（A 分支已改对，D6 三项核对复算一致） |
| 权重 Release | `d-day4-response-sft-2ep-20261017` 已发布（10/17，v2 adapter+digest，D6 已复验 API 端点） | 无 | — |

## 三、最终镜像专家材料清单核对（A 分支 `scripts/package_expert_materials.py`）

镜像内 `/root/AI技术赛道-…` 材料由该脚本按 `MATERIALS` 显式清单打包（`--output artifacts/expert_materials_latest` 后作为 `expert_materials` 构建上下文喂给 `Dockerfile.integrated`）。核对其 tip `d2a49df8`：

- **完整性**：清单 26 个文件在 A 分支 tip 全部存在，构建期 `missing` 前置检查不会触发。
- **D 侧纳入 13 个文件**：`D_DAY3_RESPONSE_LORA_SFT.md`、`D_DAY3_SFT_FULL_1P5B.md`+json+两条日志、`D_DAY4_SFT_FULL_2EP.md`+json+两条日志、`D_DAY4_RESPONSE_ERROR_ANALYSIS.md`、`d_day4_2ep_comparison_summary.txt`、`configs/response_sft_2epoch.json`、`configs/response_artifact.json`。D3/D4 的训练-推理-结论链完整。
- **缺口（本次复核主要发现）**：清单**未包含 `reports/D_DAY6_ARTIFACT_VERIFICATION.md` 与 `reports/d_day6_artifact_verification.json`**（PR #23 在建），也未包含 `reports/D_DAY5_MATERIALS_CHECK.md`（PR #21 在建）。D6 报告是「镜像内 v2 权重未被换、冷启动门禁通过」这条证据链的终点（digest 三一致、Release 端点复验、3 次门禁 8/8×2），建议随 #21/#23 合并后补入清单；补入前打包的镜像材料将缺失交付权重的核对证据。
- **打包位置注意**：清单中 `logs/C_DAY5_EMOTION_LONGTAIL075_20261016_training.txt` 只存在于 A 分支（main 没有，随 PR #22 进 main）。打包必须在 A 分支 tip 或 #22 合并后的 main 上执行，在纯 main 上跑会 missing 失败。

## 四、确定性缺陷排查（本机执行，证据如下）

1. **测试套件**：`python -m unittest discover -s tests` → **172/172 PASS**（main 基线，0.138s）。
2. **v2 候选全量 val 预测**（`results/response_gen/d4_2ep_val_full.jsonl`，10/17 远程 4090 产物）：2,354 行、0 空回复、字段干净（恰为 `id`/`response_text`）、无重复 id、回复长度 min/median/max = 2/91/535 字（2 字短回复为正确匹配极短金标，D3 minlen 探针已定性）。
3. **官方提交链路不受影响**：`run_integrated.py`（A 分支）逐行透传输入官方 id（`test1_NNNNNN_tN` 格式）并以 `validate_prediction(prediction, row["id"])` 对齐；D 侧 val 诊断文件的键制差异（下条）不触及提交路径。
4. **可追溯性缺口（记录备查，非提交阻塞、非指标错误）**：D4 全量 val 预测文件使用旧 B1 v1 键 `pp:["conversation_id",turn,idx]`，而当前 `data/processed/val_eval_input.jsonl` 使用 `eval_NNNNNN_tN` 新键，两者无法直接 join。已验证两套键经 `(conversation_id, target_user_turn_id)` **一一对应（2,354/2,354 双射，抽样 `pp:["val_conv_000001",1,1] → eval_000001_t1`）**，可无损重键化；文件内指标由生成时同代码路径计算，报告 `d4_2ep_val_full.json` 记录了 `input_path` 但未注明键制差异。后续若要对 v2 臂复算本地 BLEU-4/ROUGE-L，先做该重键化即可。
5. **无待修确定性缺陷**：冻结纪律下不新增实验；D 侧未发现需要修复的确定性问题。

## 五、需 A 决策/执行的事项（跨组，D 不代为修改）

1. **专家材料清单补入**：#21/#23 合并后，请 A 将 `reports/D_DAY6_ARTIFACT_VERIFICATION.md`、`reports/d_day6_artifact_verification.json`（建议连同 `reports/D_DAY5_MATERIALS_CHECK.md`）加入 `scripts/package_expert_materials.py` 的 `MATERIALS`，并在 18:00 内部提交目标前重新打包。
2. **冻结门禁残留终检**：PR #22 合并后确认 main 的 `configs/response_artifact.json` 指向 v2（`b3e9017e…` / `weights/response_sft_d4_2ep`）。G5 冻结门禁的一部分，22:00 冻结时未完成，需在今天提交前闭环。

## 六、复现命令

```bash
# 环境：Python 3.12.3；本机 RTX 4060 8GB（本轮纯 CPU/文件核对，GPU 卡时 0）
python -m unittest discover -s tests                     # 172/172 PASS

# v2 预测与 eval 输入交叉核对（id 双射、空回复、字段）
python - <<'EOF'   # 结果：2354=2354、集合双射 True、empty 0、wrong fields 0
import json
inp = [json.loads(l) for l in open("data/processed/val_eval_input.jsonl", encoding="utf-8")]
pred = [json.loads(l) for l in open("results/response_gen/d4_2ep_val_full.jsonl", encoding="utf-8")]
new_keys = {(r["conversation_id"], r["target_user_turn_id"]) for r in inp}
old_keys = {(json.loads(r["id"][3:])[0], json.loads(r["id"][3:])[1]) for r in pred}
print(len(inp), len(pred), new_keys == old_keys)
EOF

# A 分支专家材料清单存在性（26 文件全在，D 侧 11 个）
git ls-tree -r --name-only <a-day6-tip>   # 对照 scripts/package_expert_materials.py MATERIALS
```
