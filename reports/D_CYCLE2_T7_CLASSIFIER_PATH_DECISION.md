# D 岗位决策备忘：分类器部署路径统一口径（CYCLE2 T7 唯一 FAIL 项）

- **日期**：2026-10-31（T7 交付窗口 11/03–11/04 前）
- **触发项**：`reports/D_CYCLE2_T7_FREEZE_CHECK.md` §2.8——`configs/integrated.json:2` 的 `classifier_dir=weights/emotion_longtail075_b1_20261016` 本机不存在，按现配置冷启动抛 `FileNotFoundError`。
- **执行方式**：派 StepCode（v0.1.6 / step-5-preview）headless 只读核对 + **主会话对全部关键结论独立复跑核实**（三轮实测均本人执行，见 §4）。
- **结论**：**建议方案 B——把权重落到部署契约已写死的 `weights/emotion_longtail075_b1_20261016/`**。权重字节无误（`model.safetensors` SHA-256 `aedbacb1…`、目录 4 文件 409,584,858 B，与 C 冻结记载逐位一致），失调的只是本机解落目录名与契约目录名。

---

## 1. 问题定位：契约写死一个名字，本地解落了另一个名字

全仓引用清点（StepCode 执行，主会话抽验）：`emotion_longtail075_b1_20261016` / `c_day5_longtail075` 共 **18 个文件、34 行**命中。按是否影响部署归类：

| 类 | 位置 | 内容 |
|---|---|---|
| **部署契约** | `configs/integrated.json:2`、`configs/integrated_profile_v1_{off,emotion,both}.json:2` | `classifier_dir=weights/emotion_longtail075_b1_20261016`（4 份配置一致） |
| **部署契约** | `Dockerfile.integrated:23` | `COPY --from=model_assets /emotion_longtail075_b1_20261016/ weights/emotion_longtail075_b1_20261016/`（COPY 源与目标同用契约名） |
| **部署契约** | `scripts/stage_submission.py:26` | 暂存目录相对名硬编码 `weights/emotion_longtail075_b1_20261016` |
| **部署契约** | `tests/test_docker_model_paths.py:9-18` | 断言 Dockerfile COPY 目标集合 == `configs/integrated.json` 三个目录字段，且源名==目标名（去 `weights/` 前缀） |
| **部署契约** | `docs/CLOUD_QUICKSTART.md:9` | 镜像内交付布局文档：C 权重放 `weights/emotion_longtail075_b1_20261016/` |
| provenance/历史 | `reports/c_day5_emotion_longtail075_metrics.json:1115/13`（C 冻结 `checkpoint_path`/`output_checkpoint`）、`reports/b_t3_independent_review.json:10277`、`reports/c_cycle2_profile_v1_20261026.json:43`、`logs/C_DAY5_EMOTION_LONGTAIL075_20261016_training.txt:21-22`、`reports/C_DAY5_EMOTION_LONGTAIL_OPTIMIZATION.md:45/56/57`、`docs/EXECUTION_PLAN.md:113` | 均用契约名 |
| **分歧起点** | `reports/D_CYCLE2_T1_PREP.md:30` | D 于 T1 下载 C Release `c-day5-emotion-longtail075-20261016` zip（409,585,966 B，sha256 `67db5586…`）后**解落到 `weights/classification/c_day5_longtail075/`**，未按 `docs/CLOUD_QUICKSTART.md` 规定布局 |
| 本地实际 | `weights/classification/c_day5_longtail075/` | 唯一实存位置（`find weights -maxdepth 3 -name '*longtail075*'` 仅此一项） |

**即：部署契约五处（4 配置 + Dockerfile + stage_submission + 测试 + 文档）一致使用契约名，唯一失调的是 D 本机解落目录名。**

## 2. 影响面（不止本机冷启动）

1. **本机冷启动**：`scripts/run_integrated.py:63-75` `check_files`（sequence 分支）抛 `FileNotFoundError: .../weights/emotion_longtail075_b1_20261016/config.json`（§4 实测）。
2. **镜像构建**：`Dockerfile.integrated:23` 的 BuildKit 具名上下文 `model_assets=./weights`（见 `docs/CLOUD_QUICKSTART.md:78`）中，`./weights/emotion_longtail075_b1_20261016` **在本机缺失**——同布局的构建机 `docker build` 会在同一行失败；另两个目录（基座、adapter）均在位。**A 需确认其构建机布局。**
3. **A 的 PR #37 附带污染**：`results/cycle2_t5_fixed_20261030/catalog.json` 两处 `local_model_availability.classifier_dir` 声称 `{"path": "weights/emotion_longtail075_b1_20261016", "present": true}`，与本机事实不符（T5 候选来源核对附带发现，主会话已复算）。
4. **D 侧条件文件 provenance**：`results/response_gen/c5_conditions_val_full_report.json:3` 记录 `checkpoint: weights/classification/c_day5_longtail075`（本地解落名）——方案 B 落地后该记录仍然有效（指向真实文件），但需知悉存在两个合法路径名。

## 3. 两方案对比（均含实测）

| 维度 | 方案 A：改配置指向实际目录 | 方案 B：权重落到契约路径 |
|---|---|---|
| `check_files` 冷启动 | **PASS**（§4 实测） | **PASS**（§4 实测，原配置未动） |
| `tests/test_docker_model_paths.py` | **立即失败**（Dockerfile 目标 ≠ 配置目录，§4 复算） | 不受影响 |
| `Dockerfile.integrated` | 必须连带改 `:23`（源+目标）才能自洽（StepCode 全流程模拟：改后 COPY 目标==配置目录且上下文可解析） | 不改 |
| `scripts/stage_submission.py` | 不改（暂存时按硬编码名重写 config）——但与配置新值不一致，语义分叉 | 不改，天然一致 |
| `docs/CLOUD_QUICKSTART.md` | 应连带改（镜像内布局文档） | 不改 |
| A 未合并的 PR #37 provenance | **冲突**：T5 的 `*.config.json`/`catalog.json`/`scripts/assemble_candidate.py` 已按旧名写死，改契约引发 B 侧 provenance 复核返工 | 无冲突 |
| 改动面 | 4 配置 + Dockerfile + 文档（+ PR #37 已交付产物口径） | **零跟踪文件改动**；本地/构建机 additive 复制 409,584,858 B（`weights/*` 已 gitignore） |
| 执行方 | A（配置属部署入口，D 不代改） | D 本机可立即执行；A 构建机需同款布局 |

**建议方案 B**：契约五处已一致使用该名字，唯一失调的是本地解落名；方案 A 要反向重写契约并牵动未合并 PR 的 provenance，成本与风险都更高。

## 4. 实测证据（主会话亲手执行，WSL/python3，未加载模型）

```bash
# 现状（原配置）：FAIL
$ python3 scratch_main/check_files_probe.py
classifier_dir resolved to: .../weights/emotion_longtail075_b1_20261016
RESULT: FileNotFoundError -> .../weights/emotion_longtail075_b1_20261016/config.json

# 方案 A（scratch 配置仅改 classifier_dir=weights/classification/c_day5_longtail075）：PASS
RESULT: check_files PASS (plan A)
# 但 Docker 绑定断言复算：planA 期望集合 != Dockerfile COPY 目标集合 -> test_docker_model_paths.py 失败

# 方案 B（符号链接等价测试，验证后已清理复原）：原配置未动，PASS
$ ln -s classification/c_day5_longtail075 weights/emotion_longtail075_b1_20261016
$ python3 scratch_main/check_files_probe.py
RESULT: check_files PASS
$ sha256sum weights/emotion_longtail075_b1_20261016/model.safetensors
aedbacb14bced4417ac00d42c7cfccec722b64efaf84779fd9c8339f018e9e37
$ rm weights/emotion_longtail075_b1_20261016   # 复原，再次复跑恢复原 FAIL（现场无残留）
```

StepCode 侧补充证据（事件流已回收，与上表交叉一致）：现状 FAIL 探针输出同上；Plan A 全流程模拟（改配置 + 改 Dockerfile）后 COPY 目标==配置目录且上下文三目录全部 EXISTS；`gh release list` 显示 C Release `c-day5-emotion-longtail075-20261016`（2026-10-16）资产 `C_DAY5_EMOTION_LONGTAIL075_20261016.zip`（409,585,966 B，GitHub digest `67db5586…`）与边车、T1 登记值三方一致。

## 5. 执行清单

**方案 B（建议，D 本机已执行）**：
```bash
cp -r weights/classification/c_day5_longtail075 weights/emotion_longtail075_b1_20261016
sha256sum weights/emotion_longtail075_b1_20261016/model.safetensors   # 应得 aedbacb1…
python3 scripts/... # 复跑 check_files 冒烟（原配置）——已 PASS
```
- A 构建机：打包前确认 `weights/emotion_longtail075_b1_20261016/` 存在（或从 C Release zip 解落到该名），否则 `Dockerfile.integrated:23` 构建失败。
- C：知悉本地存在两个路径名均可指向同一权重；冻结记载（`checkpoint_path`）保持契约名不变。

**方案 A（备选，若 A/C 坚持统一到 `weights/classification/` 布局）**：改 4 份 config 的 `classifier_dir` → `weights/classification/c_day5_longtail075`；改 `Dockerfile.integrated:23` 的 COPY 源与目标；更新 `docs/CLOUD_QUICKSTART.md:9`；同步 PR #37 的 T5 provenance 口径并请 B 重审；复跑 `tests/test_docker_model_paths.py` 与 `check_files`。

## 6. 附带发现（T5 候选来源核对，主会话复算属实，建议 A 修文档）

A 的 PR #37 baseline 臂（2,354 条）经与 D 同卡源预测逐字节比对：`response_text` 0/2354 不一致（对 `results/response_gen/d4v2_val_newcard.jsonl`）、`emotion_label` 0/2354 不一致（对 `c5_conditions_val_full.jsonl`）、ID 顺序一致、`user_profile` 全空、`memory_refs` 全 `[]`；adapter 目录 digest `b3e9017e…` 与文件级 sha256 `6b75245a…` 均与 D4 v2 钉住值吻合。两处文档级瑕疵：
1. `catalog.json` 的 `local_model_availability.classifier_dir.present: true` 与事实不符（见 §2.3）；
2. `baseline.manifest.json` 的裸 `sha256`/`bytes` 字段是 CRLF 行尾口径（`lf_sha256` 才对工作区 LF 文件，如 response `lf_sha256 1cfd4301…` == 源文件实算），建议注明语义。

## 7. 复算与纪律声明

- 主会话复跑：现状 FAIL、方案 A PASS 及其测试破坏、方案 B PASS 与现场复原、Docker 绑定断言、T5 比对全部核心数字、catalog/manifest 两项瑕疵。
- StepCode 全程只读跟踪文件；其 25 分钟派发超时在上限处终止（备忘主体已落盘），证据自事件流回收。
- 本备忘及证据脚本：备忘入库 `reports/`；探针与事件流留本地 `.codex_tmp/acceptance/cycle2/d_20261031/`（不入 Git）。
