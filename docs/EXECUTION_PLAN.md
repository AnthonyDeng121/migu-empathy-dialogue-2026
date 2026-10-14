# 2026年10月13日至20日执行计划（按训练关键路径重排）

修订日期：2026-10-13，所有时间为 Asia/Shanghai。10/12 已完成工作保留历史证据；从今天开始按资源、人工工时、机器耗时和依赖排程。详细训练任务与测速方法见 [TRAINING_WORKPLAN.md](TRAINING_WORKPLAN.md)。

## 比赛依据与边界

| 依据 | 比赛要求或当前证据 | 对排程的约束 |
|---|---|---|
| `../downloads_backup/AI技术赛道参赛细则.html`，三（一）数字人①–⑦ | 在组委会基础 Docker 镜像中集成模型、权重、推理、配置和入口；离线自动测评 | 基础镜像获取与兼容性今天开始核查，不能等训练完才准备；开发镜像不冒充正式镜像 |
| 同上，四（1）（2） | 上传须经过安全审核；最多5次成功测评；失败不占成功次数；最后成功结果覆盖历史 | 留上传/审核/失败修复窗口；只提交已验证候选，不要求用满5次 |
| `../official_materials/participant/README.md`、`run_inference.py` | 仅运行 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`；全量预测，前100条性能计时；固定两项产物 | 首次集成就适配真实入口及性能报告，区分加载/启动与推理；必须留全量评测时间 |
| `../official_materials/participant/templates/submission_schema.json` 与实际样例 | 回复、16类情感、三维画像、记忆引用；ID正则与样例冲突 | 输出ID原样对应输入，字段/枚举合法率100%；无记忆库不伪造引用 |
| `../official_materials/README.md` | 技术方案、实验设计/报告、训练日志、调优过程置于 `/root/AI技术赛道-数字人综合情感对话模型-队名` | 每轮训练同步记录材料，10/17检查完整性，最终镜像包含材料 |
| 赛事作品提交页 | 截止时间为2026-10-20 23:59 | 内部目标仍为10/19完成有效提交；卡时券在10/20约16:02提前失效，不能把最后GPU工作排到截止前 |
| 2026-10-13 队长账户与平台页面 | 两张RTX 4090券各20卡时、无卡模式10小时、30GB私有镜像与20GB文件存储免费额度；数字人专属镜像为`vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8` | 初版代码就绪后再开1卡训练；最终实例必须基于大赛专属镜像，普通PyTorch镜像只用于训练 |

排程依据上述本地官方附件、2026-10-12网页快照及队长2026-10-13登录平台后的页面核对。已确认允许开源模型和公开数据；正式评测硬件、总时限、专属镜像digest、镜像大小上限、评分权重、合成数据/外部API细则与隐藏memory bank仍待官方确认。不得把乒乓球赛题镜像套到数字人赛题。

## 资源门槛、容量与任务优先级

A已确认40个RTX 4090卡时、10小时无卡模式、存储额度、23:59截止、大赛专属镜像与平台提交路径，详见`reports/A_DAY2_PLATFORM_INVESTIGATION.md`。人员可完成所分配任务，排程不再按固定4–6人工小时裁剪；机器墙钟时间和代码依赖仍必须记录。团队决定在B1合并且C/D初版可直接运行前不开GPU实例。

A与C/D在首次下载前做容量表：基础镜像、两路模型/tokenizer、训练数据、依赖缓存、adapter/优化器/checkpoint、最终镜像和回退镜像各占多少。通知所列30GB镜像和20GB文件空间按可能上限预算，保留20%工作余量并核实平台统计口径；保留最佳及恢复所需checkpoint，不存每epoch副本。容量不足先缩小候选/清理可重建缓存，经核实再安排外部存储；不删除唯一可恢复权重或最好镜像。

训练/下载/推理可无人值守，不把机器小时都当成人工小时。优先级：冻结数据v1→C/D可运行初版→1卡短跑→首轮训练→全量评测→正式镜像/提交→一轮优化。Git/CI和表格维护完成后不继续占用整天；共享架构、额外模型和大规模超参搜索可裁剪。

10/13不开GPU是节约卡时的主动决策，不是资源缺失。若10/14 C/D初版仍未满足开机门槛，先完成代码和数据契约，不开机现场开发；一旦门槛满足，由A创建1卡训练实例并以50个优化step短跑重算ETA。保底链路继续维护。

## 里程碑与失败处理

| 门槛 | 时间 | 可核验结果 | 未达成时 |
|---|---|---|---|
| G0 资源与模型选型 | 10/13资源调查完成；10/14完成代码门槛 | A资源/卡时/镜像/截止结论；B数据v1；C/D各一个固定revision主候选及可运行初版 | 先做CPU/MPS准备；未满足开机门槛不消耗卡时，不扩大候选列表 |
| G1 首轮训练启动 | 10/14初版PR及验收后 | 1卡短跑有step增长、有限loss、checkpoint重载、显存/吞吐/ETA；随后才启动首轮 | 当日修复；ETA超预算就缩长度/轮次/模型，不能跳过冒烟 |
| G2 首轮候选 | 10/14 18:00权重，22:00候选/评测进度结论 | 分类与回复权重、哈希、配置、日志；B同集指标或真实评测ETA | 保留可用任务权重；取消次要消融，把10/15上午留给补训/评测 |
| G3 可提交baseline | 10/15 18:00候选镜像；10/16 18:00首次成功目标 | 官方基础镜像与入口、离线样例/100条、全量不崩、完整计时/材料、安全审核与提交记录 | 按反馈修兼容性；若基础镜像/平台仍阻塞，A每日跟进并保留本地候选，不能宣称已提交 |
| G4 优化停止与选择 | 10/17 12:00停止新增长训练，18:00确定候选 | 一轮优化的全量同集对比、合法率、延迟与资源；A+B选版本 | 无明确提升则保留最好已验证镜像；不为了更新版本而提交 |
| G5 代码与模型冻结 | 10/18 18:00候选全量回归；22:00代码和权重冻结 | 权重/配置/依赖/commit固定；两次冷启动断网、预测哈希一致、材料齐全 | 回退已验证候选，只修确定性缺陷并双人复核 |
| G6 镜像与有效提交 | 10/19 12:00镜像冻结并上传；18:00前内部有效提交目标 | digest、安全审核、官方成功状态、结果和截图；A+B双签 | 当天用预留窗口修上传/审核/运行故障；必要时顺延但不越过官方真实截止 |
| G7 截止缓冲 | 10/20官方截止前 | 复查有效提交、队伍信息、成绩、凭据；必要时重提交同一已验证镜像 | 不开展新研究/训练/依赖升级；不默认还有完整一天 |

T0为10/13团队实际开始时间。若开始较晚，按TRAINING_WORKPLAN的顺延条件记录变化；10/12不追溯虚构训练已完成。每轮训练ETA由短跑修正，以上是团队目标，不是主办方承诺的服务时间。

## 每日四人排班

括号中的h是主动人工预算；机器耗时和单卡顺序见TRAINING_WORKPLAN。各任务间可以重叠，但不能把单GPU训练和全量生成评测同时排成独占。

| 日期 | A：资源、集成、提交 | B：数据、评测 | C：分类训练 | D：回复训练 | 当天交付 |
|---|---|---|---|---|---|
| 10/13 | 完成卡时、存储、截止、许可、训练镜像、数字人专属镜像与提交路径调查；同步main并整理A验收PR | 提交B1全量转换PR #7 | MacBERT情感脚本、固定revision与MPS冒烟已合并PR #6 | 检索保底全量验证已合并PR #4；生成训练初版 PR #8 已提交并回应 A 五条审查修复项（修复后冒烟重跑中） | A调查报告；B/C/D实际进度入表；按团队决定不开GPU |
| 10/14 | 维护开机门槛清单；定义C分类+D回复+A组装接口和vLLM专属镜像依赖策略；复核D生成PR；门槛满足后才开1卡短跑；更新PR #5 | C 已验收 PR #7 并合并；从 main 复现 B1 全量转换和哈希；复核 A PR #5 平台事实与计划并提交 review | 从最新 main 适配 B1 classification JSONL；补三维画像 head；拆分训练/最终推理依赖，禁止覆盖专属镜像 PyTorch；完成 MPS dry-run 并提交 PR #9，由 D 验收 | 固定一个可被 vLLM 0.19 加载的开源生成模型及 revision/许可证/体积；完成 LoRA/SFT 脚本和 CPU dry-run；PR #8 经 A 复验收已合并 main（merge commit `1dff405`）；适配 B1 response JSONL 后在远程 4090 完成 D2 冻结集首轮（1.5B 单 epoch，4,219.7s，adapter `62a695ea…`，期间克服一次平台抢占回收） | B PR #7 已合并且 C 全量校验通过；C PR #9 已经 D 独立复证验收并合并 main（merge commit `0b2b189`）；D PR #8 已合并；D2 首轮候选已产出（PR #10 含适配与证据，待 A 审）；A PR #5 含平台调查与明确开机门槛；远程实例发生过一次抢占式 GPU 回收（已恢复） |
| 10/15 | baseline正式镜像构建、断网/计时/材料检查3–4h；上传/审核准备1h | 12:00前首轮全量结论；容器全量回归与指标独立复核3–4h | 修首轮主要问题；只启动一个有假设的候选优化2–3h；支持导出/review1h | 根据首轮结论只启动一个必要优化2–3h；支持镜像/加载1–2h | 可提交baseline；首轮选择依据；下一轮训练明确预算，不能阻塞baseline发布 |
| 10/16 | 首次官方提交/运行跟踪2–3h；只修兼容问题1–2h | 官方与本地差异、全量对照2–3h；评测优化候选1–2h | 完成优化候选并复核分类错误2–3h；必要兼容支持1h | 完成优化候选并复核回复/格式2–3h；必要兼容支持1h | 最迟当日18:00首次成功目标；候选v2或保留baseline的明确结论 |
| 10/17 | 候选资源/镜像对比2–3h；材料和提交完整性检查1–2h | 12:00前收齐新增长训练产物，全量排名/消融3–4h；与A选择1h | 不新增长训练；补必要校准/导出2–3h；材料1h | 不新增长训练；必要解码/长度检查2–3h；材料1h | 18:00选定最终候选；完整实验表、训练日志和材料草稿 |
| 10/18 | 冷启动断网/全量回归3–4h；候选发布与冻结1h | 独立复跑与泄漏复核3–4h；签署候选结论1h | 权重/标签/依赖核对1–2h；故障与review支持1–2h | 权重/tokenizer/adapter核对1–2h；故障与review支持1–2h | 两次预测哈希一致，性能报告完整；22:00代码及模型冻结 |
| 10/19 | 12:00前镜像digest冻结/上传；跟踪审核和成功状态2–4h，保留故障缓冲 | 核对镜像/指标/资料和有效提交2–3h，双签1h | 仅复现确定性分类问题、材料复核1–2h | 仅复现确定性回复问题、材料复核1–2h | 18:00前有效提交内部目标，保留最好镜像与可重提交包 |
| 10/20 | 截止前复查有效状态、队伍信息、截图/编号；只修提交故障 | 复核最后有效成绩和提交内容 | 只支持确定性故障复现 | 只支持确定性故障复现 | 官方截止前提交有效；不新增模型实验 |

## 成功测评机会策略

1. 第1次成功：10/15–16的完整baseline兼容性探针；取得官方基础镜像并满足本地门槛后才提交。
2. 第2次成功：10/17–18，只有全量同环境验证明确提升且性能未退步的候选；不提升就不使用。
3. 第3次成功：10/19，仅在必要修复或已验证提升时使用；已有最好成功结果可直接保留，不必须重提。
4. 第4/5次保留给截止前有证据的故障修复/重提交。每次记录成功次数余额、镜像digest、时间、审核状态和结果；运行失败按规则不占成功机会。

最后一次成功覆盖此前成绩；不得提交本地更差版本“试试看”。上传时间按镜像字节/实测吞吐计算并加至少50%余量；安全审核与官方排队时间未知，10/15开始实际流程，不能把10/20上午首次上传当可靠计划。

## 审查、状态与历史

分支从最新main同步，禁止直接推main或force push。固定review闭环A→B→C→D→A；相关公共接口追加模块负责人复核；最终发布A+B双签。训练与review并行，正式镜像必须包含已验收产物。

Ready=实际依赖满足才可开始；Blocked=写明阻塞人、缺什么、预计解除时间、降级方案；Integration Gate=有完整交付仍待review/合并；Done=合并main且测试与验收证据齐全。训练依赖GPU资源，不把未确认资源的任务统一标Ready。

### 10/12历史记录（保留原事实）

| 日期 | 负责人 | 计划交付 | 执行状态 | 证据或PR | 固定验收人 | 验收状态 | 验收结论或待修改问题 | 下一步 |
|---|---|---|---|---|---|---|---|---|
| 10/12 | A | 官方资料归档、入口契约、项目骨架 | Integration Gate | `docs/OFFICIAL_REQUIREMENTS.md`、`reports/VERIFICATION.md`、`reports/day1_docker_val100_metrics.json` | B | 待验收 | 样例 3/3、验证 100/100 断网通过；待 B 拉取复核 | B 复现容器证据并提交 review |
| 10/12 | B | 全量数据审计、Schema、指标框架 | Done | `docs/DATA_AUDIT.md`、`reports/data_audit.json` | C | Approved | 历史基线已完成；后续改动按新闭环走 PR | 建立 conversation 级训练清单 |
| 10/12 | C | 标签映射和多数类分类 baseline | Done | PR #1、merge commit `e7b251e`、`reports/CLASSIFICATION_BASELINE.md` | D | Approved | 已完成本地验收并合并 `main`；多数类分类 baseline 和公共 Schema 改动通过 | 开始情感分类 encoder baseline |
| 10/12 | D | 检索回复 baseline 与回复接口 | Integration Gate | PR #4、分支 `feature/d-retrieval-baseline`、全量单测 18/18（`tests/test_retrieval.py` 锁定检索契约）、`reports/baseline_val_full_metrics.json`、`reports/RETRIEVAL_BASELINE.md`、`results/baseline/`（全量 2354 校验通过） | A | 待验收 | PR #4 待 A 审查；分数仅为保底；A 需复现运行与输出 Schema | A 审查通过后合并；建立生成 baseline |

### 10/13当前状态（只记录已知事实）

| 工作 | 负责人 | 状态 | 证据/实际依赖 | 下一步/验收人 |
|---|---|---|---|---|
| 协作CI与实验流程 | A | Integration Gate | PR #5、commit 3af555a；远程checks已通过 | B复核；检索实验脚本不是神经网络训练脚本 |
| GPU/账户额度/存储/截止/专属镜像/提交路径调查 | A | Integration Gate | `reports/A_DAY2_PLATFORM_INVESTIGATION.md`；40个RTX 4090卡时、10小时无卡、30GB镜像/20GB文件额度、23:59截止、vLLM 0.19专属镜像 | B按报告命令复核并review PR #5；初版就绪前不开GPU |
| 数据v1及分类/生成样本 | B | Done | PR #7 已合并，merge commit `63801edf32dfa837c8041288a21bf9cc55faafd9`；C 从 main 全量校验：train 18,653、val 2,354 分类样本，B1 train/val conversation overlap 0，文件 SHA-256 与产物报告一致 | B 继续按计划复现正式产物 |
| 分类选型与CPU准备 | C | Integration Gate | `reports/C_DAY2_CLASSIFIER_PREP.md`、`reports/C_DAY2_EMOTION_SMOKE.md`、`reports/c_day2_emotion_smoke.json`；Mac M5 MPS 冒烟 1,600/18,657 点、50 steps、全量 val 2,354 点；未使用 B1 v1 | D待验收；B1 数据 v1 到位后重跑，GPU训练仍待 A 核实资源 |
| 回复选型与CPU准备 | D | Done | 检索baseline与官方训练数据；候选选型（Qwen2.5-1.5B/0.5B-Instruct）、LoRA SFT 训练模板与本机 4060 冒烟已交付：`src/response/sft_data.py`、`scripts/train_response_sft.py`、`scripts/run_response_inference.py`、`configs/response_sft.json`、`configs/response_sft_0p5b.json`、`reports/D_DAY2_RESPONSE_SFT_PREP.md`；修正后全量 1 epoch ETA：1.5B 28,243s≈7.85h、0.5B 17,971s≈4.99h（裸估，+30% 余量另计）；66/66 单测；本机 CUDA 不计平台卡时；PR #8 已由 A 复验收并合并 main（2026-10-13 14:08:45Z，merge commit `1dff405`） | 适配 B1 response JSONL 后按 D2 跑冻结集首轮；等 A 开机门槛开 1 卡 ≥50 step 短跑复测 ETA，再定 1.5B/0.5B 与 epoch/长度 |
| 分类/生成GPU训练 | C/D | Integration Gate | B1 已合并；C PR #9 已由 D 验收并合并 main（merge commit `0b2b189`）；D 首轮 LoRA/SFT 已产出候选（D2 Day 3 行：1.5B 单 epoch、4,219.7s、adapter `62a695ea…`，全量 val 诊断 BLEU-4 0.1075/ROUGE-L 0.2542、0 空输出）；期间远程实例 GPU 被抢占回收一次，用户重启后恢复；C 的 C2 训练仍待平台资源 | D 补原模型同集臂完成三路对比；C 按 A 开机门槛启动 C2；B2 同环境评测两路候选 |

未来日期不预先填写Ready或Done；每个任务按TRAINING_WORKPLAN ID记录开始/结束、实际人工/GPU小时、ETA、报告/权重哈希、review和下一步。成员已有工作须先核验再更新，不因计划重排抹掉完成证据。

每天使用下表追加实际记录，机器等待也记录起止，不以排班目标时间代替实际完成时间：

| 任务ID/负责人 | 状态 | 实际开始/结束 | 人工h/GPU卡时 | ETA/依赖/阻塞人 | 配置/commit/报告/权重哈希 | 验收人/review | 下一步 |
|---|---|---|---|---|---|---|---|
| C1 / C | Integration Gate | 起始墙钟时间未记录；结束 2026-10-13 11:29:59 +08；脚本计时 215.266s | 人工时未单独计；平台 GPU 卡时 0；本机 MPS | 平台资源已确认；等待 B1 PR #7 合并、B1输入适配和画像head；正式 GPU 训练仍 Blocked | seed `20261012`；batch 32；max length 256；50/50 steps；脚本 SHA-256 `ba3db44cbc5a9ab755881eb90390d0805bc786a15ab45e1c486b30d3564ebd84`；smoke checkpoint SHA-256 `d0d604247c05657942dec865a3a5eadacebc4b1a2ea8d5271dfb66ff9ddadfc3`；`reports/C_DAY2_CLASSIFIER_PREP.md` | D待验收 | 10/14验收B PR #7，适配B1并补画像后提交PR；通过A开机门槛再训练 |
| B1 / B | Done | 2026-10-14；起止时间未记录 | 人工时未单独计；平台 GPU 卡时 0 | PR #7 已合并；main 全量复现/字段、哈希及 train-val conversation 隔离检查通过；分类 train 18,653、val 2,354 行 | merge commit `63801edf32dfa837c8041288a21bf9cc55faafd9`；B1 `classification_train.jsonl` SHA-256 `4d4e949813acca37ea5e827f4455f60e941f96c078357b8dc560a70b23a07cd5`；val SHA-256 `30159857e5e65614fcde6f236f1c1b1a51ab6b1f6d8b7ffcd4fbf807f5f07201` | C 已验收 | B 继续按计划从 main 复现正式产物 |
| C1 Day 3 / C | Integration Gate | 2026-10-14；完整开始/结束墙钟时间未记录；训练 8.794s | 人工时未单独计；平台 GPU 卡时 0；Apple Silicon MPS | 对 B1 全量 18,653/2,354 行完成结构、标签、上下文、source split 与 conversation 隔离校验；实际训练抽样 96/32，3 optimizer steps，image compatibility 未验证 | seed `20261014`；batch 16；max length 256；MacBERT fixed revision；head SHA-256 `c3e71877be4ff216ec9573a9f443c1cb39ca49aa141448ad51e90677d34c5c45`；checkpoint tree SHA-256 `9ad18bd5720a69c65837d63f16fe0b8415df4672aa72da81edd96cbbbfefef2f`；`reports/C_DAY3_EMOTION_PROFILE.md`、`reports/c_day3_emotion_profile.json`；PR #9 | D 已验收（smallrespon，2026-10-14；独立复证：97/97 单测、B1 重生哈希三方一致、跨环境复跑训练链与预测链路）并合并 main（merge commit `0b2b189`）；C 可自行调整本行措辞 | D review 后按 A 开机门槛启动平台 50-step 短跑；不将本机冒烟当作候选训练完成 |
| A1 / A | Integration Gate | 2026-10-13完成 | 平台GPU卡时0 | 等待B复核PR #5；最终镜像实测依赖C/D初版 | `reports/A_DAY2_PLATFORM_INVESTIGATION.md`、`docs/OFFICIAL_REQUIREMENTS.md`、`docs/TRAINING_WORKPLAN.md` | B待验收 | B按报告命令复现并review；10/14执行开机门槛与接口任务 |
| D1 / D | Integration Gate | 模型落盘 14:18–14:19 +08；冒烟/对照日志 19:04–19:22 +08；修复后重跑 20:26–20:57 +08（CPU 回归 20:31、1.5B 20:45、0.5B 20:57 完成） | 人工时未单独计；平台 GPU 卡时 0；本机 RTX 4060 8GB | 修正后全量 1 epoch：1.5B 28,243s（12.1061 秒/step，+30% 余量 36,717s）、0.5B 17,971s（7.7031 秒/step，余量 23,363s）；此前 46/36 分钟为计时 bug 产物已撤回；依赖 B1 v1 合并与 A 开机门槛 | seed `20261013`；LoRA r8 α16；batch 1×累积 8；max_length 2048；18,657 样本；53 步冒烟（3 warm-up + 50 计时步）；1.5B adapter SHA-256 `10b595d0330b304f6ef1f1c4985cad37176532bf8c9605a57b78528d7969bf85`、0.5B `d8e65e72148ea090fa27873621887b4ec38b4bf284f3f7b7f8a120529322381a`；PR #8（A 五条修复项已回应）；`reports/D_DAY2_RESPONSE_SFT_PREP.md`、`reports/d_day2_response_comparison_val100.json`、`logs/D_DAY2_SFT_SMOKE_*.txt` | A复验收 | 适配 B1 response JSONL 后按 D2 跑冻结集首轮；条件注入对比经 `--condition-jsonl` 接入 |
| 待执行时填写 | 按真实依赖填写 | 待实测 | 待实测 | 待核实 | 有交付才填写 | 待验收 | 当日更新 |
| D2 Day 3 / D | Integration Gate（候选已产出，待 B2 评测与 A review） | 2026-10-14：代码适配 11:50–12:40；平台 GPU 阻塞 11:50–13:10（抢占回收，见下）；用户重启实例 13:10 后 GPU 恢复；冒烟 13:12–13:20；全量训练 13:23–14:37（4,219.7s）；val 全量推理 14:17 起（进行中，ETA 约 15:50） | 人工时未单独计；平台 GPU 卡时：训练 1.17 + 冒烟 0.2 + 推理约 1.6（进行中），按平台账单记录 | 阻塞已解除：远程实例调度方式为抢占，GPU 被平台回收（11:50–13:10 EPERM），用户重启实例后恢复；代码与数据侧无阻塞 | commit `6d187bc`（B1 适配）+ `6761204`（阻塞记录/runbook）；冒烟 1.8153 秒/step、外推 4,233s（+30% 5,503s）；全量 2,332 步、1.8094 秒/step、mean loss 2.969→2.159、18,653/18,653 样本、超长跳过 0、显存 6.72GiB、first loss 有限、梯度范数 0.880、392 张量变化、离线重载 392/392+3/3 复现；adapter 36,981,072 字节 SHA-256 `62a695ea594a1546e217ee5edd691131051a4e64e3ecfcec91af75de7a133fcf`（落盘独立核验一致）；`reports/D_DAY3_RESPONSE_LORA_SFT.md`、`reports/d_day3_sft_full_1p5b.json`、`reports/D_DAY3_SFT_FULL_1P5B.md`、`logs/D_DAY3_SFT_FULL_1P5B.txt`；staging 在 `/mnt/storage/migu-d2/`；报告口径修正（卡时末行）有单测锁定，117/117 通过 | A 审 PR #10；B2 同集三路评测 | val 全量推理（无条件臂）已完成：2,354 条、8,844.5s、3.757 秒/条、22.4 tok/s、0 空输出；本地字符级诊断 BLEU-4 0.1075 / ROUGE-L 0.2542（同代码路径重算检索 baseline 0.0294 / 0.1439，约 3.7×/1.8×）；原模型同集臂推理中（补三路对比）；C2 权重就绪后用 `--condition-jsonl` 预测条件臂复跑；22:00 前按 B2 指标定采用 |
