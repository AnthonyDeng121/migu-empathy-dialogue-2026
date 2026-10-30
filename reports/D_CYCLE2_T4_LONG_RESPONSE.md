# D 岗位 CYCLE2 T4：长回复改进（已完成，NO-GO）

日期：2026-10-28 至 10-29（Asia/Shanghai）。分支：`feature/d-day7-deadline-standby`。状态：**已完成：T4 训练 + 新卡无人值守评测，判定 NO-GO（D4 v2 维持交付候选）；实现与管线随 PR #39 待 A 审**。

对应执行计划 T4（D 列）：「根据自己已完成的误例分析，选**一个**长回复改进变量（样本权重或训练步数）；短实验信号成立再完成候选，保留必要训练，不盲增生成长度」。

## 一、失败模式定位（全量 val 2,354 条，D4 v2）

数据源：`reports/d_day4_response_error_analysis.json`（D4 v2 全量 val 预测，2354 条）。

### 1.1 长度匹配是长回复的主导失败模式

| gold 桶 | n | pred/gold 中位数 | pred/gold 均值 | 明显偏短（<0.7） |
|---|---:|---:|---:|---:|
| 51-100 | 822 | 0.913 | 1.000 | 189 |
| 101-200 | 626 | 0.897 | 0.998 | 182 |
| **>200** | **528** | **0.803** | **0.837** | **210（39.8%）** |

>200 字桶的 pred/gold 分位：p10=0.38 / p25=0.59 / p50=0.80 / p75=1.04 / p90=1.37。

**解读**：中短回复上模型长度校准良好（均值 1.00），但 gold >200 字时系统性收缩——四成长回复被压缩到不足七成。这是可训练的偏差，不是解码上限造成的（pred_len 最大 470 字符，远低于 256 token 上限）。

### 1.2 不是「越长越好」，而是「长度匹配」

>200 字桶内按 pred_len 四分位：

| 四分位 | n | pred_len 中位 | gold_len 中位 | mean ROUGE-L |
|---|---:|---:|---:|---:|
| Q1（最短） | 132 | 154 | 252 | 0.1782 |
| Q2 | 132 | 247 | 282 | **0.2003** |
| Q3 | 132 | 358 | 386 | 0.1956 |
| Q4（最长） | 132 | 432 | 474 | 0.1877 |

倒 U 形：过短与过长都差，匹配 gold 长度的 Q2 最好。**因此 T4 目标是把偏短的 40% 拉回匹配区间，不是整体加长。**

### 1.3 训练数据侧证据

`data/processed/b1_full_v1/response_train.jsonl`（18,653 条）：

| 桶 | 条数 | 占比 | 字符占比 |
|---|---:|---:|---:|
| >200 | 5,853 | 31.4% | **70.4%** |
| 101-200 | 5,383 | 28.9% | 19.1% |
| 51-100 | 4,631 | 24.8% | 8.5% |

长样本占条数 31.4%、占字符 70.4%。当前训练用 transformers 内置 loss（**按 token 计的平均交叉熵**），长样本本就已按其 token 数占约 70.4% 的梯度份额，并非"按条数计权被稀释"——此处按先写后纠的原则记录更正。样本加权真正改变的是：>200 桶每个 token 的权重翻倍，使其梯度份额从约 70% 升到约 82%，且 batch_size=1 下长样本单步梯度精确翻倍（见 2.1 实现节）。假设因此更偏向"把长回复写长、写匹配"，但不保证——长度匹配才是目标（1.2 节倒 U），最终以 2.2 判定线为准。

### 1.4 格式类问题（不属本轮变量）

markdown 粗体 342、列表符 169、近空 40、标题 33、引用 29、重复 n-gram 28、重复句 9。按计划原文，格式清洗后处理与长度惩罚/截断**不在本期批准范围**，本轮不动。

## 二、选定变量与设计

**选定变量：长回复样本加权（long-sample upweighting），单变量。**

其余全部固定为 D4 v2 口径：同基座、同数据、同 2 epoch、同 batch/累积/lr/warmup/seed、同 LoRA 超参、condition_mode=gold。

### 2.1 权重规则（先定后跑，不看结果调参）

按 gold 回复字符数分桶加权，`sample_weight` 乘在每条样本的 loss 上：

| gold 字符数 | 桶 | 权重 |
|---|---|---:|
| ≤100 | 短 | 1.0 |
| 101–200 | 中 | 1.0 |
| >200 | 长 | **2.0** |

只提升最长桶的权重、不动其余，使 >200 桶的期望梯度份额从约 70.4% 升到约 82.7%（2×70.4/(2×70.4+29.6)），同时保留其余桶全部样本与权重不变。这是单变量：权重规则是唯一改动。

**实现（2026-10-28 编码后补充，与上文规则一致）**：`scripts/build_long_response_weights.py` 生成加权训练文件（每条样本加 `sample_weight` 字段，行序与原文件一致，已实跑：18,653 条中 5,853 条权重 2.0，输入 sha256 与计划记录一致）；`scripts/train_response_sft.py` 的 `_weighted_token_loss` 把每条样本的 token 损失和乘以自身权重、再除以**未加权**的监督 token 总数——全 1.0 时与 transformers 内置 token-mean 逐位一致（单测实证），且 batch_size=1 下 2.0 权重使该样本梯度精确翻倍（加权均值归一化会把 w 在分子分母间约掉，静默失效，单测防范）。`src/response/sft_data.py` 负责把行级 `sample_weight` 透传到样本并校验为正有限数。测试证据：WSL 全量 382 项通过（HEAD 369 + 新增 13），含 tiny-model e2e「全 1.0 与不加权重逐位一致」「2.0 权重必须改变训练轨迹」。

**先验依据**（不看结果）：>200 桶占训练 token 70.4%；把它的每 token 权重提到 2.0 后，其期望梯度份额从 70.4/(70.4+29.6)≈70% 升到约 82.7%，与「模型欠拟合长回复」的观察同向。

**否决的备选变量**（记录备选而非偷偷换）：训练步数加倍（4 epoch）。理由：D2→D4v2 的 1→2 epoch 已拿到主要收益，4 epoch 会在同等算力下把短样本也一起过拟合，无法把收益归因到长回复；且算力预算翻倍。

### 2.2 判定标准（先定后跑）

- 主指标：160 条筛选集（`t2_selection_input.jsonl`）**同设备**（同一张 4090）顺序生成的字符级 ROUGE-L 与 BLEU-4。
- 必看切片：**>200 字桶 ROUGE-L**（当前 D4 v2 0.1958，全量 val 口径）与**偏短率（pred/gold<0.7）**。
- 通过线：>200 桶 ROUGE-L 相对 D4 v2 同设备基线**提升 ≥0.01**，且全集 ROUGE-L 不下降超过 0.005。
- 未达线即 no-go，记录负结果，不盲搜第二组权重。

### 2.3 短探针（先于全量）

按计划「先短跑测时再完成必要训练」：先跑 `--max-steps` 短探针（如 120 步）测步速与 loss 曲线形态，确认加权生效（长桶 loss 占比变化）后再启动全量 2 epoch。

## 三、执行环境（2026-10-28 历史记录，实例已销毁）

> 本节为 bitahub 实例（`xj-member.bitahub.com:42168`）的历史档案：训练在该实例完成，实例已于 19:59 由用户销毁，
> T4 adapter 已拉回本机 `weights/t4_longweight_2ep/`（sha256 `bc6ca023…`，36,981,072 B）。网络结论对新实例仍有效。

新实例：`xj-member.bitahub.com:42168`（root），主机名 `bitahub-a21667084185235456238240`。

| 项 | 值 |
|---|---|
| GPU | 单卡 RTX 4090 24G（独占） |
| CPU / 内存 | 13 核 / 503 GB |
| 磁盘 | **30 GB overlay**（紧张：基座 3.09G + 数据 0.2G + checkpoint，需按步清理） |
| 系统 | Ubuntu 24.04.1 LTS |
| Python | `/opt/conda/bin/python` = 3.12.13 |
| torch | 2.11.0+cu128（CUDA 12.8，`torch.cuda.is_available()=True`） |
| transformers / peft / accelerate | 4.57.6 / 0.21.2 / 1.15.0（与 T3 同版本） |

### 3.1 网络结论（实测）

- GitHub Release 直连：**449 KB/s 实测可用**（38,194,371 B / 85s，sha256 已核验一致）。
- HuggingFace 直连：**不通**（30MB range 请求超时）；hf-mirror.com：**慢但可用**（14:55–15:41 后台下载完 3,087,467,144 B 完整基座，约 46 分钟，含一次网关断流后重试）。结论更正：基座最终经 hf-mirror 在线拉取成功，无需 scp 上传。
- 内网 10.10.2.104:22：**不可达**（T3 实例已不可用）。
- SSH 长连接会被中间网关（198.18.0.27）断流，**长任务必须 nohup/setsid 后台化再轮询**，不能挂在 ssh 前台。

### 3.2 文件就位情况（2026-10-28 15:45 核验）

| 文件 | 状态 | 校验 |
|---|---|---|
| 基座 `/root/migu-work/qwen15b.safetensors` | ✅ 远端已下载完成（3,087,467,144 B） | 远端 sha256 待与本机 `dd924a11…` 核对 |
| `data/processed/t4_long_response/response_train.jsonl` | ✅ 本机已生成（加权训练文件，T4 唯一变量数据） | sha256 `8ef2a5dcfe7cd6a81f6eca8ba29808fec50abd635666d199a4fa5759ff027049` |
| `data/processed/b1_full_v1/classification_train.jsonl` | ⬜ 待传（condition_mode=gold 的条件文件，174 MB） | sha256 以 `data/processed/b1_full_v1/SHA256SUMS` 为准 |
| `data/processed/b1_full_v1/response_val.jsonl` | ⬜ 待传（18 MB） | sha256 `ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2` |
| `results/response_gen/t2_selection_input.jsonl` | ⬜ 待传（160 条筛选集） | — |
| `weights/response_sft_d4_2ep/`（对照臂 adapter） | ⬜ 待传（37 MB，同设备重跑对照臂） | — |
| 仓库代码（scripts/ src/ configs/ tests/） | ⬜ 待传（T4 提交，含 `configs/response_sft_t4_longweight.json`） | commit 见 PR |

预计 scp 总时长：数据约 230 MB，按 700 KB/s 多流约 6–10 分钟（基座已免传）。

## 四、剩余工作与执行方式（2026-10-29 定稿）

训练已完成（18:33 PASS，4,664 步 / mean loss 2.6998 / 1.6039 s/step / reload 校验过 / 显存峰值 7.21G），adapter 在本机
`weights/t4_longweight_2ep/`。剩余全部是**新实例上的同卡生成与指标**，由无人值守管线一次跑完，不需要人盯：

1. **上传**（实例开通后）：`bash scripts/remote/upload_workload.sh`——4 条并行流 rsync（仓库树 / 两个 adapter / val 输入 / 条件与筛选集，约 150 MB），
   传完双侧 sha256 核验（`sha256sum -c MANIFEST.sha256`），同时远程后台经 hf-mirror 拉基座（约 46 分钟，断流自动重试），
   哈希等于 `dd924a11…` 才写 `base_ready`。
2. **启动**（一条命令）：`bash scripts/remote/launch_workload.sh`——自动探测 python/选卡、生成 `workload.env`、
   装 cron 自愈（装不上则用户态循环，二选一）、`setsid` 起管线。
3. **管线内容**（`run_workload.sh` + `workload_stages.sh`，同卡串行；stage marker + 推理 journal 双幂等，进程死亡/重启后续跑）：
   - 生成 8 臂：T4/D4v2 各 160 条筛选集（无条件）+ T4/D4v2 各 2,354 条全量 val（无条件，**全新 resume 目录**——断点 journal 的
     identity 钉死环境字段，不能跨机续，bitahub 已产的 160+747 条只作历史证据）+ T4/D4v2 各 160 条筛选集（仅情感、情感+画像，T5 用）。
   - 指标：筛选集两臂对比（`compare_response_arms`）、val 错误分析（`analyze_response_errors`，>200 桶 ROUGE-L 与偏短率）、
     T5 四组条件对比。
   - 预注册条件臂全量触发线（`decide_conditional_fullval.py`，先定后跑）：某条件臂相对同 adapter 关闭臂 ROUGE-L 提升 ≥0.005
     且逐条 wins−losses ≥20，才自动补跑该臂全量 val（上限 +2.7h）。
4. **预计耗时**：约 8 小时（区间 6.5–11：基座拉取 ~0.8h + 6 个筛选集臂 ~1.3h + 2 个全量臂 ~4.8–5.6h + 指标 ~0.3h；
   触发条件臂全量再 +2.7h）。显存按 bs=1 变长序列估计 7–8G，24G 卡无压力；磁盘约 5G。
5. **回来后**：拉回全部产物 → 按 2.2 判定线出结论（达标发 Release + 提 PR 交 A 审，未达记 no-go）→ 用户销毁实例。
   完整恢复方法见 `docs/UNATTENDED_GPU_WORKLOAD.md`。

## 五、执行结果与判定（2026-10-29 20:56 完成，无人值守管线一次跑完）

实例：SeeTaCloud 单卡 4090（`connect.nmb1.seetacloud.com:42033`）。14:10:28 PREFLIGHT_OK → 20:56:17 WORKLOAD_DONE，
**总耗时 6h46m，全程无人值守、零中断**（8 个臂的 journal 各只有 1 次 start/end，全部同卡单次生成；预检/上传核验/
stage 幂等全部按设计工作）。触发线判定器输出 NO_TRIGGER，未补跑条件臂全量。

### 5.1 T4 主判定：**NO-GO**（长样本加权未产生可测收益）

全量 val 2,354 条（同卡，T4 vs D4 v2）：

| 指标 | T4 | D4 v2 | Δ | 判定线 |
|---|---:|---:|---:|---|
| >200 字桶 ROUGE-L | 0.1956 | 0.1950 | **+0.0006** | 需 ≥ +0.01 → **未达** |
| 全集 mean ROUGE-L | 0.2605 | 0.2603 | +0.0002 | 降幅 ≤0.005 → 通过（无提升） |
| 逐条 wins/losses/ties | 1038 / 1016 / 300 | — | 掷硬币 | — |
| mean BLEU-4 | 0.1131 | 0.1122 | +0.0009 | — |

160 条筛选集：T4 0.2494 vs D4v2 0.2508（Δ=−0.0014），逐条 71/73/16。分桶看 T4 只在 101–200 字桶微优
（0.2156 vs 0.2139）、短桶微劣，**长桶没有发生预期的长度匹配改善**（1.2 节倒 U 的 Q2 区间未被打动）。
格式问题量级与 D4v2 相当（markdown 粗体 337、列表 174、近空 36）。

**结论：NO-GO。** 按 2.2 节先定后跑的判定线记录负结果，不盲搜第二组权重；T4 adapter 留档本机
`weights/t4_longweight_2ep/`（sha256 `bc6ca023…`），**不发 Release**；交付候选维持 **D4 v2** 不变。

### 5.2 T5 条件矩阵：**NO-GO**（第三次条件负结果，维持条件关闭）

160 条筛选集、同卡、三条件 × 两 adapter（预注册线：ΔROUGE-L ≥0.005 且 wins−losses ≥20）：

| adapter | 条件臂 | ROUGE-L | vs 关闭臂 Δ | 逐条 W/L/T | margin |
|---|---|---:|---:|---:|---:|
| D4v2 | 仅情感 | 0.2524 | +0.0016 | 69/57/34 | +12 |
| D4v2 | 情感+画像 | 0.2513 | +0.0005 | 65/55/40 | +10 |
| T4 | 仅情感 | 0.2470 | −0.0024 | 51/67/42 | −16 |
| T4 | 情感+画像 | 0.2490 | −0.0004 | 65/57/38 | +8 |

最好的一臂（D4v2 仅情感）Δ 与 margin 均低于预注册线——160 条上的 +0.0016 是噪声量级，与 10/23 同设备实验
（Δ=−0.0007）和 10/15 C Day3 条件臂（−0.2%）方向一致。**维持部署默认：条件关闭。** 画像臂结论标注
provisional（B 对画像 v1 的 T3 独立复核仍未 recorded），但因未过线不影响采用决策。

### 5.3 T6 资源与一致性数据（不额外占 GPU）

- **速度**：8 臂均为 ~4.1–4.3 s/条（p50 2.6–3.0s，~19–20 tok/s）；两个全量 val 臂各 10,042s / 10,094s（约 2.79h）
- **显存**：`gpu_usage.log` 4,869 个采样点，峰值 5,299 MB / 24G（bs=1 推理无优化器状态，远低于训练的 7.2G）
- **停止行为**：8 臂 empty_responses 全为 0；max_new_tokens=256 与部署一致
- **训练推理一致性**：T4 adapter 经上传 manifest + `ADAPTER_SHA256S` 完整 sha256 双侧钉扎（`bc6ca023…`/`6b75245a…`），
  训练侧 reload 校验见 §二；LoRA 合并探针按计划原文为可选，未安排

## 六、验证证据索引

- 本报告所有数字由主会话从 `reports/d_day4_response_error_analysis.json` 与 `data/processed/b1_full_v1/response_train.jsonl` 直接重算（脚本口径见文中），未采信子智能体结论。
- 训练数据哈希以 `data/processed/b1_full_v1/SHA256SUMS` 为准。
- 本地 BLEU-4/ROUGE-L 为字符级诊断实现（`src/evaluation/metrics.py`），官方指标未公布，本报告所有分数不冒充官方分数。
- 新卡评测前的本地接线冒烟（2026-10-29，WSL CPU，命令形状与远端管线逐字一致）：无条件臂 2 样本 PASS（35.2s）、仅情感臂 2 样本 PASS（condition_mode=predicted、coverage 2/2）；`--resume-dir` 双跑实证复用（第一次 new=1、第二次 reused=1/new=0，指纹一致）；`compare_response_arms`（2 臂与 3 臂）与 `analyze_response_errors` 均用历史产物实跑通过。
