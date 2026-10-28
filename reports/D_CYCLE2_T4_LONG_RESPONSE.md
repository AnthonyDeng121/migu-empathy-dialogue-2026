# D 岗位 CYCLE2 T4：长回复改进（进行中）

日期：2026-10-28（Asia/Shanghai）。分支：`feature/d-day7-deadline-standby`。状态：**CPU 侧分析完成，GPU 侧待执行**。

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

## 三、执行环境（2026-10-28 已就绪）

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

## 四、下一步（未完成）

1. 远端基座 sha256 与本机核对；传条件文件/val/筛选集/adapter/代码。
2. 实例上跑短探针（`--max-steps 120`）测步速，确认加权生效（report 的 `data.sample_weights.weighted_samples` = 5,853）。
3. 全量 2 epoch 训练（预计 4090 约 3.1h，含 +30% 余量约 4h），`--checkpoint-every 250`，三件套守护（`scripts/remote/`）。
4. 同卡顺序生成 160 条筛选集 + 全量 val 2,354 条（T4 臂与 D4 v2 对照臂）。
5. 按 2.2 判定标准出结论；达标则发布 Release 并提 PR 交 A 审，未达标记 no-go。

## 五、验证证据索引

- 本报告所有数字由主会话从 `reports/d_day4_response_error_analysis.json` 与 `data/processed/b1_full_v1/response_train.jsonl` 直接重算（脚本口径见文中），未采信子智能体结论。
- 训练数据哈希以 `data/processed/b1_full_v1/SHA256SUMS` 为准。
- 本地 BLEU-4/ROUGE-L 为字符级诊断实现（`src/evaluation/metrics.py`），官方指标未公布，本报告所有分数不冒充官方分数。
