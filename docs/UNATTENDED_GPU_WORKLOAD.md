# 无人值守 GPU workload 恢复手册（D 岗位 CYCLE2 T4/T5/T6）

本手册服务于一个目标：**新 GPU 实例开通后，主人会话只做「上传 + 一条启动命令」，随后数小时不看管；
管线自动完成全部剩余 GPU 工作并自愈；回来后拉产物、判定、收尾。**

对应报告：`reports/D_CYCLE2_T4_LONG_RESPONSE.md`（§四）。计划行：`docs/EXECUTION_PLAN_CYCLE2.md` 的 T4/T5/T6 D 行。

## 1. 这次 GPU 会话要跑完什么（全部剩余 GPU 工作，先定后跑）

T4 训练已完成，本会话**不含任何训练**，只做同卡顺序生成与指标：

| 顺序 | stage | 输入 | 条数 | 预计 |
|---|---|---|---:|---|
| 1 | T4 筛选集（无条件） | `t2_selection_input.jsonl` + T4 adapter | 160 | ~7–15 min |
| 2 | D4v2 筛选集（无条件） | 同上 + D4v2 adapter | 160 | ~10–15 min |
| 3 | T4 全量 val（无条件） | `response_val.jsonl` + T4 adapter | 2,354 | ~2.6–3 h |
| 4 | D4v2 全量 val（无条件） | 同上 + D4v2 adapter | 2,354 | ~2.2–2.6 h |
| 5–8 | T4/D4v2 筛选集 × {仅情感, 情感+画像} | + C Day5 条件 / 情感+画像 v1 条件 | 160×4 | ~10–15 min/臂 |
| 9–13 | 指标：T4 两臂对比、val 错误分析（>200 桶/偏短率）、T5 四组条件对比 | CPU | — | ~15 min |
| 14 | 预注册条件臂全量触发线判定（见下） | CPU | — | 秒级 |
| 15 | （仅当触发）条件臂全量 val | 由决策 JSON 指定 | 2,354 | ~2.6–3 h |

**同设备纪律**：所有臂同一张卡、`CUDA_VISIBLE_DEVICES` 全程钉死，串行生成；跨设备比较一律不采信。
**断点 journal 不能跨机续**（identity 钉死 python 路径/平台/包版本/GPU 名等，设计使然）：新实例一律用**全新 resume 目录**，
bitahub 实例已产的 160+747 条只作历史证据（在 git 里），不复用。

**预注册判定线（看结果前定稿，`scripts/remote/decide_conditional_fullval.py`）**：
某条件臂相对**同 adapter** 的关闭臂 ROUGE-L 均值提升 ≥0.005 **且**逐条配对 wins−losses ≥20，才自动补跑该臂全量 val（最多一次）。
T4 主判定线不变：>200 字桶 ROUGE-L 相对 D4v2 同设备基线提升 ≥0.01 且全集降幅 ≤0.005。

**预计总耗时：约 8 小时**（区间 6.5–11：基座拉取 ~0.8h + 6 个筛选集臂 ~1.3h + 2 个全量臂 ~4.8–5.6h + 指标 ~0.3h；
触发条件臂全量再 +2.7h）。显存预计 7–8G（bs=1 变长序列），磁盘约 5G。

## 2. 实例开通后：主人会话的两步

```bash
# 0) 连接参数（实例开通时从平台页面拿；WSL 侧域名可能解析到轮换的 Clash fake-IP，以当次 nslookup 为准）
export UPLOAD_HOST=<实例域名或IP> UPLOAD_PORT=<端口> UPLOAD_USER=root PASSFILE=<askpass脚本>
# 1) 上传 + 双侧核验 + 基座后台拉取（约 5 分钟传完；基座 hf-mirror 约 46 分钟，断流自动重试）
bash scripts/remote/upload_workload.sh
# 2) 一条命令启动（自动探测 python/选卡、生成 workload.env、装 cron 自愈、setsid 起管线）
ssh … "bash /root/migu/repo/scripts/remote/launch_workload.sh"
```

之后主人会话可以停机；管线全自动运行。SSH 网关约 2.5 分钟断流是正常的，所有远程任务都已 setsid 后台化，
轮询用短命令（`tail`、`cat`），不要把长任务挂在前台。

## 3. 管线怎么自愈（无人值守的关键）

- `run_workload.sh`：预检（python/torch/peft、CUDA、基座哈希、上传清单哈希）→ 按 stage 顺序执行 →
  条件臂触发判定 → 收尾校验 + `WORKLOAD_DONE`。每个 stage 一个 marker（`stages/NN_<name>.done`），**重启后跳过已完成 stage**；
  推理自身还有 journal（`--resume-dir`），进程死在半路续跑剩余样本。stage 失败 → 退出码 1，等 supervisor 重试（有预算上限）。
- `workload_supervisor.sh`：cron 每 10 分钟确认管线存活（死了就拉），全部 arm 行数达标 + 末行 JSON 合法 + 指标产物存在才判 ALL_DONE；
  重试超预算写 `WORKLOAD_GAVE_UP` 停工（不无限重启）。
- cron 装不上时（无网/无包）自动退用户态循环，二者只选其一，不会双重启动。

## 4. 回来后：拉产物与判定（主人会话重启后）

```bash
# 1) 看状态（短命令）
ssh … "cat /root/migu/repo/workload_status.json; tail -5 /root/migu/repo/workload.log"
# 2) 拉回全部产物（results/response_gen/*newcard*、reports/*newcard*、workload.log、gpu_usage.log 等）
# 3) 本地判定：
#    - T4：>200 桶 ROUGE-L（t4_val_error_analysis_newcard.json 的 arms.t4.rouge_l_by_gold_bucket）vs D4v2 同卡基线，
#      提升 ≥0.01 且全集 mean_rouge_l 降幅 ≤0.005 → 达标（发 Release + 提 PR 交 A 审）；否则 no-go 记录。
#    - T5：t5_*_vs_off_newcard.json 的四组对比 + t5_conditional_decision.json（trigger/reason）。
#    - T6：各臂 performance_report 的速度/停止行为 + gpu_usage.log 显存。
# 4) 更新报告/计划行/记忆，commit+push，需要时提 PR；确认产物全部在本机后，用户销毁实例。
```

## 5. 已知坑（都已在脚本/流程里防住）

- SSH 网关 2.5 分钟断流 → 一切远程长任务 setsid nohup；轮询用短命令。
- WSL 域名解析到轮换 fake-IP → 连接参数以当次 `nslookup` 为准；askpass 放 `/home/cubesaint/`（`/tmp` 会被空闲回收）。
- 镜像可能没有 cron → launch 脚本自动 apt 安装，装不上退用户态循环。
- 裸 cron 环境不读 env 文件会秒退 → crontab 行先 `. $ROOT/workload.env`。
- pkill/pgrep 模式自匹配 → 查进程用 PID 或字符类；supervisor 自身命令行不含 "run_workload.sh" 字面量。
- hf-mirror 慢且会断 → 基座拉取带重试循环，哈希等于 `dd924a11…` 才写 `base_ready`，管线预检会等它。
- 30G 小盘实例 → 本 workload 无训练 checkpoint，产物约 5G，无压力；基座拉取失败时可用 `UPLOAD_BASE=1` 改本机上传。
