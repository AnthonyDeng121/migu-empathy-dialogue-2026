# scripts/remote — 无人值守长训练三件套

周期 checkpoint（训练脚本 `--checkpoint-every` / `--resume-from`）解决「进程死了」；
这三个脚本解决「进程没人拉」和「机器重启了」。全部参数经环境变量注入，不含硬编码路径。

| 脚本 | 作用 |
|---|---|
| `train_resilient.sh` | 训练进程级守护：死了就从最新 checkpoint 重拉（训练脚本会先校验实验身份，不一致直接拒绝，不会张冠李戴） |
| `watch_and_generate.sh` | 守候训练 wrapper 退出后，按 `GEN_CMDS` 逐条顺序生成各臂（同卡串行，守住同设备比较纪律） |
| `supervisor.sh` | cron 自愈：每 10 分钟确认 wrapper/管线存活（死了就拉），两臂文件行数达标且末行 JSON 完整才判 ALL_DONE；机器重启也不怕（cron 自启） |

## 用法（以 CYCLE2 T3 为例）

**环境必须落成 env 文件**：cron 以裸环境运行，直接 `export` 后装 crontab 会让
supervisor 一启动就因未设置变量退出，"机器重启自愈"实际是断的。把 ROOT/OUT/PY/
TRAIN_CMD/GEN_CMDS/GPU/ARMS/ARM_LINES 全部写进 `$ROOT/remote_env.sh`，cron 行与
手动启动都先 source 它：

```bash
export ROOT=/home/hanrongduo/liuhuanfeng
export OUT=$ROOT/weights/t3_unconditional_2ep
export PY=/home/7T/shared/miniconda3/envs/torch/bin/python
export TRAIN_CMD="CUDA_VISIBLE_DEVICES=3 $PY scripts/train_response_sft.py \
  --train data/processed/b1_full_v1/response_train.jsonl \
  --train-format b1 --config configs/response_sft_unconditional_2ep.json \
  --model-dir base_qwen15b --device cuda --local-files-only \
  --output-dir $OUT --report-json reports/t3_uncond_2ep_report.json \
  --report-md reports/t3_uncond_2ep_report.md --checkpoint-every 250 {RESUME}"
# 其余：GPU / GEN_CMDS / ARMS / ARM_LINES 同样写入（ARMS 只放行数一致的文件；
# 160 条筛选集与 2354 条全量 val 混在一个 ARM_LINES 里会永远判不齐）
{
  echo "export ROOT=$ROOT"
  echo "export OUT=$OUT"
  echo "export PY=$PY"
  printf 'export TRAIN_CMD=%q\n' "$TRAIN_CMD"
  printf 'export GEN_CMDS=%q\n' "$GEN_CMDS"
  echo "export GPU=3"
  printf 'export ARMS=%q\n' "$ARMS"
  echo "export ARM_LINES=$ARM_LINES"
} > "$ROOT/remote_env.sh"

setsid nohup bash scripts/remote/train_resilient.sh > /dev/null 2>&1 &

export GPU=3
export GEN_CMDS="$PY scripts/run_response_inference.py --input results/response_gen/t2_selection_input.jsonl \
  --model base_qwen15b --adapter $OUT --output results/response_gen/t3_unconditional.jsonl \
  --max-new-tokens 256 --device cuda
$PY scripts/run_response_inference.py --input results/response_gen/t2_selection_input.jsonl \
  --model base_qwen15b --adapter weights/response_sft_d4_2ep \
  --output results/response_gen/t3_d4v2_off_samedevice.jsonl --max-new-tokens 256 --device cuda"
setsid nohup bash scripts/remote/watch_and_generate.sh > /dev/null 2>&1 &

export ARMS="results/response_gen/t3_unconditional.jsonl results/response_gen/t3_d4v2_off_samedevice.jsonl"
export ARM_LINES=160
( crontab -l 2>/dev/null | grep -v supervisor.sh; \
  echo "*/10 * * * * . $ROOT/remote_env.sh && bash $ROOT/scripts/remote/supervisor.sh >> $ROOT/supervisor.log 2>&1" ) | crontab -
```

## 无人值守生成型 workload（无训练，仅批量生成+指标）

上面的三件套服务「训练 + 训练后生成」。CYCLE2 T5 只剩批量推理：模型已训完，新实例上要无人值守地
跑完 8 个生成臂 + 5 个指标 + 判定器触发的第 9 个臂。这套四件套就是干这个的，**不跑任何训练命令**。

| 脚本 | 作用 |
|---|---|
| `launch_workload.sh` | 实例开通后**执行一次**的入口：探测带 CUDA 的 python、挑空闲最大的卡、生成 `workload.env`、装 cron（装不上就用户态循环）、`setsid nohup` 拉起主管线。幂等可重跑 |
| `run_workload.sh` | 主管线：预检 → 逐 stage 生成/算指标 → 条件臂 → 产物审计。被 supervisor 反复重启，靠 marker 续跑 |
| `workload_stages.sh` | 默认 stage 定义（可被 `STAGES_FILE` 替换）：定义 `workload_stages()`，打印 `名字\|命令` 一行一个 |
| `workload_supervisor.sh` | cron 每 10 分钟自愈：产物齐了判 `ALL_DONE`，管线死了就拉，连续白拉满 `MAX_ATTEMPTS` 次写 `WORKLOAD_GAVE_UP` |

### workload.env 契约

`launch_workload.sh` 自动生成，cron / supervisor / run_workload.sh 三方 source 同一份：

```
ROOT / PY / GPU                     远程仓库根、带 torch+transformers+peft 的 python、CUDA_VISIBLE_DEVICES
MODEL_DIR / T4_ADAPTER / D4V2_ADAPTER  基座快照与两个 adapter（相对 ROOT）
SEL_INPUT / VAL_INPUT               160 条筛选集 / 2354 条全量 val 输入
C5_CONDITIONS / BOTH_COND_SEL / BOTH_COND_VAL   三份条件文件（emotion-only / 筛选集 both / val both）
OUT_DIR / ARMS_SPEC / DONE_FILES    产物目录、"路径:期望行数" 清单、额外的完成判据
STAGES_FILE / BASE_WAIT_MINUTES / ADAPTER_SHA256S / MAX_ATTEMPTS   可选覆盖
```

`ARMS_SPEC` 格式是 `相对路径:期望行数`，空格分隔（160 与 2354 两种行数可以共存，因为每个文件各带自己的期望值）：
`results/response_gen/t4_longweight_selection_newcard.jsonl:160 .../d4v2_val_newcard.jsonl:2354`。
条件臂的输出路径只有判定器跑完才知道，所以 run_workload.sh 把它写进
`workload_arms_effective.txt` / `workload_done_effective.txt`，supervisor 优先读这两份。

### 预检不过，绝不生成

任一步失败都会写 `preflight.json`（含失败原因、python/torch/transformers/peft/accelerate 版本、GPU 名、基座 sha256 与等待秒数）
并 `exit 1`，一个 stage 都不会碰：

1. **python 栈**：`import torch, transformers` 必须本来就能导入——镜像预装 torch 2.5.1（与 `requirements-response.txt`
   钉版一致），代码在 transformers 4.57.6 与 5.17.0 两个栈上都验证过，所以这两个**绝不安装、绝不改动**；
2. **依赖修复**：`import peft` 或 `import accelerate` 失败时，执行
   `$PY -m pip install --no-input peft==0.21.2 accelerate==1.15.0`（钉版同上）后复查，修了哪些模块、最终版本号、
   pip 日志（`workload_deps_install.log`）都记进 `preflight.json`；pip 本身失败同样算预检失败；
3. **CUDA**：`import torch, transformers, peft; assert torch.cuda.is_available()`；
4. 等 `$ROOT/base_ready` 出现，且其中记录的 sha256 与 `model.safetensors` 实际值一致（默认再核一遍已知良好值
   `dd924a11…`，不一致直接失败；`BASE_SHA256_STRICT=0` 可降级为告警）；
5. `sha256sum -c MANIFEST.sha256`（上传脚本生成的清单，覆盖全部上传文件）；
6. 两个 adapter 的 `adapter_model.safetensors` sha256（`ADAPTER_SHA256S="路径:sha256前缀"`，缺省只验存在性+大小并告警）。

**目标实例（AutoDL）**：PyTorch 2.5.1 镜像、Python 3.12、Ubuntu 22.04、单卡 4090 24G。conda 在 `/root/miniconda3`，
非交互 ssh 的 PATH 里没有它，所以 `launch_workload.sh` 依次显式试 `/opt/conda/bin/python`、`/root/miniconda3/bin/python`、
`python3`、`python`，第一个能导入 torch 栈且 `torch.cuda.is_available()` 的胜出；都找不到就直接报错退出，
**绝不静默退到 CPU**。系统盘 30G，`ROOT` 用默认 `/root/migu/repo` 即可。

### 恢复模型：stage marker + resume journal 双重幂等

- **stage marker**：`$ROOT/stages/<NN>_<name>.done`。重启后已完成的 stage 直接跳过，只跑没打标记的；
  失败的 stage **不留** marker，supervisor 下次自然重试它。
- **resume journal**：每个生成臂带 `--resume-dir`，`src/common/inference_resume.py` 的内容寻址 journal
  会续上写了一半的 JSONL（指纹含设备/GPU/代码路径，跨机复用一个 journal 会直接报错，不会混两个设备的字节）。
- 进程被硬杀、SSH 断流、机器重启都一样：marker 决定跳过哪些，journal 决定某个臂从第几条继续。

### cron / 用户态二选一

镜像可能没有 cron。`launch_workload.sh` 先试 `apt-get install -y cron`：装得上就装 crontab 行
`*/10 * * * * . $ROOT/workload.env && bash $ROOT/scripts/remote/workload_supervisor.sh >> $ROOT/supervisor.log 2>&1`
（**先 source env 文件**——裸 cron 不设变量，supervisor 会秒退，这是三件套踩过的坑）；装不上就用
`setsid nohup bash -c 'while true; do … sleep 600; done'` 的用户态循环。两条路只会选一条，不会双重启动。

### 用法

```bash
# 实例开通后，执行一次（幂等）
bash scripts/remote/launch_workload.sh
# 查看：tail -f $ROOT/workload.log；$ROOT/preflight.json；$ROOT/workload_status.json
# 完成判据：grep -c WORKLOAD_DONE $ROOT/workload.log
```

## 纪律

- **守候 wrapper，不要守候训练进程**：wrapper 会把死掉的训练立刻重拉起来，守候训练进程会误判训练结束。
- **cron 是临时设施**：任务结束后把 crontab 里的 supervisor 行删掉，不留永久定时任务。
- **恢复正确性不由「没中断」证明**：证据是 `tests/test_train_response_sft.py` 的故障注入 e2e（tiny 模型上模拟存档后死亡，恢复后最终权重与未中断运行逐字节一致）。本目录脚本只保证「死了有人拉、重启不瞎续」。
- 产物（权重/预测/报告）必须取回本机或入仓库；远程是消耗品。
