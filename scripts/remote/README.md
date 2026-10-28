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

## 纪律

- **守候 wrapper，不要守候训练进程**：wrapper 会把死掉的训练立刻重拉起来，守候训练进程会误判训练结束。
- **cron 是临时设施**：任务结束后把 crontab 里的 supervisor 行删掉，不留永久定时任务。
- **恢复正确性不由「没中断」证明**：证据是 `tests/test_train_response_sft.py` 的故障注入 e2e（tiny 模型上模拟存档后死亡，恢复后最终权重与未中断运行逐字节一致）。本目录脚本只保证「死了有人拉、重启不瞎续」。
- 产物（权重/预测/报告）必须取回本机或入仓库；远程是消耗品。
