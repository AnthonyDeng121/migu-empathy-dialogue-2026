# 推理断点恢复

集成推理和 D 独立回复推理均支持显式 `--resume-dir`。首次运行即需启用此参数，后续用相同命令和同一进度目录继续；没有启用时产生的旧临时文件不能转换成恢复进度。默认官方入口仍为 `bash start.sh TEST_FILE RESULT_DIR`，不启用恢复。

## 集成推理

```bash
python scripts/run_integrated.py TEST_FILE RESULT_DIR \
  --config configs/integrated.json \
  --resume-dir .codex_tmp/progress/integrated_run1
```

## D 独立回复推理

```bash
python scripts/run_response_inference.py \
  --input data/processed/val_eval_input.jsonl \
  --model weights/hf/Qwen2.5-1.5B-Instruct \
  --adapter weights/response_sft_d4_2ep \
  --output results/run1/response.jsonl \
  --performance-report results/run1/performance.json \
  --device cuda \
  --resume-dir .codex_tmp/progress/response_run1
```

需要预测条件时，继续使用现有 `--condition-jsonl` 和 `--condition-fields {both,emotion}`，整个条件文件内容也参与恢复校验。示例模型路径应按实际已交付目录填写。

## 恢复契约

- `manifest.json` 保存输入原始字节（因此包括内容和顺序）、全部推理配置、模型目录内所有文件的流式 SHA-256 摘要，以及 `src/`、`scripts/` 下所有 Python 文件和运行环境指纹。模型文件包含 tokenizer、标签、阈值、生成默认配置和 adapter；不以文件大小或修改时间代替内容校验。
- 环境包括 Python、平台、解释器路径、主要依赖版本、CUDA/cuDNN、可见 GPU 型号、确定性选项和相关环境变量。同一模型路径下的文件替换会被检测。为保守起见，无关推理脚本变化也会使旧指纹失效。
- 校验失败明确报错，使用新的进度目录重新运行。不要编辑 manifest 绕过检查。文件哈希会读取全部权重，启动时有磁盘开销；不做按时间戳跳过校验的优化。
- `records.jsonl` 每完成一个合法预测后写入并 `fsync`，记录输出、索引、样本计算时间和生成 token 数，并用校验链检测完整行损坏。恢复只接受与输入严格一致的连续前缀。最后一行若没有换行符则视为未提交并截断重算；带换行的损坏记录直接报错。
- `writer.lock` 使用操作系统文件锁，防止同一进度目录同时运行两个写入者。进程被杀死后锁会释放，锁文件保留无需手动删除。不同进度目录仍需使用不同输出路径。
- 进度目录必须放在代码与模型资产目录之外；权重和代码在一次任务及其恢复期间应固定。迁移环境、修改代码或变更配置后应开启新目录。
- 最后一个未持久化的样本可能重算，已经持久化的合法记录不会重算。已完成进度可以再次发布结果，但当前实现仍加载模型和进行文件校验。
- 只有全部样本完成后，才通过临时文件替换正式预测；异常时保留进度及原正式文件，保留输出目录其他文件。报告也独立原子写入。预测和报告不构成跨文件事务：若进程在两次替换之间终止，重跑可从进度重新发布两者，消费者不应在命令成功前读取结果。

## 计时含义

所有启用恢复的运行（包括首次）均标记 `resume.cold_start=false`，报告版本包含 `resume`，不得作为正式冷启动性能成绩。

- 集成报告的 `latencies_ms` 仍对应输入前 100 个样本，包含缓存中已完成样本的原始计算时间。`rounds` 为实际已计时数量，`complete=false`，明确不作为正式性能验收。分类、提示、编码、生成、解码及校验的计时范围保持原定义，排除进度落盘开销。
- 回复报告的 `total_seconds`、均值及 token/s 使用跨轮次累计的成功 `generate` 调用时间，`full_set_estimate_seconds=null`，不外推冷启动吞吐量。未开启恢复时维持原有端到端循环计时。
- `resume.cumulative_successful_sample_seconds` 汇总成功样本的计算时间；`current_attempt_wall_seconds` 记录当前尝试从进度校验完成后的墙钟时间（含加载与写入开销，不含启动前文件指纹计算）。
- `attempts.jsonl` 逐次记录开始、结束、完成数量和本次墙钟时间；正常抛出异常也记录失败结束。硬中断可能只有开始事件，无法完整统计该次失败计算，必须保留这个限制，不以成功样本时间声称总 GPU 用量。

## 本地验证

```bash
python -m unittest tests.test_inference_resume -v
python -m unittest discover -s tests -q
```

恢复测试使用假模型、小文件和独立子进程，不需要下载权重或启动公共 GPU。覆盖集成/回复故障恢复、硬退出后恢复、尾行中断、缓存损坏、并发锁、输入/模型/条件/配置/代码/环境变化，以及跨轮次前 100 条计时。
