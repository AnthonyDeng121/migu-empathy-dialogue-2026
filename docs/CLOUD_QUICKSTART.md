# RTX 4090 云端试跑

这是两模型推理的准备包，不会启动训练，也不等同于正式比赛镜像验收。

## 模型交付

将完整文件夹放到项目目录下：

- C：`weights/emotion_profile/`，包含 `checkpoint_metadata.json`、`heads.safetensors`、`encoder/`（配置及权重）和 tokenizer 文件。正式候选应使用 C 全量训练的产物，不用三步试跑产物替代。
- D 基座：`weights/hf/Qwen2.5-1.5B-Instruct/`，完整配置、分片权重和 tokenizer；revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- D adapter：`weights/response_sft_d2_1p5b/`，包含 `adapter_config.json`、`adapter_model.safetensors`。当前候选 adapter SHA-256 为 `62a695ea594a1546e217ee5edd691131051a4e64e3ecfcec91af75de7a133fcf`。

D 的持久交付为 GitHub Release `d-day3-response-sft-20261015`，使用资产 `D_DAY3_RESPONSE_SFT_D2_1P5B_20261015.zip`（adapter 与 tokenizer）。基座仍需固定 revision 的本地副本；完整目录指纹见 `configs/response_artifact.json`。保留包内原始 `adapter_config.json`，不要因其中训练机路径而直接改写。merged 分卷是备用方案，生成结果与基座加 adapter 不保证一致；换用须重新全量评测，当前候选继续使用基座加 adapter。

已有云端模型不必下载再上传；在 `configs/integrated.json` 填写云端绝对路径即可。相对路径以项目根目录为准。

当前 C 候选是 Release `c-day3-retrained-20261015`（包 SHA-256 `550e9c4743671eb736338c8193b7cdc3f82386c63a86ac2d143dcf034c2f7b50`），使用从 C 源 commit `54e65f0` 引入的 `EmotionProfileCheckpoint`，不能使用旧版联合头加载方式。`classifier_format=packaged` 读取每维校准阈值并采用模型记录的标签顺序。文件哈希固定在 `configs/integrated_artifacts.json`。

该 Release 的画像完全匹配率未超过空画像基线，所以默认 `profile_mode=empty`；情感照常由 C 模型预测。`profile_mode=model` 可用于对比画像头，不代表该画像头已验收通过。

## 本机打包和上传

```powershell
python scripts/package_cloud.py
scp -P <SSH端口> artifacts/cloud_source.zip root@<服务器地址>:/root/cloud_source.zip
```

包中只含源代码、配置、测试、技术文档和三条官方样例，不含模型、报告、日志、账号或官方训练集。需要训练 C 时，另提供 B 转换的数据与 C 训练所需模型；本包只准备集成推理。

云端解压到 `/root/participant/`，另行传入模型或配置其现有路径。初次解压需使用空目录，避免覆盖现有参赛工程。

## 云端环境

使用数字人专属 `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8` 实例，先确认 4090 实际可用。这里使用 Transformers 推理，不启动 vLLM 服务；最终入口在专属镜像内运行。

优先用平台预置 Python，例如 D 使用的 `/opt/conda/envs/conda_env/bin/python`。先检查，不直接安装项目的训练 requirements，避免替换平台 PyTorch/CUDA。

若平台缺少 PEFT/Accelerate，使用继承平台软件的独立环境，避免修改原环境：

```bash
/opt/conda/envs/conda_env/bin/python -m venv --system-site-packages /root/migu-inference-venv
/root/migu-inference-venv/bin/python -m pip install --no-deps peft==0.21.2 accelerate==1.15.0
export PYTHON_BIN=/root/migu-inference-venv/bin/python
```

仅在平台已有其余依赖时使用上述补装方式；预检不通过就先修复环境。镜像冻结时需一并保存该环境，或在官方环境内按经验证的方法安装同版本补充依赖。

```bash
cd /root/participant
export PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/conda_env/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" scripts/cloud_preflight.py --verify-hashes
bash scripts/cloud_smoke.sh
```

预检只读取文件、导入软件和查看 GPU；不能代替真实模型加载。缺依赖时按错误补充并复核兼容性，不盲目升级。D 已跑通的环境：torch 2.10.0+cu128、transformers 4.57.6、peft 0.21.2、accelerate 1.15.0、safetensors 0.7.0；C 与该环境的兼容性仍需真实试跑。

## 比赛形式的启动命令

```bash
export INFERENCE_BACKEND=integrated
export PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/conda_env/bin/python}
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

产物为 `submission.jsonl` 和 `performance_report.json`。两个模型各加载一次；计时含分类、生成和输出校验，不含模型加载，记录前100条并处理完整输入。只有3条样例时 `complete=false` 是正常的；需至少100条合法输入才有完整性能统计。

默认 D 回复不注入 C 预测标签，沿用 D 已验证的无条件候选。比较后才将 `condition_response` 改为 `true`；运行时不读取验证集真值。

同名产物已存在时拒绝覆盖，请使用新结果目录。GPU 不可用时立即停止，先处理平台资源问题。样例通过后，再做100条、全量和容器断网检查；离线环境变量不能代替容器 `--network none`。

最终保存镜像前，将 `INFERENCE_BACKEND=integrated`、正确的 `PYTHON_BIN` 和模型路径固化到镜像环境，确保评委只执行官方两参数命令即可启动。当前开发 Dockerfile 仍为保底镜像，不能直接作为最终集成镜像提交。

另提供 `Dockerfile.integrated`，固定集成入口且不替换平台 PyTorch。待确认实际官方镜像引用、模型文件齐备后，可用 BuildKit 构建：

```bash
python scripts/package_expert_materials.py --output artifacts/expert_materials_day4
docker build -f Dockerfile.integrated --build-arg OFFICIAL_BASE_IMAGE=<官方完整镜像引用> --build-arg TEAM_NAME=<正式队名> --build-context model_assets=./weights --build-context expert_materials=./artifacts/expert_materials_day4 -t migu-integrated:candidate .
```

构建需要可访问的软件包源；运行不需要联网。该配方需在实际官方基础镜像上构建并执行 GPU/断网回归才算验收，源码中有 Dockerfile 不代表镜像已构建成功。专家材料由明确清单打包，含技术方案、实验记录和训练日志及 SHA-256，不包含本地验收记录；需按最终候选复核材料完整性。

若实例没有 Docker，通过平台保存当前实例镜像。先使用 `scripts/stage_submission.py --config configs/cloud_runtime.json --python /root/a-integration-20261015/venv/bin/python --destination /root/migu-submission-day4` 建立新目录；它复制模型并固化入口环境，不覆盖现有 `/root/participant`。验证新目录后再切换官方路径，最终必须重跑 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`。保存时保留指定 Python 环境，将专家材料放入 `/root/AI技术赛道-数字人综合情感对话模型-正式队名`，并核对基础镜像来源。平台保存、安全审核、B 复核和官方成功状态仍分别记录，不能以试跑代替。

验收报告仅保存在本地忽略目录 `.codex_tmp/acceptance/`，不提交或上传 GitHub。

## Day6 最新候选（2026-10-17）

集成配置锁定 C `c-day5-emotion-longtail075-20261016` 与 D `d-day4-response-sft-2ep-20261017`。
C 使用 `classifier_format=sequence`，按 C 训练脚本的 `tuple(sorted(EMOTIONS))` 解释 16 类输出，保持左截断、256 token、用户/助手中文前缀；画像保持 `profile_mode=empty`。D 使用固定 Qwen2.5-1.5B-Instruct revision 加 2-epoch adapter，不使用旧 D2 adapter 或 merged 备选；不注入分类条件，最大生成 256 token。

`configs/integrated_artifacts.json` 固定 C 文件哈希与 D adapter 权重哈希；`configs/response_artifact.json` 固定 D 基座/adapter 目录摘要。摘要按 POSIX 相对路径区分大小写排序。Release 的 HANDOFF/SHA256SUMS 是交接附件，不放入模型目录摘要范围。

正式候选必须通过预检、两次断网冷启动、同集全量推理和 B 独立复核。局部指标回退须在选型中明确：C mixed 类 F1 下降，D 21–50 字桶 ROUGE-L 下降约 0.8%；整体指标提升不证明所有切片均改善。

B 通过后，在比赛平台保存包含 `/root/participant` 实际目录及指定 Python 环境的实例镜像，记录镜像标识/digest；使用保存后的镜像恢复新实例，复核固定模型哈希并以官方两参数入口断网冷启动。原实例内目录试跑不能替代保存后恢复检查。随后选择该镜像，完成安全审核并发起官方测评，保留结果编号和截图。未确认成功状态前不能记为官方有效提交。
