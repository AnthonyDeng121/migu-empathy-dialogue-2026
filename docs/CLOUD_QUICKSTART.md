# RTX 4090 云端试跑

这是两模型推理的准备包，不会启动训练，也不等同于正式比赛镜像验收。

## 模型交付

将完整文件夹放到项目目录下：

- C：`weights/emotion_longtail075_b1_20261016/`，包含 `config.json`、`model.safetensors`、`tokenizer.json` 和 `tokenizer_config.json`；使用标准 sequence classification 格式。
- D 基座：`weights/hf/Qwen2.5-1.5B-Instruct/`，完整配置、分片权重和 tokenizer；revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- D adapter：`weights/response_sft_d4_2ep/`，包含 `adapter_config.json`、`adapter_model.safetensors` 与交付的 tokenizer 文件。当前候选 adapter SHA-256 为 `6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e`。

D 的持久交付为 GitHub Release `d-day4-response-sft-2ep-20261017`，使用资产 `d4_2ep_adapter.zip`（adapter 与 tokenizer）。基座仍需固定 revision 的本地副本；完整目录指纹见 `configs/response_artifact.json`。保留原始 `adapter_config.json`。先校验包内 SHA256SUMS，再将 10 个模型/tokenizer 文件放入部署目录；HANDOFF/SHA256SUMS 不纳入模型目录摘要。旧 D2 adapter 与 merged 分卷不属于本次候选。

已有云端模型不必下载再上传；在 `configs/integrated.json` 填写云端绝对路径即可。相对路径以项目根目录为准。

当前 C 候选是 Release `c-day5-emotion-longtail075-20261016`（包 SHA-256 `67db558687a42c6b10f31075ff1fecf14ffc6ba881c4d14d0bcd387b0ea0c526`），配置 `classifier_format=sequence`，通过 `AutoModelForSequenceClassification` 加载，按训练脚本的 `tuple(sorted(EMOTIONS))` 解释输出。文件哈希固定在 `configs/integrated_artifacts.json`。

该 checkpoint 仅预测情感，不含画像头，因此必须使用 `profile_mode=empty`；`profile_mode=model` 会被配置校验拒绝。

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
python scripts/package_expert_materials.py --output artifacts/expert_materials_latest
docker build -f Dockerfile.integrated --build-arg OFFICIAL_BASE_IMAGE=<官方完整镜像引用> --build-arg TEAM_NAME=<正式队名> --build-context model_assets=./weights --build-context expert_materials=./artifacts/expert_materials_latest -t migu-integrated:candidate .
```

构建需要可访问的软件包源；运行不需要联网。该配方需在实际官方基础镜像上构建并执行 GPU/断网回归才算验收，源码中有 Dockerfile 不代表镜像已构建成功。专家材料由明确清单打包，含技术方案、实验记录和训练日志及 SHA-256，不包含本地验收记录；需按最终候选复核材料完整性。

若实例没有 Docker，在从大赛专属公有镜像创建的实例内完成部署和验证，再使用平台的「上传大赛镜像」入口。使用 `scripts/stage_submission.py --config configs/cloud_runtime.json --python /opt/conda/envs/conda_env/bin/python --destination /root/migu-submission-latest` 建立新目录；配置中的模型路径应指向上述最新权重。此脚本不覆盖现有目录。验证新目录后再切换官方路径，最终必须重跑 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`。上传时保留指定 Python 环境，将专家材料放入 `/root/AI技术赛道-数字人综合情感对话模型-正式队名`。平台已明确：私有镜像创建的实例无法上传大赛镜像；「保存镜像」及私有镜像恢复不能代替参赛上传路径。上传、安全审核、B 复核和官方成功状态仍分别记录，不能以试跑代替。

验收报告仅保存在本地忽略目录 `.codex_tmp/acceptance/`，不提交或上传 GitHub。

## Day6 最新候选（2026-10-17）

集成配置锁定 C `c-day5-emotion-longtail075-20261016` 与 D `d-day4-response-sft-2ep-20261017`。
C 使用 `classifier_format=sequence`，按 C 训练脚本的 `tuple(sorted(EMOTIONS))` 解释 16 类输出，保持左截断、256 token、用户/助手中文前缀；画像保持 `profile_mode=empty`。D 使用固定 Qwen2.5-1.5B-Instruct revision 加 2-epoch adapter，不使用旧 D2 adapter 或 merged 备选；不注入分类条件，最大生成 256 token。

`configs/integrated_artifacts.json` 固定 C 文件哈希与 D adapter 权重哈希；`configs/response_artifact.json` 固定 D 基座/adapter 目录摘要。摘要按 POSIX 相对路径区分大小写排序。Release 的 HANDOFF/SHA256SUMS 是交接附件，不放入模型目录摘要范围。

正式候选必须通过预检、两次断网冷启动、同集全量推理和 B 独立复核。局部指标回退须在选型中明确：C mixed 类 F1 下降，D 21–50 字桶 ROUGE-L 下降约 0.8%；整体指标提升不证明所有切片均改善。

B 通过后，从大赛专属公有镜像实例的「上传大赛镜像」入口提交，记录平台生成的镜像标识/digest、上传和审核日志。镜像必须保留 `/root/participant` 实际目录及指定 Python 环境。上传后的镜像级验证按平台支持的方式进行，并关联上传前后的模型哈希、配置和断网官方入口证据；原实例内目录试跑不能冒充已完成上传后验证。不要为了这一步改从私有镜像创建参赛实例。审核通过后再发起官方测评，保留结果编号和截图。未确认成功状态前不能记为官方有效提交。

2026-10-18 在原大赛公有实例（SSH 端口 58346）重新部署：候选目录 `/root/migu-submission-latest-20261018`，独立推理环境 `/root/migu-inference-20261018`，官方入口 `/root/participant/start.sh`。模型和推理环境都在系统盘，不依赖 `/mnt/storage`。旧候选与数据保留；不要将旧 Day4 模型路径用于本次上传。

## ??????????2026-10-21?

??????? `configs/inference_config.json`?`configs/integrated.json` ???? `run_inference.py`??????????????????????? C/D ?????stage_submission.py ????? Python ???`configs/cloud_runtime.json` ?????????? integrated.json???????????????????? `/root/participant` ??????????????????????

????? C Day5 longtail075?D Day4 2-epoch?condition_response=false?profile_mode=empty?max_new_tokens=256??????????????????????????????????
