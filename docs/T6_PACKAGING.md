# T6 独立画像打包支持

T6 只扩展打包工具，不决定画像采用、不制作最终交付镜像。默认 `configs/integrated.json` 保持空画像，情感 Day5、回复 D4 v2、条件关闭。

## 暂存

沿用 Linux 平台命令，将配置显式切到已有固定画像候选：

```bash
python scripts/stage_submission.py --config configs/integrated_profile_v1_off.json --python /opt/migu-inference/bin/python --destination /tmp/participant-t6
```

独立画像模式复制完整 checkpoint（encoder、heads、tokenizer、元数据中的标签/阈值/max_length），并把路径改成包内相对路径。空画像模式不复制画像。暂存前验证配置及模型结构，目标目录必须不存在。Python 环境仍须保留在平台镜像中。

## Docker

`model_assets` context 现在还需包含 `profile_encoder_v1_b1_20261026/`。保留原官方基镜像和 expert_materials context。默认配置不变；验证画像路径可显式传入：

```bash
--build-arg INTEGRATED_CONFIG=configs/integrated_profile_v1_off.json
```

构建时对选中配置和独立画像配置均执行文件/哈希预检。画像复用已有 torch、transformers、safetensors；未增加依赖或替换官方环境包。实际导入/加载检查仍应在官方镜像环境进行。

## 预检

```bash
python scripts/cloud_preflight.py --config configs/integrated_profile_v1_off.json --files-only --verify-hashes
python -m unittest tests.test_t6_packaging tests.test_docker_model_paths
```

`configs/integrated_artifacts.json` 固定画像全部文件的 SHA-256，覆盖标签/阈值元数据、tokenizer、编码器与 heads；独立画像模式缺 manifest、缺/多文件或内容变化均拒绝。该清单是已有 v1 候选的文件固定记录，不表示批准采用。
