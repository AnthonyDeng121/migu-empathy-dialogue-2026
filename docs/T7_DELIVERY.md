# A T7 固定候选交付

选型沿用 B T6：Day5 情感、D4v2 回复、条件关闭、空画像、生成上限 256。C T6/T7 清单已合入 main；画像 v1、情感 0.65 和 D 新训练候选不进入默认包。

## 生成候选

模型放到 `configs/integrated.json` 指定的目录后执行：

```bash
python scripts/package_t7.py --destination artifacts/participant_t7 --python-bin /root/migu-inference-20261018/bin/python
python scripts/package_expert_materials.py --output artifacts/expert_materials_t7
```

目的目录必须不存在。打包先核对冻结配置和已有模型哈希，再复制真实权重、代码、配置、官方兼容入口及两参数启动脚本。脚本可在本地生成 Linux 包；`--python-bin` 是目标镜像中的路径，存在性与依赖必须在目标环境检查。

`candidate_manifest.json` 记录源提交、是否有未提交修改、相对路径、字节数及逐文件 SHA-256，不记录本机绝对目录。清单不包含自身。`configs/integrated.original.json` 保留原始配置；原成功镜像保持为回退版本。Python 环境须随平台镜像保留，不能只上传候选目录。

## 目标环境检查

先在新目录检查模型哈希和 Python 依赖。若系统支持挂载与网络命名空间，可用以下方式检查官方绝对路径，保留宿主 `/root/participant`：

```bash
unshare --mount --net bash -c 'set -eu
mount --make-rprivate /
mount --bind /root/migu-t7-candidate-20261104 /root/participant
cd /root/participant
bash start.sh /root/migu-t7-source-20261104/data/raw/test_inference_data.jsonl /root/migu-t7-check-20261104/cold1'
```

更换为新的结果目录再启动一次。两次都应加载真实模型，输出完整 ID、合法 schema、非空回复、空画像及关闭条件，并核对模型和包文件未变。没有外部网络连接和路由的命名空间用于运行期断网验证；HF 离线环境变量只作为补充。少量样例的性能报告 `complete=false` 属预期，不作为 100 条正式性能或全量评测。

## 交付边界

B T6 的历史 tokenizer/template 对应及 effective repetition_penalty 未完全确认，限制历史逐字复现声明；本次当前资产的逐文件哈希和入口检查另行记录，不将历史 UNKNOWN 改成 PASS。不重生成全量预测，也不宣称提升官方分数。

验收日志、输出和环境快照只保存于本地忽略目录 `.codex_tmp/acceptance/`。目标实例中的临时检查结果在下载后移除，候选及专家材料保留为交付物。官方基础镜像来源还需平台记录确认；主机自报环境不能证明来源。候选包与入口自检不等同于上传大赛镜像或正式提交。最终发布依照既有 A/B 复核流程。
