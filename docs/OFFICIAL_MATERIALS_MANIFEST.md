# 官方资料清单

`official_materials/` 是原始资料的逐字节副本。开发产物位于 `team_project/`；任何清洗或转换数据只能写入 `team_project/data/processed/`。

| 文件名 | 相对路径 | 类型 | 字节 | 修改时间 | SHA-256 | 用途 | 已读 | 提交必需 |
|---|---|---|---:|---|---|---|---|---|
| README.md | README.md | md | 1404 | 2026-10-12T10:19:47 | 1169b39704fed500e966f71389aa6df9f3a638126d95e2b446ea226bebaf5785 | 官方资料总说明与提交包内专家评审材料路径要求 | 是 | 视最终镜像清单；数据集本身不提交 |
| README.md | participant/README.md | md | 4279 | 2026-10-12T10:19:47 | d0dd0df7e8cceb990aae2a79e3de94994c09c117a21709a99442e9c792a202c7 | participant v2 运行契约、后端、产物和计时说明 | 是 | 视最终镜像清单；数据集本身不提交 |
| inference_config.json | participant/configs/inference_config.json | json | 478 | 2026-10-12T10:19:47 | bfc8fdc4f14d717c2ca06ecf4f84aa61405edc546237c2dead45a89be51b248a | 官方参考推理配置 | 是 | 视最终镜像清单；数据集本身不提交 |
| requirements.txt | participant/requirements.txt | txt | 47 | 2026-10-12T10:19:47 | f138a3692fc198597db34b45a50a3078a991bcc80a6a2afe8a3c0a2c534297a4 | 官方参考依赖下限 | 是 | 视最终镜像清单；数据集本身不提交 |
| run_inference.py | participant/run_inference.py | py | 19467 | 2026-10-12T10:19:47 | 8d91ecffd58f7c9cd1c46696a74f73db648c88019ad30cd003d4b686d33dfaf9 | 官方参考推理实现 | 是 | 视最终镜像清单；数据集本身不提交 |
| start.sh | participant/start.sh | sh | 4488 | 2026-10-12T10:19:47 | f8ed932171974a8e99214efe5144982938475163a363fb421f2a84f2640e5ff4 | 官方唯一启动入口脚本 | 是 | 视最终镜像清单；数据集本身不提交 |
| submission_schema.json | participant/templates/submission_schema.json | json | 1590 | 2026-10-12T10:19:47 | 91c42d56ae433212aab7905eabdbb96cf8463c2f887dc46b3f31647ab3a5d591 | 官方提交对象 JSON Schema | 是 | 视最终镜像清单；数据集本身不提交 |
| submission_template.json | participant/templates/submission_template.json | json | 219 | 2026-10-12T10:19:47 | d6a630a114ffd91e83fab02deb22a1a7e2bd84266519f45bf775f48a47e31a3c | 官方提交对象样例 | 是 | 视最终镜像清单；数据集本身不提交 |
| test_inference_data.jsonl | test_inference_data.jsonl | jsonl | 6334 | 2026-10-12T10:19:47 | 194f824a70fcc5a73e05e818eb0968045c165edc0cdf31e197d135103b910ebd | 官方推理入口测试样例 | 是 | 视最终镜像清单；数据集本身不提交 |
| 咪咕仝学平台实例创建demo V1.docx | 咪咕仝学平台实例创建demo V1.docx | docx | 4226626 | 2026-10-12T10:19:47 | 308654ff2c0defb7a47e028be8e456bf53fcfef4fda0255952b8129d224dec5e | 咪咕仝学实例创建和 vLLM 公有镜像操作演示 | 是 | 视最终镜像清单；数据集本身不提交 |
| train_public.jsonl | 训练-验证-数据集/train/train_public.jsonl | jsonl | 69575005 | 2026-10-12T10:19:48 | 184a47471f8d6f70fa21b9265b0a4d4e51c3748ba25af39a801884a2da35341a | 公开训练数据 | 是 | 视最终镜像清单；数据集本身不提交 |
| train_public.schema.json | 训练-验证-数据集/train/train_public.schema.json | json | 2537 | 2026-10-12T10:19:48 | a9467e520b06a96859acaa2ff022e395d08ff6ad5e8ae3532a22cc24c630cada | 训练数据 JSON Schema | 是 | 视最终镜像清单；数据集本身不提交 |
| val_public.jsonl | 训练-验证-数据集/val/val_public.jsonl | jsonl | 6078596 | 2026-10-12T10:19:48 | 097c0639ef7370a18a54670ec3c8390ce5dc87c1ed95f436cef07b68dfed7668 | 公开验证数据 | 是 | 视最终镜像清单；数据集本身不提交 |
| val_public.schema.json | 训练-验证-数据集/val/val_public.schema.json | json | 2535 | 2026-10-12T10:19:48 | 4b995d80aba99883bda63e2fa18b161a2784e501f98316295f4618b0a0c4af10 | 验证数据 JSON Schema | 是 | 视最终镜像清单；数据集本身不提交 |
