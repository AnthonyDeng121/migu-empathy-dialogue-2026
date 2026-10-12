# 原始数据目录

此目录结构提交到 GitHub，但原始数据文件不提交。

每位成员把已经下载的官方数据放到以下相对位置：

```text
data/raw/
└─ 训练-验证-数据集/
   ├─ train/
   │  ├─ train_public.jsonl
   │  └─ train_public.schema.json
   └─ val/
      ├─ val_public.jsonl
      └─ val_public.schema.json
```

放置后使用 `docs/OFFICIAL_MATERIALS_MANIFEST.md` 中的 SHA-256 核对文件。代码必须从配置或命令行读取该路径，不得写个人绝对路径。

