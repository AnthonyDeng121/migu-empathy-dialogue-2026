# C Day 3 重训补充交接：情感 + 用户画像

**交接结论：情感分类候选在 B1 v1 验证集上有提升；画像预测仍未超过全空画像完全匹配基线。整体联合模型处于 Integration Gate，不能标记为全部验收通过。**

本次使用本机 Apple M5 / PyTorch MPS 训练，没有消耗平台 GPU 卡时。指标来自实际训练和保存后重载推理；数据集与运行摘要见[机器可读结果](c_day3_retrained_metrics.json)。

## 情感模型

从现有情感 baseline checkpoint 出发，在 B1 v1 上做 1 轮类别加权微调。损失权重采用归一化的类别频次平方根倒数，并限制在 `[0.35, 4.0]`，降低长尾类别训练不足。

| 同一验证集（2,354 条） | Accuracy | Macro-F1 |
|---|---:|---:|
| 微调前 baseline | 67.84% | 0.4032 |
| B1 类别加权微调后 | **70.94%** | **0.5738** |
| 变化 | +3.10 个百分点 | +0.1706 |

训练集 18,653 条、验证集 2,354 条，conversation 级重叠为 0。训练循环计时 1,479.459 秒（约 24.7 分钟，不含数据读取、预处理和模型保存）。类别不平衡仍有影响：`care` 在验证集 11 条中 F1 为 0；`surprise` 只有 9 条，召回偏低。

## 用户画像

冻结上面的情感编码器与情感分类头，在编码表征上训练三组画像头。训练前按 conversation 从 B1 训练集划出 15% 作阈值校准，剩余 85% 用于拟合画像头；官方验证集没有参与 checkpoint 或阈值选择。画像头按训练校准集 micro-F1 选择 checkpoint 和各维阈值。

| 画像维度 | 验证 Micro-F1 | 完全匹配率 | 全空画像完全匹配率 |
|---|---:|---:|---:|
| personality_traits | 0.2874 | 0.4044 | 0.6185 |
| interests | 0.1997 | 0.6075 | 0.7489 |
| style | 0.2703 | 0.2761 | 0.5174 |
| 三维宏平均 | **0.2525** | **0.4293** | **0.6283** |

画像模型学到了非空预测，但宏平均完全匹配率低于始终输出空数组的基线。另用训练校准集以完全匹配为目标调整阈值，验证宏平均完全匹配率为 0.6223，仍低于 0.6283。该画像头目前只能作为实验候选，不能写成通过。

画像特征提取与训练计时 345.517 秒（约 5.8 分钟）。本机保存的联合 checkpoint 重载后，在完整验证集上重新推理，情感与三维画像汇总指标均与报告一致。

## 可复现代码与模型产物

本 PR 包含训练脚本、重载推理类和上面的指标摘要。由于仓库是 Public，且两个 checkpoint 各约 391 MiB，模型权重保存在本机被 Git 忽略的 `weights/` 目录，没有随 PR 上传或发布。PR 合并不会自动把权重交给其他成员；需要组内选定模型文件共享渠道后，组长才能下载并实际运行。

本机产物及 SHA-256：

- 情感 checkpoint：`weights/emotion_balanced_b1_20261015`；模型权重 `51692b19961935a9edfcc4f47d6897cd79cdf4de528e845d015a07ffe00fd72e`；目录树 `399f4ce2d26c40b7877d0edead75d9410ba8f8ae55cb0e57b69f2469af9c21cd`。
- 联合 checkpoint：`weights/emotion_profile_balanced_b1_20261015`；编码器 `20376e0571baf16eff1036721471fb3b2224ce4e899dbaa5b612463a4f452fe2`；任务头 `dfd63c9475775a1b99c1d5f641c544690e1fd3c95cba477bb020e09320e44c9b`；目录树 `6b975dd0eb375eb34fa3d874f11f629eaada9974676551d7a563bb5db9ba5eb2`。

从项目根目录运行（要求 B1 v1 数据、`weights/emotion_baseline` 本地 checkpoint 和训练依赖已准备好）：

```bash
python scripts/finetune_emotion_balanced.py --device mps
python scripts/train_profile_heads.py \
  --device mps \
  --emotion-checkpoint weights/emotion_balanced_b1_20261015 \
  --emotion-report results/emotion_balanced_b1_20261015.json \
  --output-dir weights/emotion_profile_balanced_b1_20261015 \
  --report results/emotion_profile_balanced_b1_20261015.json
```

## 范围与下一步

- 本机 MPS 结果不代表赛事平台 GPU 训练，也没有验证官方基础镜像兼容性。
- 本地材料没有确认官方画像评分算法或验收阈值，因此不能承诺官方分数或最终通过。
- 情感 checkpoint 可交组长复核；画像头先保留为实验候选。若官方评测采用集合完全匹配，应继续以空画像作为对照，直到画像模型超过基线。
- 按团队 A→B→C→D→A 验收顺序，本 PR 交 D 复核代码与训练记录；画像质量仍需 B 按官方评分口径独立复核。
