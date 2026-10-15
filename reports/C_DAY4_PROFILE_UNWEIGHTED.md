# C Day 4：画像头取消正类加权对照

**结论：本轮优化完成，但画像候选 no-go。** 取消正类加权后，验证集宏平均 Micro-F1 略升，宏平均 exact match 从 Day3 的 0.4293 降至 0.3825，仍显著低于全空基线 0.6283；新 checkpoint 不替换 Day3 候选，也不发布。

## 实验设计与执行

- 假设：Day3 的正类重加权带来过多画像标签预测；只将正类权重指数从 `0.5` 改为 `0.0`（等价于不做正类重加权），其他配置、数据划分、模型和阈值选择规则保持固定。
- 设备：Apple M5 / PyTorch MPS / 32 GB 统一内存；Python 3.12.14、PyTorch 2.14.0、Transformers 5.17.0；平台 GPU 卡时为 0。
- B1 v1：训练 18,653 条，验证 2,354 条；训练/验证 conversation overlap 为 0。训练内 15% conversation 留出集用于阈值校准和 epoch 选择；验证指标每轮仅作诊断记录，不用于阈值、checkpoint 或早停选择。
- seed `20261015`，冻结 Day3 情感编码器与情感头；画像头训练 60 epochs，选中 epoch 52，最大长度 256，学习率 `0.01`。训练与特征提取共 374.817 秒。

## 同集结果

| 画像维度 | Day3 加权 exact / Micro-F1 | Day4 不加权 exact / Micro-F1 | 全空 exact |
|---|---:|---:|---:|
| personality_traits | 0.4044 / 0.2874 | 0.2783 / 0.2971 | 0.6185 |
| interests | 0.6075 / 0.1997 | 0.5319 / 0.1921 | 0.7489 |
| style | 0.2761 / 0.2703 | 0.3373 / 0.3153 | 0.5174 |
| 三维宏平均 | **0.4293 / 0.2525** | **0.3825 / 0.2682** | **0.6283 exact** |

这项单变量变更未达到画像采用门槛。虽然 Micro-F1 宏平均增加 0.0157，exact match 下降 0.0469；相对全空 exact 基线仍低 0.2458。保留 Day3 画像候选供后续评估，但画像部署暂用空画像基线。

## 可复现性与产物

训练命令（需本机 B1 v1 数据及 Day3 情感 checkpoint）：

```bash
python scripts/train_profile_heads.py \
  --train data/processed/b1_full_v1/classification_train.jsonl \
  --val data/processed/b1_full_v1/classification_val.jsonl \
  --emotion-checkpoint weights/emotion_balanced_b1_20261015 \
  --emotion-report results/emotion_balanced_b1_20261015.json \
  --output-dir weights/profile_unweighted_b1_20261015 \
  --report results/profile_unweighted_b1_20261015.json \
  --seed 20261015 --max-length 256 --batch-size 32 --epochs 60 \
  --learning-rate 0.01 --calibration-fraction 0.15 \
  --positive-weight-exponent 0 --device mps
```

- 完整训练历史和数据哈希：[机器可读报告](c_day4_profile_unweighted_metrics.json)。
- checkpoint 仅保存在本机忽略目录 `weights/profile_unweighted_b1_20261015`；总计 409,687,410 字节，目录树 SHA-256 `1b1bee3d8cf5175ea7d7c771015f3aa7ecd5455b740c3f93e2bce1ce98bfd4b0`。不进入 Git 或公开 Release，因为结果 no-go。
- 使用 PR #15 的 `EmotionProfileCheckpoint` 加载 checkpoint 后，对官方测试输入 3/3 条完成 MPS 推理，情感/画像标签结构校验通过。
- 旧 C1 CLI `scripts/predict_emotion_profile.py` 读取单数键 `profile_threshold`，而该 checkpoint 格式使用逐维 `profile_thresholds`，因此旧 CLI 不能加载它；本轮以 PR #15 中的新 checkpoint loader 完成加载冒烟。赛事 vLLM 镜像和官方评分仍未验证。

## 验收状态

Day4 C3 实验和证据已完成，候选结论为 no-go；等待 D 复核脚本与报告。当前最稳妥的画像输出仍是全空基线，不应把本轮结果表述为画像模型已达标。
