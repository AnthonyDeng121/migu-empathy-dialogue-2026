# 数据审计报告

审计脚本按原始 JSONL 只读扫描，固定随机种子为 20261012；本次审计不抽样，也不生成或覆盖原始数据。

## 核心规模

| 指标 | 训练集 | 验证集 |
|---|---:|---:|
| 会话数 | 8338 | 900 |
| 预测样本数 | 18657 | 2354 |
| 平均轮数（消息数） | 14.14 | 13.036 |
| 最大轮数（消息数） | 50 | 50 |
| 平均上下文字符数 | 1473.445 | 1145.245 |
| 平均回复字符数 | 218.471 | 158.24 |

官方数据没有 `user_id` 字段，因此无法可靠统计用户数或检测同一用户跨集合泄漏；`conversation_id` 可用于会话级隔离。

## 交叉集合泄漏检查

- 相同 conversation_id：0。
- 完全相同规范化上下文：0。
- 完全相同最后一条用户消息：0。
- 完全相同标准回复：18（回复复用不等同于上下文泄漏，需单独解释）。

## 标签与质量

训练集情感分布：`{"anxiety": 6340, "relaxed": 2819, "helplessness": 1575, "fear": 1388, "sadness": 1085, "mixed": 945, "loneliness": 876, "anger": 855, "shame": 752, "gratitude": 695, "neutral": 654, "pride": 234, "joy": 171, "disgust": 125, "care": 79, "surprise": 64}`。

验证集情感分布：`{"anxiety": 751, "relaxed": 409, "mixed": 177, "helplessness": 142, "sadness": 131, "fear": 124, "gratitude": 118, "shame": 118, "loneliness": 115, "anger": 115, "neutral": 53, "pride": 40, "disgust": 21, "joy": 20, "care": 11, "surprise": 9}`。

训练集画像分布：`{"personality_traits": {"<empty>": 10492, "sensitive": 5165, "high_conscientiousness": 2281, "conservative": 1364, "introverted": 717, "assertive": 675, "agreeable": 361, "emotionally_stable": 336, "open": 59, "extroverted": 37, "casual": 27}, "interests": {"<empty>": 12919, "career_development": 1472, "social": 764, "study_exam": 746, "pets": 744, "games": 704, "sports_fitness": 625, "art_design": 341, "reading_writing": 316, "travel_outdoor": 144, "music": 133, "film_animation": 108, "programming_technology": 82}, "style": {"<empty>": 8731, "rational": 4418, "direct": 2603, "brief": 1166, "detailed": 1121, "high_emotional_expression": 945, "colloquial": 654, "indirect": 516, "formal": 128, "humorous": 81, "emoji_user": 27, "low_emotional_expression": 3}}`。

验证集画像分布：`{"personality_traits": {"<empty>": 1456, "sensitive": 487, "conservative": 253, "high_conscientiousness": 205, "assertive": 74, "agreeable": 53, "introverted": 52, "emotionally_stable": 9, "casual": 4, "open": 4, "extroverted": 2}, "interests": {"<empty>": 1763, "career_development": 128, "pets": 106, "study_exam": 95, "sports_fitness": 84, "games": 81, "reading_writing": 40, "art_design": 32, "social": 23, "travel_outdoor": 21, "music": 19, "film_animation": 13, "programming_technology": 1}, "style": {"<empty>": 1218, "rational": 585, "direct": 281, "detailed": 147, "brief": 120, "indirect": 59, "high_emotional_expression": 57, "colloquial": 54, "formal": 22, "humorous": 8}}`。

训练集结构错误：`{"prediction point does not reference adjacent user/assistant turns": 4}`；验证集结构错误：`{}`。
Unicode 替换字符：训练 0，验证 0；异常控制字符：训练 0，验证 0。

## 结论与处理约束

- 训练单位应展开为 prediction point，但划分必须以 conversation_id 为组，禁止把同一会话的不同预测点拆到训练和验证两侧。
- 空画像数组是合法标签，不应当作缺失值；画像指标应按三个维度分别比较后取平均。
- memory_refs 依赖未提供的 memory bank。测试输入也没有 memory bank 字段，当前 baseline 输出空数组；正式方案必须向官方确认隐藏评测是否另行提供记忆库。
- 高相似上下文检测当前给出完全一致与最后用户句完全一致两档。语义近重复需要在建模阶段使用向量或 MinHash 分块检查，且不能用于修改官方原始数据。
