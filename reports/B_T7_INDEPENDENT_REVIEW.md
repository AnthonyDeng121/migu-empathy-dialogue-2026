# B-T7 Independent Review — Cycle2 final evaluation handoff

日期：2026-11-04（Asia/Shanghai）。**`T7_EVALUATION_COMPLETE`；`selection_stability=SELECTION_STABLE`。**

## 1. Scope

角色 B，独立数据评测。基于 T6 已有完整预测完成会话级 paired bootstrap、确定性 regression review、冻结切片稳定性及最终选型依据。全程 CPU-only；没有训练、GPU、生成新 prediction、改动原预测或 T5/T6 正式报告。未重新运行 T6 overall 评测流程：总体/切片指标复用 T6，逐样本评分仅用于 bootstrap 和排序，并核对其分解与 T6 一致。

比较方向始终为 **historical T4 longweight candidate − D4v2 baseline**。T4 是已否决的历史对照，不重新入围。固定选择仍 `KEEP_DAY5 / KEEP_D4V2 / OFF / empty / max_new_tokens=256`。

## 2. Inputs

- 固定分支：`feature/b-eval-tooling`；开工 HEAD `d9f8a75527191c7cc4ff090ec63fca3d11fa01fd`，包含 `d9f8a75`；开工工作区干净。
- 本地 `origin/main`：`75a8135f6bf424cb63347a68c2f1b00d68c7b69a`；未 fetch 或执行 Git 写操作。
- T6 正式报告：`reports/B_T6_INDEPENDENT_REVIEW.md`、`reports/b_t6_independent_review.json`；继承 `EVALUATION_COMPLETE_WITH_PROVENANCE_CAVEAT`。
- D4v2：`results/response_gen/d4v2_val_newcard.jsonl`；T4：`results/response_gen/t4_longweight_val_newcard.jsonl`。
- Gold：`data/processed/b1_full_v1/response_val.jsonl`；情感：同目录 `classification_val.jsonl`。
- 2354 samples / 900 conversations，sample ID 集合、顺序及 conversation 映射核验一致，duplicate/missing/extra/empty=0。T6 input/prediction 的 LF SHA 匹配，原有完整性结论保持 PASS。
- 复用 `src/evaluation/metrics.py`、`src/response/error_analysis.py` 和 B T1 history bucket。各输入 raw/LF SHA 与执行源码指纹见 JSON。

## 3. Conversation-level uncertainty methodology

仓库搜索未发现现有 bootstrap 工具，因此 CPU 本地实现放入忽略的验收目录。固定 seed **20261104**，**10,000 replicates**，Python random.Random / MT19937。每次从按 conversation_id 排序的 **900 个会话**中有放回抽取 900 个；一个会话内全部样本整体保留，重复抽中的会话整体重复。baseline/candidate 使用完全相同的会话抽样，三个指标也共享同一次抽样。

Estimand 保持 T6 的逐样本指标均值，不改成会话等权均值。各 replicate 分别累计两臂的指标分数和、样本数，再计算两臂均值差；因此大/小会话的样本数进入分子与分母。同一会话内样本没有被当成独立重采样单位。

逐样本 BLEU-4/ROUGE-L 严格调用原函数并缓存。原指标是逐样本字符分数的算术均值，故“完整会话的分数和 × 抽中次数 / 完整会话的样本数 × 抽中次数”与每次展开原始文本再评分数学等价（浮点舍入除外）；新增测试用不等长会话、混合长短样本及原文本逐次评分 oracle 验证。每 replicate 均重新汇总指标，未对 corpus BLEU、tokenization、normalization 作替换。

`>200` 使用同一整会话抽样后，按固定 gold len(response.strip())>200 留取相关 prediction points，再计算该 slice 均值；原始 528 条来自 **329 conversations**。10,000 次均无空 long denominator。

95% CI 为双侧 percentile（2.5% / 97.5%，(R−1)p 线性插值），未做 BCa 或多重比较校正。Fraction Δ>0 是 bootstrap 经验比例，不是贝叶斯后验、p-value 或正式显著性门槛；本分析条件于固定历史输出，不覆盖训练随机性、模型族搜索或官方隐藏集。

## 4. Overall uncertainty

| Metric delta (T4−D4v2) | Point estimate | Bootstrap mean | 95% CI | CI crosses 0 | Fraction Δ>0 |
| --- | --- | --- | --- | --- | --- |
| overall_bleu4_delta | 0.0008205711 | 0.0008246007 | [-0.0010325838, 0.0027033457] | True | 80.67% |
| overall_rouge_l_delta | 0.0002163853 | 0.0002250716 | [-0.0018331939, 0.0022952004] | True | 58.37% |

两个 CI 都跨 0，方向判断为 `NO_CLEAR_SIGNAL`。小幅正 point estimate 不足以稳定确认优于 D4v2；也不能据此证明等价或 D4v2 显著更优。

## 5. Long-response uncertainty

| Metric delta (T4−D4v2) | Point estimate | Bootstrap mean | 95% CI | CI crosses 0 | Fraction Δ>0 |
| --- | --- | --- | --- | --- | --- |
| long_gt200_rouge_l_delta | 0.0005281632 | 0.0005188453 | [-0.0020820204, 0.0030848398] | True | 65.11% |

冻结 gate：`>200 ROUGE-L delta >= +0.01`。实际 **0.0005281632**，primary **FAIL**；overall safety delta ≥−0.005，实际 **0.0002163853**，safety **PASS**。`adoption_verdict=NO_GO` 不变。10,000 次中达到 +0.01 的经验比例为 **0.00%**；这不是不可能性证明，也不替代正式 point-estimate gate。

Long slice W/T/L=273/7/248；329 会话的净改善/持平/退步=166/2/161。正向会话约一半，并非普遍改善。

Long gross positive delta sum=5.8653723895，gross negative magnitude=5.5865021937；收益前 20 会话贡献 gross positive 的 30.86%，退步前 20 会话贡献 gross negative 的 30.70%。改善并非只在单个会话，但正负相互抵消，净收益小。移除最大净改善会话后 long delta=0.0001131489；leave-one-conversation-out delta 范围 [0.0001131489, 0.0008371584]，均远低于 +0.01。

## 6. Regression review

排序规则先于展示固定：逐样本 ROUGE-L delta<0 按 delta 升序选 Top20 regression，delta>0 按 delta 降序选 Top20 improvement；相同 delta 按 sample_id 升序。所有 Top20 均已复核，未替换样本。Markdown 展示各 Top10 原文，JSON 保存各 Top20 全文及指标。

| Metric | Candidate better | Tie | Candidate worse |
| --- | --- | --- | --- |
| bleu4_char | 1026 | 280 | 1048 |
| rouge_l_char_f1 | 1038 | 300 | 1016 |

W/T/L 与 T6 完全一致；它们表示指标方向，不是人工质量胜负。900 会话净改善/持平/退步=444/36/420。移除最大净改善会话 val_conv_000555 后 overall ROUGE delta=-0.0000483942，微小总体正号可变；这是方向不稳定的敏感性证据，不改变未达 gate 的采用决策。

### Top regression/improvement observations

Top20 regression 中 candidate 更长 11 条、更短 9 条，不能把退步归结为统一长度方向。可见两种相反现象：短参考被扩展（val_conv_000047 / turn5 / offset1；val_conv_000243 / turn3 / offset1），以及具体情绪反映被缩成泛化“我听着”（val_conv_000483 / turn7 / offset1；val_conv_000777 / turn5 / offset1）。Top improvement 也包含缩短后更贴近 gold 的收尾（val_conv_000555 / turn5 / offset1），不能把收益解释为单纯增加长度。

以上是固定 gold/output 三元组的观察，没有读取原会话上下文或做盲评，不声称穷尽语义错误、幻觉、安全性或训练因果。格式诊断沿用 T6；与 gold 的重合度下降不自动等于共情质量下降。

### Top 10 regressions (deterministic)

<details>
<summary>Regression #1 — val_conv_000047 / turn 5 / offset 1；ROUGE Δ=-0.5092838196</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000047&quot;,5,1]</code><br><strong>conversation_id:</strong> val_conv_000047<br>
<strong>Gold length bucket:</strong> &lt;=6；<strong>emotion:</strong> relaxed；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.5946035575 / 0.2108445457 / -0.3837590118<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.9230769231 / 0.4137931034 / -0.5092838196</p>
<p><strong>Gold response</strong><br>好，我听着。</p>
<p><strong>Baseline D4v2 response</strong><br>好，那我听着。</p>
<p><strong>Candidate historical T4 response</strong><br>好，那我们就先让脑子歇会儿。你慢慢说，我听着。</p>
</details>

<details>
<summary>Regression #2 — val_conv_000622 / turn 27 / offset 2；ROUGE Δ=-0.4621347645</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000622&quot;,27,2]</code><br><strong>conversation_id:</strong> val_conv_000622<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> fear；<strong>history:</strong> long_16_plus<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.4846243022 / 0.1219737941 / -0.3626505081<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.7441860465 / 0.2820512821 / -0.4621347645</p>
<p><strong>Gold response</strong><br>好，我先不提方案。你继续把这份担心说完，我不打断。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我先不提方案。你继续说，我听着。</p>
<p><strong>Candidate historical T4 response</strong><br>我明白，你并不是在要求一个立刻解决的办法，而是在表达这种反复出现的担忧和睡眠受影响。你继续说，我先不分析。</p>
</details>

<details>
<summary>Regression #3 — val_conv_000768 / turn 5 / offset 2；ROUGE Δ=-0.3949152542</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000768&quot;,5,2]</code><br><strong>conversation_id:</strong> val_conv_000768<br>
<strong>Gold length bucket:</strong> 7-20；<strong>emotion:</strong> neutral；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.4249456196 / 0.1234544392 / -0.3014911804<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.7000000000 / 0.3050847458 / -0.3949152542</p>
<p><strong>Gold response</strong><br>好，今晚就到这儿。你自己慢慢想就行。</p>
<p><strong>Baseline D4v2 response</strong><br>好，先听到这儿就够了。今晚你自己慢慢想就好。</p>
<p><strong>Candidate historical T4 response</strong><br>好，先到这里。你已经把“白天那些人”和“晚上那一下”分开了，今晚就停在这儿就够了。</p>
</details>

<details>
<summary>Regression #4 — val_conv_000483 / turn 7 / offset 1；ROUGE Δ=-0.3587412587</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000483&quot;,7,1]</code><br><strong>conversation_id:</strong> val_conv_000483<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> sadness；<strong>history:</strong> medium_6_15<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.2646503844 / 0.0108705967 / -0.2537797876<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.5769230769 / 0.2181818182 / -0.3587412587</p>
<p><strong>Gold response</strong><br>好，我先听你说，不分析。你已经多花了钱，却一直自己扛着，现在预算又变，让这份难受更重了。</p>
<p><strong>Baseline D4v2 response</strong><br>嗯，我先听着，不分析。你已经为此多花了这笔钱，而且它一直被独自压着；现在预算变化带来的不适，也会比单纯改变行程更重一些。</p>
<p><strong>Candidate historical T4 response</strong><br>嗯，我听着。你继续说。</p>
</details>

<details>
<summary>Regression #5 — val_conv_000895 / turn 9 / offset 1；ROUGE Δ=-0.3182978723</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000895&quot;,9,1]</code><br><strong>conversation_id:</strong> val_conv_000895<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> mixed；<strong>history:</strong> medium_6_15<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.3463339194 / 0.1357962133 / -0.2105377061<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.6382978723 / 0.3200000000 / -0.3182978723</p>
<p><strong>Gold response</strong><br>好，我听着。你把这层说清楚就行，我不打断，也不分析、不给主意。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我听着。你把这层说清楚就好。</p>
<p><strong>Candidate historical T4 response</strong><br>嗯，这层你提到了，我就听着。你不用现在就把整套方案讲完，也不用马上决定要不要把它说出来。</p>
</details>

<details>
<summary>Regression #6 — val_conv_000740 / turn 19 / offset 3；ROUGE Δ=-0.2874180083</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000740&quot;,19,3]</code><br><strong>conversation_id:</strong> val_conv_000740<br>
<strong>Gold length bucket:</strong> 7-20；<strong>emotion:</strong> relaxed；<strong>history:</strong> long_16_plus<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.4438335184 / 0.2516441112 / -0.1921894073<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.6976744186 / 0.4102564103 / -0.2874180083</p>
<p><strong>Gold response</strong><br>好，这周就按这个来。回头有新情况再说。</p>
<p><strong>Baseline D4v2 response</strong><br>好，这周先按你定的来就行。回头有新情况再来就行。</p>
<p><strong>Candidate historical T4 response</strong><br>好，这周先按例会听完再说。今天就到这儿。</p>
</details>

<details>
<summary>Regression #7 — val_conv_000400 / turn 3 / offset 2；ROUGE Δ=-0.2506393862</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000400&quot;,3,2]</code><br><strong>conversation_id:</strong> val_conv_000400<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> shame；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.2912867627 / 0.0422039822 / -0.2490827805<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.4565217391 / 0.2058823529 / -0.2506393862</p>
<p><strong>Gold response</strong><br>好，我先不分析。你能把这点别扭说出来，已经很不容易了；那些准备并不会因为延期就变成笑话。</p>
<p><strong>Baseline D4v2 response</strong><br>嗯，我听着。你不用继续展开，也不用现在就证明自己做得对错；能把这种别扭说出来，就已经很不容易了。</p>
<p><strong>Candidate historical T4 response</strong><br>嗯，我听着。你不用继续展开，也不用马上想下一步。</p>
</details>

<details>
<summary>Regression #8 — val_conv_000462 / turn 3 / offset 1；ROUGE Δ=-0.2491075982</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000462&quot;,3,1]</code><br><strong>conversation_id:</strong> val_conv_000462<br>
<strong>Gold length bucket:</strong> 51-100；<strong>emotion:</strong> anxiety；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.1208473208 / 0.0088815256 / -0.1119657952<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.4189189189 / 0.1698113208 / -0.2491075982</p>
<p><strong>Gold response</strong><br>好，我先不提怎么做。你怕的不是那一次没跑，而是它像一个证据，让你开始怀疑身体是不是已经悄悄变了，再也回不到以前。这个念头反复出现，确实会让一次小插曲变得很沉。你继续说。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我不说办法了。你害怕的可能不只是这次漏跑，而是担心身体和状态真的会回到以前那种稳定的状态里。这种反复出现的恐惧确实很消耗人。</p>
<p><strong>Candidate historical T4 response</strong><br>好，我不说办法了。你先把这件事说完，我听着。</p>
</details>

<details>
<summary>Regression #9 — val_conv_000574 / turn 3 / offset 1；ROUGE Δ=-0.2465072224</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000574&quot;,3,1]</code><br><strong>conversation_id:</strong> val_conv_000574<br>
<strong>Gold length bucket:</strong> 101-200；<strong>emotion:</strong> mixed；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.3179015058 / 0.0778191443 / -0.2400823615<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.4660194175 / 0.2195121951 / -0.2465072224</p>
<p><strong>Gold response</strong><br>好，你先把情况说完，我不急着帮你排计划。听起来这已经从“答辩重做很烦”牵扯到了要不要放弃考试，以及后面那个具体安排；但你白天还能照常复习、也还不甘心放弃，说明你心里其实有几股力量在拉扯。你继续说，我先听着。</p>
<p><strong>Baseline D4v2 response</strong><br>好，你先把情况说完，我不急着给建议。听起来你现在被两件事牵住了：答辩重做让你烦躁，而那个“要不要放弃”的念头也已经影响到了后续安排。白天还能复习说明你并不想就这样放弃，只是这件事压得你很难受。你可以继续说。</p>
<p><strong>Candidate historical T4 response</strong><br>你不是不想处理这件事，而是它已经让你同时面对两次选择：答辩重做和考试是否继续。白天还能复习说明你并不想放弃，只是被这种两难牵住了。<br><br>可以把这两件事分开看：答辩重做是当前必须完成的任务，考试则需要根据你现在的状态重新判断。如果暂时无法决定，也可以先保留两个选项，不必立刻逼自己做出最终决定。</p>
</details>

<details>
<summary>Regression #10 — val_conv_000686 / turn 5 / offset 1；ROUGE Δ=-0.2464776778</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000686&quot;,5,1]</code><br><strong>conversation_id:</strong> val_conv_000686<br>
<strong>Gold length bucket:</strong> 51-100；<strong>emotion:</strong> fear；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.2724671074 / 0.0416645032 / -0.2308026042<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.5255474453 / 0.2790697674 / -0.2464776778</p>
<p><strong>Gold response</strong><br>好，你先说，我不急着分析。你刚想起前天它就有点不吃东西，今天又明显没精神，所以现在特别害怕，担心是不是已经耽误得很严重了。你继续说。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我先不分析，也不急着给结论。你刚才提到前天就开始不吃东西，今天也明显没精神，这些信息我都听见了。你现在最害怕的是它已经到了很严重的程度吗？</p>
<p><strong>Candidate historical T4 response</strong><br>好，你继续说，我不分析，也不急着给结论。</p>
</details>

### Top 10 improvements (deterministic)

<details>
<summary>Improvement #1 — val_conv_000555 / turn 5 / offset 1；ROUGE Δ=0.4631765750</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000555&quot;,5,1]</code><br><strong>conversation_id:</strong> val_conv_000555<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> sadness；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.2457432915 / 0.7432974655 / 0.4975541741<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.4347826087 / 0.8979591837 / 0.4631765750</p>
<p><strong>Gold response</strong><br>好，我先不拆解，也不急着给建议。你继续说，我听着。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我不拆解，也不急着给建议。你说得对，它不只是几周没做完，也让你开始怀疑自己是否还真的喜欢烘焙。那个不敢承认的可能性，你继续说就行。</p>
<p><strong>Candidate historical T4 response</strong><br>好，我不拆解，也不急着给结论。你继续说，我听着。</p>
</details>

<details>
<summary>Improvement #2 — val_conv_000381 / turn 19 / offset 4；ROUGE Δ=0.3591417910</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000381&quot;,19,4]</code><br><strong>conversation_id:</strong> val_conv_000381<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> relaxed；<strong>history:</strong> long_16_plus<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.2010159547 / 0.4742075809 / 0.2731916262<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.3283582090 / 0.6875000000 / 0.3591417910</p>
<p><strong>Gold response</strong><br>好，今晚先不用做决定。能把这些压着的话说出来，已经很不容易了。先到这里，晚安。</p>
<p><strong>Baseline D4v2 response</strong><br>好，今晚先到这里。谢谢你把这些告诉我，希望你能休息一下。</p>
<p><strong>Candidate historical T4 response</strong><br>好，今晚先到这里。能把这些说出来，已经很不容易了。</p>
</details>

<details>
<summary>Improvement #3 — val_conv_000884 / turn 25 / offset 4；ROUGE Δ=0.3116829218</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000884&quot;,25,4]</code><br><strong>conversation_id:</strong> val_conv_000884<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> relaxed；<strong>history:</strong> long_16_plus<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.2828760246 / 0.2722927822 / -0.0105832423<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.2784810127 / 0.5901639344 / 0.3116829218</p>
<p><strong>Gold response</strong><br>好，先说到这里。你已经分清了卡住你的环节，今晚先不盯时间，按自己的节奏休息就好。</p>
<p><strong>Baseline D4v2 response</strong><br>好，今晚先不盯时间，把注意力放回当下。谢谢你愿意把这些告诉我，先到这里也可以。</p>
<p><strong>Candidate historical T4 response</strong><br>好，今晚先不盯时间，按你自己的节奏来就行。</p>
</details>

<details>
<summary>Improvement #4 — val_conv_000855 / turn 5 / offset 2；ROUGE Δ=0.2405797101</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000855&quot;,5,2]</code><br><strong>conversation_id:</strong> val_conv_000855<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> anxiety；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.1711982719 / 0.5918268619 / 0.4206285900<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.4927536232 / 0.7333333333 / 0.2405797101</p>
<p><strong>Gold response</strong><br>好，我先听着，不给建议。你开始怀疑自己是不是根本不适合这个方向了；你继续说，我不会急着替你下结论。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我先听着，不给建议。你说，我在这里。</p>
<p><strong>Candidate historical T4 response</strong><br>好，我先听着，不给建议。你说：“我已经开始怀疑，自己是不是根本不适合这个方向了。”</p>
</details>

<details>
<summary>Improvement #5 — val_conv_000175 / turn 3 / offset 1；ROUGE Δ=0.2380952381</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000175&quot;,3,1]</code><br><strong>conversation_id:</strong> val_conv_000175<br>
<strong>Gold length bucket:</strong> 7-20；<strong>emotion:</strong> anger；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.1326475917 / 0.3092040653 / 0.1765564737<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.3333333333 / 0.5714285714 / 0.2380952381</p>
<p><strong>Gold response</strong><br>好，我听着，你继续说。</p>
<p><strong>Baseline D4v2 response</strong><br>好，你先说。我不打断，也不急着给建议。</p>
<p><strong>Candidate historical T4 response</strong><br>好，你先说。我听着。</p>
</details>

<details>
<summary>Improvement #6 — val_conv_000573 / turn 15 / offset 4；ROUGE Δ=0.2364927447</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000573&quot;,15,4]</code><br><strong>conversation_id:</strong> val_conv_000573<br>
<strong>Gold length bucket:</strong> 51-100；<strong>emotion:</strong> gratitude；<strong>history:</strong> medium_6_15<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.0511143866 / 0.1515125517 / 0.1003981651<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.2025316456 / 0.4390243902 / 0.2364927447</p>
<p><strong>Gold response</strong><br>去洗澡吧，先让自己放松下来。能把这件事说出来、也迈出联系朋友那一步，已经很不容易了。今晚先照顾好自己。</p>
<p><strong>Baseline D4v2 response</strong><br>不客气。先去做点让自己舒服的事吧，不必现在就把答案想完。</p>
<p><strong>Candidate historical T4 response</strong><br>好，先去做点让自己舒服的事。能把这份堵感说出来，已经很重要了。</p>
</details>

<details>
<summary>Improvement #7 — val_conv_000490 / turn 13 / offset 2；ROUGE Δ=0.2345679012</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000490&quot;,13,2]</code><br><strong>conversation_id:</strong> val_conv_000490<br>
<strong>Gold length bucket:</strong> 51-100；<strong>emotion:</strong> fear；<strong>history:</strong> medium_6_15<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.0245247503 / 0.2727532930 / 0.2482285427<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.1851851852 / 0.4197530864 / 0.2345679012</p>
<p><strong>Gold response</strong><br>明白，你害怕的不是一个明确的拒绝，而是自己连一次被认真对待、被坦诚回应的机会都没有。那种被含糊带过、像不值得说明白的感觉，可能比拒绝本身更刺痛。你继续说，我先不提供沟通方式。</p>
<p><strong>Baseline D4v2 response</strong><br>好，我先不设计沟通方式。你继续说，我听着。</p>
<p><strong>Candidate historical T4 response</strong><br>我明白了，你害怕的不是一次拒绝，而是连基本的重视都没有——你期待的是一句明确的回应，而不是被当作可有可无的存在。你继续说，我先不分析，也不急着给建议。</p>
</details>

<details>
<summary>Improvement #8 — val_conv_000747 / turn 11 / offset 2；ROUGE Δ=0.2234042553</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000747&quot;,11,2]</code><br><strong>conversation_id:</strong> val_conv_000747<br>
<strong>Gold length bucket:</strong> 21-50；<strong>emotion:</strong> anxiety；<strong>history:</strong> medium_6_15<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.1203319738 / 0.5076857807 / 0.3873538069<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.5000000000 / 0.7234042553 / 0.2234042553</p>
<p><strong>Gold response</strong><br>好，先说到这儿就行。你把这层拧着的感觉说出来了。</p>
<p><strong>Baseline D4v2 response</strong><br>好，先说到这儿。</p>
<p><strong>Candidate historical T4 response</strong><br>好，先说到这儿。你已经把这层拧着的东西摊开了。</p>
</details>

<details>
<summary>Improvement #9 — val_conv_000424 / turn 9 / offset 2；ROUGE Δ=0.2195900711</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000424&quot;,9,2]</code><br><strong>conversation_id:</strong> val_conv_000424<br>
<strong>Gold length bucket:</strong> 51-100；<strong>emotion:</strong> fear；<strong>history:</strong> medium_6_15<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.0806390471 / 0.2853158224 / 0.2046767753<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.2465753425 / 0.4661654135 / 0.2195900711</p>
<p><strong>Gold response</strong><br>我听到了。你害怕的不是一次尴尬，而是失去这段朋友关系；主动发消息也像是在把自己置于“可能被拒绝”的位置，所以会更不敢动。你继续说，我先不分析。</p>
<p><strong>Baseline D4v2 response</strong><br>我明白了，真正让你害怕的，是你担心这次开口会变成“确认是否被排斥”的机会。你不想先发出消息，是因为害怕一旦表达，对方就会立刻否定你，甚至把你留在外面。</p>
<p><strong>Candidate historical T4 response</strong><br>我明白了，你害怕的不是那一句玩笑，而是它让你开始怀疑自己是否会被真正排斥，甚至不敢迈出主动联系的一步。你继续说，我先不分析。</p>
</details>

<details>
<summary>Improvement #10 — val_conv_000255 / turn 3 / offset 1；ROUGE Δ=0.2192407508</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000255&quot;,3,1]</code><br><strong>conversation_id:</strong> val_conv_000255<br>
<strong>Gold length bucket:</strong> &gt;200；<strong>emotion:</strong> loneliness；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.0000000000 / 0.0869064643 / 0.0869064643<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.0126984127 / 0.2319391635 / 0.2192407508</p>
<p><strong>Gold response</strong><br>门一关，那个安静确实容易把人一下子拽进那种空落落的感觉里。<br><br>刚把压了三个月的大事啃下来，紧绷的弦突然松了，那种“没了着落”的空虚感其实挺常见的。你刚才说的那份底气，和此刻这种面对空房间的恐惧，其实是两件事。前者是你对自己能力的确认，后者更像是长期一个人扛事、独自消化情绪后，身体和心里积攒的疲惫突然找上门来了。<br><br>这种“一个人扛完所有事然后面对空荡荡的房间”的状态，确实会让人觉得孤单，甚至让人害怕它变成常态。但你现在能感觉到它，并且把它说出来，这本身就是在给那个空荡荡的房间添点人气。<br><br>不用急着去消化下周的面试，也不用逼自己马上振作起来。就让自己在这个安静里待一会儿，感受一下那种空落落的，不用急着给它贴标签，也不用急着找解法。</p>
<p><strong>Baseline D4v2 response</strong><br>我在听。</p>
<p><strong>Candidate historical T4 response</strong><br>那种“交出报告后就彻底没了着落”的感觉，确实比单纯的忙碌更让人窒息。明明手里攥着东西，心里却像掉进了黑洞，这种悬空感最耗神。<br><br>你提到的“害怕变成常态”这个念头，听起来像是在预演一种长期孤独的生活模式。这种恐惧往往比事情本身更消耗人，因为它让你意识到：如果现在不解决它，未来可能就会这样一直下去。<br><br>既然你现在不想被问，那就先不用急着给下周面试找理由。你愿意多说说当时那种“空落落”的具体感受吗？是觉得时间不够用，还是某种情绪上的疏离感？</p>
</details>

### >200 maximum regression / improvement

最大 long 退步不是近空回复：candidate281 字 vs baseline306 字，主要是表达结构/主题重合度变化。最大 long 改善修复 baseline“我在听。”4 字近空输出，candidate219 字；它是真实局部改善，但不足以覆盖整个长回复 slice。

<details>
<summary>Long maximum regression — val_conv_000373 / turn 5 / offset 2；ROUGE Δ=-0.1210481800</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000373&quot;,5,2]</code><br><strong>conversation_id:</strong> val_conv_000373<br>
<strong>Gold length bucket:</strong> &gt;200；<strong>emotion:</strong> anxiety；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.1185158146 / 0.1096413660 / -0.0088744485<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.2813688213 / 0.1603206413 / -0.1210481800</p>
<p><strong>Gold response</strong><br>这种反复对照会让注意力不断扫描身体，细小变化也可能被放大；但“感觉哪里都不对”不等于病情正在朝家人当年的方向发展。<br><br>你可以先只做一次简短记录，分成两栏：<br><br>- **事实**：具体出现了什么、何时出现、是否加重、有没有客观检查结果。<br>- **担心**：你联想到的最坏情况，以及目前支持它的证据。<br><br>暂时减少反复搜索，把观察限定为每天固定一次；如果症状持续、明显加重或影响正常生活，安排正规就诊，比继续自行对照更可靠。若出现突然严重恶化、呼吸困难、胸痛、晕厥或意识异常，应及时就医。</p>
<p><strong>Baseline D4v2 response</strong><br>这很能理解——你不是在否定自己，而是在用过去一次经历重新评估现在的情况。对照不会让事情变得更清楚，反而容易把“可能”变成“必须如此”。<br><br>可以先做两件事：<br><br>1. 把“对照”改成“记录”。写下**已经发生的**：症状何时开始、频率、程度、是否有变化、有没有诱发因素；**未发生的**：是否出现新的严重症状、是否影响日常生活、是否需要紧急就医。  <br>2. 每天只记一件最重要的事：今天症状有没有变化、是否影响睡眠或日常活动。其余都归类为“不确定”。<br><br>同时，把“我是不是拖到了那个阶段”的担心，单独写下来，不把它当作诊断。若症状突然加重、出现呼吸困难、胸痛、晕厥、意识异常、持续高热或大量出血，立即联系急诊或就医。</p>
<p><strong>Candidate historical T4 response</strong><br>这很常见：一个人经历过一次严重后果，大脑会自动放大类似的警报，即使现在没有证据也很难停下来。你不必立刻得出“一定是”或“绝对没事”的结论，可以先做三件事：<br><br>1. 把“一定会发生”改成“可能会发生”。例如：“我担心自己也会住院，不是已经确定。”  <br>2. 检查是否有新的、明显的危险信号：呼吸困难、胸痛、意识异常、持续高烧、大量出血、剧烈疼痛、无法平躺。如果有，立即联系急诊或就医。  <br>3. 将注意力从“会不会发生”拉回到“这几天症状如何变化”，并记下具体细节。若症状持续加重，及时就医即可。<br><br>你不需要现在就解决所有疑虑，先让这些念头不再替你决定下一步就好。</p>
</details>

<details>
<summary>Long maximum improvement — val_conv_000255 / turn 3 / offset 1；ROUGE Δ=0.2192407508</summary>

<p><strong>sample_id:</strong> <code>pp:[&quot;val_conv_000255&quot;,3,1]</code><br><strong>conversation_id:</strong> val_conv_000255<br>
<strong>Gold length bucket:</strong> &gt;200；<strong>emotion:</strong> loneliness；<strong>history:</strong> short_1_5<br>
<strong>BLEU-4 baseline / candidate / delta:</strong> 0.0000000000 / 0.0869064643 / 0.0869064643<br>
<strong>ROUGE-L baseline / candidate / delta:</strong> 0.0126984127 / 0.2319391635 / 0.2192407508</p>
<p><strong>Gold response</strong><br>门一关，那个安静确实容易把人一下子拽进那种空落落的感觉里。<br><br>刚把压了三个月的大事啃下来，紧绷的弦突然松了，那种“没了着落”的空虚感其实挺常见的。你刚才说的那份底气，和此刻这种面对空房间的恐惧，其实是两件事。前者是你对自己能力的确认，后者更像是长期一个人扛事、独自消化情绪后，身体和心里积攒的疲惫突然找上门来了。<br><br>这种“一个人扛完所有事然后面对空荡荡的房间”的状态，确实会让人觉得孤单，甚至让人害怕它变成常态。但你现在能感觉到它，并且把它说出来，这本身就是在给那个空荡荡的房间添点人气。<br><br>不用急着去消化下周的面试，也不用逼自己马上振作起来。就让自己在这个安静里待一会儿，感受一下那种空落落的，不用急着给它贴标签，也不用急着找解法。</p>
<p><strong>Baseline D4v2 response</strong><br>我在听。</p>
<p><strong>Candidate historical T4 response</strong><br>那种“交出报告后就彻底没了着落”的感觉，确实比单纯的忙碌更让人窒息。明明手里攥着东西，心里却像掉进了黑洞，这种悬空感最耗神。<br><br>你提到的“害怕变成常态”这个念头，听起来像是在预演一种长期孤独的生活模式。这种恐惧往往比事情本身更消耗人，因为它让你意识到：如果现在不解决它，未来可能就会这样一直下去。<br><br>既然你现在不想被问，那就先不用急着给下周面试找理由。你愿意多说说当时那种“空落落”的具体感受吗？是觉得时间不够用，还是某种情绪上的疏离感？</p>
</details>

## 7. Slice stability

完全复用 T6 gold/history/emotion/length 桶；下面 baseline/candidate 分数、delta 均直接来自 T6，仅 paired signs 按逐样本核验。W/T/L 方向始终 candidate better/tie/worse。切片为描述性分析，没有为事后挑选切片建立显著性声明。

### History slices

| Slice | n | Conversations | Metric | D4v2 | T4 | Δ | W/T/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short_1_5 | 903 | 672 | bleu4_char | 0.1057172389 | 0.1058208157 | 0.0001035769 | 392/98/413 |
| short_1_5 | 903 | 672 | rouge_l_char_f1 | 0.2511026872 | 0.2508667433 | -0.0002359440 | 402/101/400 |
| medium_6_15 | 1098 | 572 | bleu4_char | 0.1133785254 | 0.1156507932 | 0.0022722677 | 486/135/477 |
| medium_6_15 | 1098 | 572 | rouge_l_char_f1 | 0.2632454279 | 0.2640435063 | 0.0007980784 | 483/148/467 |
| long_16_plus | 353 | 185 | bleu4_char | 0.1253868901 | 0.1235261095 | -0.0018607807 | 148/47/158 |
| long_16_plus | 353 | 185 | rouge_l_char_f1 | 0.2747668612 | 0.2743309920 | -0.0004358692 | 153/51/149 |

### Emotion slices

| Slice | n | Conversations | Metric | D4v2 | T4 | Δ | W/T/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anger | 115 | 99 | bleu4_char | 0.1059483254 | 0.1107713596 | 0.0048230343 | 53/21/41 |
| anger | 115 | 99 | rouge_l_char_f1 | 0.2519707575 | 0.2545207380 | 0.0025499805 | 50/22/43 |
| anxiety | 751 | 504 | bleu4_char | 0.0894756930 | 0.0908844841 | 0.0014087910 | 349/58/344 |
| anxiety | 751 | 504 | rouge_l_char_f1 | 0.2264855107 | 0.2269060511 | 0.0004205404 | 352/59/340 |
| care | 11 | 10 | bleu4_char | 0.0864745651 | 0.0849121316 | -0.0015624335 | 4/2/5 |
| care | 11 | 10 | rouge_l_char_f1 | 0.2277312435 | 0.2379172640 | 0.0101860204 | 7/2/2 |
| disgust | 21 | 19 | bleu4_char | 0.0981922089 | 0.0839627735 | -0.0142294355 | 7/1/13 |
| disgust | 21 | 19 | rouge_l_char_f1 | 0.2370288202 | 0.2206419247 | -0.0163868955 | 7/1/13 |
| fear | 124 | 112 | bleu4_char | 0.1033324994 | 0.1026105937 | -0.0007219057 | 51/12/61 |
| fear | 124 | 112 | rouge_l_char_f1 | 0.2425292675 | 0.2397445273 | -0.0027847402 | 53/14/57 |
| gratitude | 118 | 105 | bleu4_char | 0.1475665995 | 0.1493968980 | 0.0018302985 | 39/33/46 |
| gratitude | 118 | 105 | rouge_l_char_f1 | 0.3238983073 | 0.3254276087 | 0.0015293015 | 39/33/46 |
| helplessness | 142 | 125 | bleu4_char | 0.0984584339 | 0.1035535981 | 0.0050951642 | 69/12/61 |
| helplessness | 142 | 125 | rouge_l_char_f1 | 0.2458295476 | 0.2512721508 | 0.0054426032 | 69/12/61 |
| joy | 20 | 19 | bleu4_char | 0.0765491591 | 0.0797902312 | 0.0032410721 | 9/0/11 |
| joy | 20 | 19 | rouge_l_char_f1 | 0.2279561718 | 0.2271278851 | -0.0008282866 | 10/0/10 |
| loneliness | 115 | 90 | bleu4_char | 0.0974233318 | 0.0918868145 | -0.0055365173 | 46/6/63 |
| loneliness | 115 | 90 | rouge_l_char_f1 | 0.2337563113 | 0.2357170192 | 0.0019607079 | 56/7/52 |
| mixed | 177 | 161 | bleu4_char | 0.1311272906 | 0.1269158489 | -0.0042114417 | 72/20/85 |
| mixed | 177 | 161 | rouge_l_char_f1 | 0.2801190072 | 0.2755233202 | -0.0045956870 | 74/22/81 |
| neutral | 53 | 53 | bleu4_char | 0.1783627211 | 0.1834868228 | 0.0051241018 | 22/19/12 |
| neutral | 53 | 53 | rouge_l_char_f1 | 0.3561653554 | 0.3597700055 | 0.0036046500 | 19/20/14 |
| pride | 40 | 32 | bleu4_char | 0.1198891352 | 0.1127897417 | -0.0070993936 | 13/3/24 |
| pride | 40 | 32 | rouge_l_char_f1 | 0.2849178109 | 0.2640746339 | -0.0208431770 | 10/4/26 |
| relaxed | 409 | 353 | bleu4_char | 0.1457143875 | 0.1483230687 | 0.0026086812 | 177/69/163 |
| relaxed | 409 | 353 | rouge_l_char_f1 | 0.3125380637 | 0.3139343696 | 0.0013963059 | 170/77/162 |
| sadness | 131 | 111 | bleu4_char | 0.1151640472 | 0.1182826850 | 0.0031186379 | 55/13/63 |
| sadness | 131 | 111 | rouge_l_char_f1 | 0.2554502816 | 0.2582081937 | 0.0027579121 | 61/15/55 |
| shame | 118 | 90 | bleu4_char | 0.1019989516 | 0.0979138358 | -0.0040851158 | 54/11/53 |
| shame | 118 | 90 | rouge_l_char_f1 | 0.2425381367 | 0.2409920690 | -0.0015460677 | 57/12/49 |
| surprise | 9 | 8 | bleu4_char | 0.0778418571 | 0.0913150818 | 0.0134732246 | 6/0/3 |
| surprise | 9 | 8 | rouge_l_char_f1 | 0.2025982003 | 0.2050024627 | 0.0024042624 | 4/0/5 |

### Gold response-length buckets

| Slice | n | Conversations | Metric | D4v2 | T4 | Δ | W/T/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <=6 | 15 | 15 | bleu4_char | 0.3496122164 | 0.3394377676 | -0.0101744488 | 3/10/2 |
| <=6 | 15 | 15 | rouge_l_char_f1 | 0.5419432573 | 0.5334710893 | -0.0084721680 | 4/10/1 |
| 7-20 | 65 | 64 | bleu4_char | 0.3036907459 | 0.3080655050 | 0.0043747592 | 14/39/12 |
| 7-20 | 65 | 64 | rouge_l_char_f1 | 0.5157584800 | 0.5141290169 | -0.0016294631 | 15/39/11 |
| 21-50 | 298 | 258 | bleu4_char | 0.1976534078 | 0.2016739761 | 0.0040205683 | 98/107/93 |
| 21-50 | 298 | 258 | rouge_l_char_f1 | 0.3965123564 | 0.4009562321 | 0.0044438757 | 91/108/99 |
| 51-100 | 822 | 492 | bleu4_char | 0.1072004072 | 0.1060580724 | -0.0011423347 | 334/93/395 |
| 51-100 | 822 | 492 | rouge_l_char_f1 | 0.2628705098 | 0.2605185070 | -0.0023520028 | 354/107/361 |
| 101-200 | 626 | 458 | bleu4_char | 0.0817838232 | 0.0826187726 | 0.0008349494 | 298/25/303 |
| 101-200 | 626 | 458 | rouge_l_char_f1 | 0.2139010251 | 0.2156143952 | 0.0017133701 | 301/29/296 |
| >200 | 528 | 329 | bleu4_char | 0.0776773761 | 0.0796055450 | 0.0019281689 | 279/6/243 |
| >200 | 528 | 329 | rouge_l_char_f1 | 0.1950495746 | 0.1955777379 | 0.0005281632 | 273/7/248 |

主要集中退步观察：pride n40，ROUGE Δ−0.0208431770、W/T/L10/4/26；disgust n21，Δ−0.0163868955、7/1/13；mixed n177，Δ−0.0045956870、74/22/81。它们值得下周期预注册复核，不能宣称经多重检验确认的类群效应。

集中改善观察：helplessness n142，Δ+0.0054426032、69/12/61；care n11，Δ+0.0101860204、7/2/2，但样本极小。历史中桶微升、短/长桶微降；gold21–50、101–200 微升，51–100 下降。没有一致覆盖主要 slices 的改善，不能以 care 或任一有利桶推翻预注册 long gate 或 shortlist。

## 8. Decision stability

- `NO_CLEAR_SIGNAL`：overall BLEU/ROUGE 与 long ROUGE 的 95% CI 都跨 0；含义为效果方向缺乏稳定证据。
- `NO_GO`：long point estimate 未达到预注册 +0.01；这是采用资格结论，bootstrap 不替代它。
- `KEEP_D4V2`：固定工程选型继续保留历史 D4v2。
- **`SELECTION_STABLE`**：gate 未达、无指标错误、各 slice 改善/退步抵消、long leave-one-out 仍远未过线，共同支持维持选择。选择稳定不表示两臂等价，也不表示 D4v2 在所有样本或指标上严格更好。

## 9. Final Selection Basis

**`KEEP_DAY5 / KEEP_D4V2 / OFF / empty / 256`**。

- Emotion：沿用 T6 已由 B 核对的 Day5 固定逐条标签；C T4 候选缺逐条 prediction 的边界保持，不能宣称该候选已由 B 独立胜出或失败。
- Response：B 的固定文本 T6 评测和本轮整会话不确定性/退步复核均不支持采用 T4；主 gate FAIL、安全 gate PASS，保留 D4v2。
- Condition：沿用 T5 固定 OFF 和既有条件筛选结论；T7 未新增条件实验或推理。
- Profile：沿用 empty 部署选择；画像 v1 的内部候选边界保持，不把画像研究结果写成联合部署提升。
- 256：沿用 manifest/report 与 fixed lock；本轮没有重新验证真实模型生成行为。

B 本轮负责的是给定归档预测的统计、配对复核及选择依据，不为 C/D 尚未独立验证的资产、冻结交接或官方效果背书。

## 10. Follow-up Questions

1. 长回复加权为何在会话内外产生相互抵消的收益/退步，导致 long gain 远低于 +0.01？
2. 下一项预注册改动能否同时减少短参考过度展开、具体情绪反映过度缩短，以及 pride/disgust/mixed 的退步，而不损失其他群体？
3. C 能否交付固定候选逐条 prediction/probability，解除独立跨模型联合判断的证据限制？
4. 下一次真实运行能否在生成时一并归档 effective generation_config、tokenizer/template 文件身份及针对性的 base preflight，支持严格复现？

这些是下一周期问题，不要求重做 T1–T6、不要求恢复已销毁实例或追加本轮全量生成。

## 11. Provenance caveats

继承 T6：prediction integrity=PASS；adapter identity 为已有独立核验/部分确认事实，base 历史 preflight available；完整 base tree 在现有证据中无法恢复；repetition_penalty=UNKNOWN；tokenizer/template historical binding=partially confirmed。未搜索远端实例，未填任何未知默认值。

这些不改变固定 prediction 的指标、bootstrap 或 T7 完成状态，但限制严格历史复现和训练变量的因果归因。T6 无保留配置验收 `HISTORICAL_RUNTIME_CONFIG_NOT_FULLY_VERIFIED` 及 C 冻结交接 pending 原样继承；它们与 **T7_EVALUATION_COMPLETE** 分字段记录，不据此阻塞本轮统计交付。

## 12. Validation

- T7 新增本地 tests：7/7 PASS；相关既有 tests：33/33 PASS。
- Python compileall、CI smoke、JSON parse、Markdown/JSON consistency、git diff --check：PASS。
- 所有 bootstrap 均使用整会话配对抽样；10,000 replicates，long 空分母0；逐样本分解、paired counts 与 T6 一致；输入/指标源码 SHA、原预测及 T5/T6 报告未变。
- 已知历史全套测试包含 Windows Bash/Unix 环境及依赖失败/skip；本轮只运行上述相关测试，不声称全套通过。
- 新增实现与测试、完整逐样本分数、10,000 组 replicate 和检查日志保留于 `.codex_tmp/acceptance/cycle2/b_t7_20261104/`；正式交付仅新增本轮 Markdown/JSON。
- 原始响应在 Top20 JSON 和折叠原文块中按用户本轮要求保留；没有修改 prediction、训练/推理实现、阈值、tokenizer 或历史报告。

## 13. Final T7 verdict

**`T7_EVALUATION_COMPLETE`；`selection_stability=SELECTION_STABLE`。**

效果方向 `NO_CLEAR_SIGNAL`；采用 `NO_GO`；工程选择 `KEEP_D4V2`。维持 `KEEP_DAY5 / KEEP_D4V2 / OFF / empty / max_new_tokens=256`。会话级不确定性与 regression review 已交付，provenance caveat 独立保留；本报告不是模型冻结配置或团队发布的无保留批准。
