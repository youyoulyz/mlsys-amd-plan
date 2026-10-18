# Prefill / Decode 激活专家聚类与话题漂移分析

## 0. 结论摘要

本分析基于现有 trace 的 expert route，不使用 GSM8K，也不重新运行模型。目标是判断：

1. 激活是否存在极端集中，例如“90% 激活集中在 10% 专家”。
2. prefill 和 decode 是否具有可区分的 expert 激活指纹。
3. math / python / moe 等话题之间的 top-10 专家是否明显漂移。
4. 同一个话题内部，专家排名是否随时间稳定。
5. 是否能从逐层 expert route 形成可解释的聚类特征。

核心结论：

- **“90% 激活集中在 10% 专家”不成立。**
- 每层 top-10% 专家平均承载约 46.7%-53.4% 的 route。
- 每层 average top-20% 专家承载约 65.6%-73.2% 的 route。
- 要达到 90% route 覆盖，平均需要每层约 92-110 个专家，也就是 35.9%-43.0% 的专家。
- 激活是有明显长尾的，但属于中等集中，不属于极端 90/10 分布。
- **prefill 和 decode 是很强的两个不同聚类。**
- prefill 与 decode 的逐层 top-10 平均只重叠约 0.55-0.83/10，fingerprint cosine 只有约 0.24-0.29。
- prefill 内部的 7k / 32k / 60k 高度相似，top-10 overlap 为 6.0-8.5/10。
- decode 内部的 math / python / moe 存在可区分性，但不是三个完全独立的簇：
  - math vs python：top-10 overlap 2.78/10
  - math vs moe：top-10 overlap 2.70/10
  - python vs moe：top-10 overlap 5.57/10
- **top-10 本身漂移很大，甚至同一话题内部也漂移。**
- 同一话题 first-85 vs last-85 的真实 top-10 overlap 只有约 2.8-3.1/10。
- 同一话题随机抽样 85 vs 85 的 top-10 overlap 基线约为 6.9-7.0/10。
- 因此该漂移不是单纯采样噪声，而是真实存在于 decode 的早/晚生成阶段。
- 如果要做聚类或路由特征，不建议只用 10 个 expert；应至少使用 top-10% 或完整的 256 维加权向量。

## 1. 数据与统计口径

输入：

- Trace：`experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/traces/trace-410395.tsv`
- Decode 逐专家统计：`experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/analysis-v1/decode_expert_stats.csv`
- Prefill 逐专家统计：`experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/analysis-v1/prefill_expert_stats.csv`

场景：

| 类型 | 场景 | 样本 |
|---|---|---:|
| Decode | math | 255 tokens |
| Decode | python | 255 tokens |
| Decode | moe | 255 tokens |
| Prefill | prefill_7k | 7,056 tokens |
| Prefill | prefill_32k | 32,055 tokens |
| Prefill | prefill_60k | 60,058 tokens |

模型结构：

- 40 layers
- 256 experts/layer
- top-8
- 每 token 共 320 个 expert route

注意：

- Expert id 是 layer-local 的，因此所有 top-k 和 overlap 比较都在同一层内完成。
- `top-10%` 指每层 256 个专家中的 26 个。
- `top10_abs` 指每层绝对 top-10。
- `top10_overlap` 指同一层两个场景的 top-10 集合交集大小除以 10。
- `effective_experts = exp(entropy)`，可以理解为“有效专家数”。

## 2. 激活集中度

### 2.1 汇总

| 场景 | top-1% / 3 experts | top-5% / 13 experts | top-10 / absolute | top-10% / 26 experts | top-20% / 51 experts | Gini | Effective experts |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_math | 11.5% | 31.6% | 26.9% | **46.7%** | 65.6% | 0.649 | 115.0 |
| decode_python | 11.6% | 33.5% | 28.2% | **50.5%** | 70.7% | 0.692 | 102.6 |
| decode_moe | 14.1% | 36.0% | 30.9% | **52.8%** | 73.1% | 0.712 | 95.0 |
| prefill_7k | 13.6% | 36.0% | 30.7% | **52.8%** | 72.3% | 0.707 | 97.1 |
| prefill_32k | 14.7% | 37.1% | 31.9% | **53.3%** | 72.5% | 0.711 | 94.8 |
| prefill_60k | 14.9% | 37.2% | 32.1% | **53.4%** | 72.6% | 0.712 | 94.4 |

解释：

- 如果专家路由接近均匀分布，top-10% 只能覆盖 10% route。
- 实测 top-10% 覆盖 46.7%-53.4%，约为均匀分布的 4.7-5.3 倍，说明存在明显热点。
- 但这距离“10% 专家覆盖 90% route”很远。
- prefill 的集中度略高于 decode。
- decode 的三个话题中，moe 最集中，math 最分散。

### 2.2 40 层全局

将 40 层各归一化后拼接成 10,240 维 fingerprint：

| 场景 | 全局 top-1% | 全局 top-5% | 全局 top-10% | 全局 top-20% | 全局 Gini |
|---|---:|---:|---:|---:|---:|
| decode_math | 11.2% | 31.9% | **46.6%** | 65.8% | 0.659 |
| decode_python | 10.8% | 33.8% | **50.4%** | 71.0% | 0.701 |
| decode_moe | 13.8% | 36.2% | **52.7%** | 73.4% | 0.721 |
| prefill_7k | 12.8% | 36.0% | **52.6%** | 72.8% | 0.712 |
| prefill_32k | 14.2% | 37.0% | **53.1%** | 73.2% | 0.718 |
| prefill_60k | 14.3% | 37.2% | **53.3%** | 73.2% | 0.719 |

结论与逐层一致：不存在 90/10 集中，只有约 50/10 集中。

### 2.3 覆盖 90% route 需要多少专家

| 场景 | k70 均值 | k80 均值 | k90 均值 | k90 中位数 | k90 范围 | k90 占 256 |
|---|---:|---:|---:|---:|---:|---:|
| decode_math | 60.5 | 81.1 | 110.1 | 107 | 68-182 | 43.0% |
| decode_python | 51.8 | 69.9 | 96.7 | 89.5 | 65-176 | 37.8% |
| decode_moe | 47.8 | 65.1 | 91.8 | 88 | 61-167 | 35.9% |
| prefill_7k | 47.7 | 66.0 | 94.4 | 95 | 59-123 | 36.9% |
| prefill_32k | 47.2 | 65.6 | 94.2 | 95 | 52-124 | 36.8% |
| prefill_60k | 47.3 | 65.4 | 93.9 | 95.5 | 54-124 | 36.7% |

所以更准确的描述是：

- 约 20% 专家覆盖 70% route。
- 约 30% 专家覆盖 80% 左右 route。
- 约 36%-43% 专家才能覆盖 90% route。

不同层差异很大。例如：

- decode_math 的 layer 0 最分散，top-10% 只覆盖 25.1%。
- decode_moe 的 layer 20 很集中，top-10% 覆盖 71.7%。
- prefill_32k 的 layer 39 最集中，top-10% 覆盖 77.1%。

因此 expert specialization 是强 layer-dependent 的。

## 3. Prefill / Decode 聚类特征

构建方式：

- 每层 256 维 expert route count 先归一化。
- 40 层拼接成 10,240 维场景 fingerprint。
- 比较 cosine 和同层 top-k set overlap。

### 3.1 Fingerprint cosine

|  | math | python | moe | prefill_7k | prefill_32k | prefill_60k |
|---|---:|---:|---:|---:|---:|---:|
| math | 1.00 | 0.58 | 0.50 | 0.28 | 0.27 | 0.28 |
| python | 0.58 | 1.00 | 0.78 | 0.26 | 0.24 | 0.24 |
| moe | 0.50 | 0.78 | 1.00 | 0.29 | 0.25 | 0.24 |
| prefill_7k | 0.28 | 0.26 | 0.29 | 1.00 | 0.93 | 0.87 |
| prefill_32k | 0.27 | 0.24 | 0.25 | 0.93 | 1.00 | 0.99 |
| prefill_60k | 0.28 | 0.24 | 0.24 | 0.87 | 0.99 | 1.00 |

可以看到两个明显聚类：

1. Prefill cluster：7k / 32k / 60k，cosine 0.87-0.99。
2. Decode cluster：math / python / moe，cosine 0.50-0.78。

跨 prefill/decode 的 cosine 只有 0.24-0.29，边界非常明显。

### 3.2 Top-10 集合漂移

| 场景对 | fingerprint cosine | top-10 overlap | top-10 Jaccard | top-26 overlap |
|---|---:|---:|---:|---:|
| math vs python | 0.58 | 2.78/10 | 0.166 | 42.2% |
| math vs moe | 0.50 | 2.70/10 | 0.160 | 39.6% |
| python vs moe | 0.78 | 5.57/10 | 0.397 | 62.8% |
| math vs prefill_7k | 0.28 | 0.58/10 | 0.032 | 10.3% |
| math vs prefill_32k | 0.27 | 0.57/10 | 0.031 | 10.5% |
| math vs prefill_60k | 0.28 | 0.57/10 | 0.031 | 10.9% |
| python vs prefill_7k | 0.26 | 0.70/10 | 0.039 | 10.0% |
| python vs prefill_32k | 0.24 | 0.55/10 | 0.031 | 9.7% |
| python vs prefill_60k | 0.24 | 0.55/10 | 0.031 | 10.7% |
| moe vs prefill_7k | 0.29 | 0.83/10 | 0.046 | 11.9% |
| moe vs prefill_32k | 0.25 | 0.65/10 | 0.037 | 10.2% |
| moe vs prefill_60k | 0.24 | 0.55/10 | 0.031 | 10.1% |
| prefill_7k vs prefill_32k | 0.93 | 7.05/10 | 0.564 | 73.0% |
| prefill_7k vs prefill_60k | 0.87 | 6.00/10 | 0.446 | 64.3% |
| prefill_32k vs prefill_60k | 0.99 | 8.53/10 | 0.754 | 89.0% |

结论：

- math 与 python / moe 的 top-10 漂移明显。
- python 与 moe 的 top-10 有较大重叠，说明这两个话题的 expert 指纹相近。
- prefill 与 decode 的 top-10 几乎不共享，是最强区分边界。
- 32k 与 60k prefill 非常相似，说明该长上下文工作负载的 expert 指纹随长度增加基本保持稳定。

### 3.3 top-10 之间的概率质量传递

| 场景对 | A 落在 B top-10 的质量 | B 落在 A top-10 的质量 | 对称均值 |
|---|---:|---:|---:|
| math vs python | 12.1% | 13.4% | 12.8% |
| math vs moe | 11.4% | 12.4% | 11.9% |
| python vs moe | 21.8% | 22.2% | 22.0% |
| math vs prefill_60k | 4.7% | 5.2% | 4.9% |
| python vs prefill_60k | 3.7% | 5.1% | 4.4% |
| moe vs prefill_60k | 4.3% | 5.0% | 4.7% |
| prefill_32k vs prefill_60k | 31.3% | 31.5% | 31.4% |

这说明：

- decode 话题之间存在一定共享核心，但不是完全相同。
- python / moe 共享约 22% 的 top-10 质量。
- prefill 与 decode 只共享约 4%-6% 的 top-10 质量。
- 同一 prefill 家族内部的 top-10 质量共享约 26%-31%。

## 4. 同一话题内部的时间漂移

把每个 255-token decode 请求拆成：

- first 85 tokens
- middle 85 tokens
- last 85 tokens

比较 first-85 与 last-85：

| 话题 | 真实 first-85 vs last-85 top-10 overlap | 同话题随机 85 vs 85 基线 | 差值 |
|---|---:|---:|---:|
| math | 2.98/10 | 6.90/10 | -3.92 |
| python | 2.83/10 | 7.00/10 | -4.18 |
| moe | 3.05/10 | 6.87/10 | -3.82 |

这不是单纯的采样噪声。即使在同一话题、同一请求内部，早段和晚段的 top-10 专家集合也会大幅变化。

可能原因：

- 生成早期和后期的 token 分布发生改变。
- 推理阶段从探索转向稳定输出。
- top-10 排名对少量高频专家的统计波动非常敏感。
- 这一现象在三个 decode 话题中都存在。

因此：

- top-10 可以描述某个时间窗口的即时指纹。
- top-10 不适合描述整个 255-token 请求的稳定身份。
- 要定义话题指纹，应使用更长窗口、top-10% 集合或完整加权分布。

## 5. Top-k 大小的影响

不同 top-k 的稳定性差异很大：

- top-10 exact set 很敏感，跨话题甚至同话题内部都会大幅漂移。
- top-26 set 更稳定，cross-topic overlap 约为 10%-63%。
- top-10% route mass 本身是比 exact set 更合适的特征。
- 256 维 route 分布归一化后的 cosine，是目前最稳定的聚类特征。

推荐的特征优先级：

1. 每层 256 维归一化 route count
2. top-10% / top-26 weighted mass
3. top-26 exact set overlap
4. top-10 exact set

## 6. 层级聚类特征

集中度有很强的 layer effect：

- 通常 layer 0-2 更分散，effective experts 接近 180-205。
- layer 8、14、20 等更集中，effective experts 约为 52-79。
- prefill 中 layer 37-39 通常最集中。
- decode 中 layer 20 往往是最集中的层之一。

逐层 top-10% 集中度的跨场景相关性：

| 场景对 | 相关性 |
|---|---:|
| math vs python | 0.905 |
| math vs moe | 0.857 |
| python vs moe | 0.930 |
| prefill_32k vs prefill_60k | 0.989 |
| prefill_7k vs prefill_60k | 0.857 |
| decode_math vs prefill_60k | -0.028 |
| decode_python vs prefill_60k | 0.100 |
| decode_moe vs prefill_60k | 0.136 |

这进一步支持两个 regime：

- 三个 decode 话题的“层间集中度结构”高度相关。
- prefill 三种上下文的层间集中度结构也高度相关。
- prefill 与 decode 的层间集中度几乎不相关。

因此 activation pattern 至少有两个可区分维度：

1. prefill vs decode 的执行模式。
2. 具体话题在 decode 内部的偏移。

## 7. 最终判断

### 7.1 “90% 激活集中在 10% 专家上”是否成立？

不成立。

实测：

- top-10% 专家承载约 46.7%-53.4% route。
- top-20% 专家承载约 65.6%-73.2% route。
- 90% route 平均需要约 92-110 个专家，约 35.9%-43.0% 的专家。

### 7.2 是否存在可区分的聚类特征？

存在，但主要体现在：

- prefill vs decode 是最强、最稳定的分界。
- decode 内部 math 与 python/moe 有明显差别。
- python 与 moe 有一定相似性。
- 同一话题内部也会随时间漂移。

### 7.3 Top-10 专家是否漂移很大？

是。

- math vs python / moe：约 2.7/10 重叠。
- python vs moe：约 5.6/10 重叠。
- prefill vs decode：通常小于 1/10。
- 同一 decode 话题的 first-85 vs last-85：约 2.8-3.1/10。

### 7.4 对 expert cache 的含义

- 单个话题没有足够稳定的“10 个专家常驻集”来覆盖全部工作负载。
- 热点是存在的，但热点集合会随 layer、prefill/decode 模式和生成阶段变化。
- 使用 topic-specific 或 phase-specific expert pinning 可能比单一全局 top-10 pinning 更合理。
- 但任何只 pin top-10 的策略都会漏掉大量 route，因为 top-10% 只能覆盖约一半激活。

## 8. 局限

1. 这里分析的是 expert route，不是 hidden state semantic clustering。
2. 没有分析 expert 权重或激活向量的表示相似性。
3. 三个 decode 话题各只有 255 tokens，只是单次 deterministic run。
4. 三个 decode 请求顺序执行，可能受前序生成内容和 cache 状态影响。
5. 长 prefill 的三种上下文不是互不相关的人工话题，prefill 内部高相似度可能反映共同的文本风格或位置统计。
6. top-10 exact overlap 对样本量和 tie-break 很敏感，因此必须结合 top-26、cosine 和 weighted mass 解释。
7. 本报告没有修改仓库代码，也没有提交 commit。

## 9. 产物

目录：

`experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/clustering-v1/`

| 文件 | 内容 |
|---|---|
| `concentration_summary.csv` | 各场景 top-k、Gini、effective experts |
| `layer_concentration.csv` | 40 层逐层集中度 |
| `global_concentration.csv` | 40 层全局集中度 |
| `experts_needed_for_coverage.csv` | 每层 k70 / k80 / k90 |
| `experts_needed_for_coverage_summary.csv` | k90 汇总 |
| `pairwise_similarity.csv` | 场景间 cosine、top-10/top-26 overlap |
| `layer_pair_drift.csv` | 每一层的场景对漂移明细 |
| `top10_mass_capture.csv` | top-10 集合捕获的概率质量 |
| `topic_bootstrap_similarity.csv` | 同话题/跨话题 bootstrap 基线 |
| `temporal_bootstrap.csv` | 同话题时间窗漂移及随机基线 |
| `top_experts_by_scenario_layer.csv` | 每个场景、每层 top-10/top-26 专家 |
| `concentration_curves.png` | 累计 top-k 覆盖曲线 |
| `experts_needed_90pct.png` | 覆盖 90% route 所需专家数 |
| `scenario_similarity.png` | 六场景 fingerprint cosine |
| `scenario_dendrogram.png` | 场景聚类树 |
| `top10_overlap_heatmap.png` | 每层 top-10 overlap 热力图 |
| `topic_separability_bootstrap.png` | 同话题与跨话题的匹配样本比较 |
