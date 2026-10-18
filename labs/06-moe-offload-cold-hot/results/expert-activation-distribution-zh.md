# Qwen3.6-35B-A3B-NVFP4 专家激活分布报告

## 0. 范围

本报告分析单卡 RTX 3080 20GB、GPU1、PCIe 4.0 x8 上 Qwen3.6-35B-A3B-NVFP4 的 MoE expert route 分布。

分析数据来自已有 trace，没有重新运行 GSM8K，也没有把 GSM8K 作为数据集：

- Long prefill：7,056 / 32,055 / 60,058 prompt tokens
- Long decode：math / Python list-vs-dict / MoE expert-cache 设计，各 255 tokens
- Batch size 1
- 40 layers
- 256 experts/layer
- top-8
- 每 token 320 个 expert route

本文中的“激活”指 routing 选中某个 expert 的次数，也就是 route slot 数，不是 hidden state activation magnitude。

## 1. 关键结论

1. **“90% 激活集中在 10% 专家上”不成立。**
2. 每层 top-10% 专家平均承载约 **46.7%-53.4%** 的 route。
3. 每层 top-20% 专家平均承载约 **65.6%-73.2%** 的 route。
4. 要达到 90% route 覆盖，平均需要每层约 **92-110 个专家**，即 **35.9%-43.0%** 的专家。
5. 专家分布有明显长尾，但属于中等集中，不是极端 90/10 分布。
6. Prefill 和 decode 是两个明显不同的激活 regime：
   - prefill 内部 fingerprint cosine：0.87-0.99
   - decode 内部 fingerprint cosine：0.50-0.78
   - prefill 与 decode 之间：0.24-0.29
7. 话题之间 top-10 漂移明显：
   - math vs python：2.78/10 overlap
   - math vs moe：2.70/10 overlap
   - python vs moe：5.57/10 overlap
   - prefill vs decode：通常低于 1/10
8. 同一个 decode 话题内部也会漂移：
   - first-85 vs last-85 top-10 overlap 约 2.8-3.1/10
   - 同话题随机重采样基线约 6.9-7.0/10
9. 如果要用于分类、缓存或 prefetch，top-10 exact set 太窄；应优先使用完整归一化 256 维分布或 top-10% 加权质量。

## 2. 统计口径

| 指标 | 定义 |
|---|---|
| `top-10%` | 每层 256 experts 中的 26 个 |
| `top-10pct_26` | top-26 experts 覆盖的 route 比例 |
| `top20pct_51` | top-51 experts 覆盖的 route 比例 |
| `effective_experts` | `exp(entropy)`，分布的有效专家数 |
| `top10_overlap` | 同一层两个场景的 top-10 交集大小除以 10 |
| `fingerprint cosine` | 40 层归一化 256 维 route 向量拼接后的 cosine |

所有 overlap 都在同一层内计算。Expert id 是 layer-local 的，不能跨层直接比较 expert id。

## 3. 整体专家分布

| 场景 | top-1% / 3 experts | top-5% / 13 experts | top-10 / absolute | top-10% / 26 experts | top-20% / 51 experts | Gini | Effective experts |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode math | 11.5% | 31.6% | 26.9% | **46.7%** | 65.6% | 0.649 | 115.0 |
| decode python | 11.6% | 33.5% | 28.2% | **50.5%** | 70.7% | 0.692 | 102.6 |
| decode moe | 14.1% | 36.0% | 30.9% | **52.8%** | 73.1% | 0.712 | 95.0 |
| prefill 7k | 13.6% | 36.0% | 30.7% | **52.8%** | 72.3% | 0.707 | 97.1 |
| prefill 32k | 14.7% | 37.1% | 31.9% | **53.3%** | 72.5% | 0.711 | 94.8 |
| prefill 60k | 14.9% | 37.2% | 32.1% | **53.4%** | 72.6% | 0.712 | 94.4 |

如果 expert 路由接近均匀，top-10% 只能覆盖 10%。实测覆盖约 47%-53%，说明热点显著，但远不到 90%。

### 3.1 40 层全局分布

将每一层归一化后再拼接：

| 场景 | 全局 top-1% | 全局 top-5% | 全局 top-10% | 全局 top-20% | 全局 Gini |
|---|---:|---:|---:|---:|---:|
| decode math | 11.2% | 31.9% | **46.6%** | 65.8% | 0.659 |
| decode python | 10.8% | 33.8% | **50.4%** | 71.0% | 0.701 |
| decode moe | 13.8% | 36.2% | **52.7%** | 73.4% | 0.721 |
| prefill 7k | 12.8% | 36.0% | **52.6%** | 72.8% | 0.712 |
| prefill 32k | 14.2% | 37.0% | **53.1%** | 73.2% | 0.718 |
| prefill 60k | 14.3% | 37.2% | **53.3%** | 73.2% | 0.719 |

结论与逐层统计一致：这是一条明显长尾，但不是极端 90/10 分布。

## 4. 覆盖 90% 需要多少专家

| 场景 | k70 均值 | k80 均值 | k90 均值 | k90 中位数 | k90 范围 | k90 占 256 |
|---|---:|---:|---:|---:|---:|---:|
| decode math | 60.5 | 81.1 | 110.1 | 107 | 68-182 | 43.0% |
| decode python | 51.8 | 69.9 | 96.7 | 89.5 | 65-176 | 37.8% |
| decode moe | 47.8 | 65.1 | 91.8 | 88 | 61-167 | 35.9% |
| prefill 7k | 47.7 | 66.0 | 94.4 | 95 | 59-123 | 36.9% |
| prefill 32k | 47.2 | 65.6 | 94.2 | 95 | 52-124 | 36.8% |
| prefill 60k | 47.3 | 65.4 | 93.9 | 95.5 | 54-124 | 36.7% |

准确的覆盖率关系是：

- 约 20% 专家覆盖 70% route。
- 约 25%-30% 专家覆盖 80% 左右 route。
- 约 36%-43% 专家才覆盖 90% route。

## 5. 逐层分布特征

不同层的专家集中度差异明显：

| 场景 | 最分散层 | 最集中层 |
|---|---|---|
| decode math | layer 0，top-10% = 25.1% | layer 9，top-10% = 62.5% |
| decode python | layer 0，top-10% = 27.0% | layer 14，top-10% = 62.5% |
| decode moe | layer 2，top-10% = 28.7% | layer 20，top-10% = 71.7% |
| prefill 7k | layer 2，top-10% = 34.7% | layer 39，top-10% = 72.4% |
| prefill 32k | layer 2，top-10% = 36.1% | layer 39，top-10% = 77.1% |
| prefill 60k | layer 2，top-10% = 36.1% | layer 39，top-10% = 76.2% |

常见模式：

- layer 0-2 通常更分散，effective experts 接近 180-205。
- layer 8、14、20 等层更集中，effective experts 约为 52-79。
- prefill 的 layer 37-39 通常最集中。
- decode moe 的 layer 20 是最集中的层之一。

逐层 top-10% 集中度的跨场景相关性：

| 场景对 | 相关性 |
|---|---:|
| math vs python | 0.905 |
| math vs moe | 0.857 |
| python vs moe | 0.930 |
| prefill 32k vs prefill 60k | 0.989 |
| prefill 7k vs prefill 60k | 0.857 |
| decode math vs prefill 60k | -0.028 |
| decode python vs prefill 60k | 0.100 |
| decode moe vs prefill 60k | 0.136 |

这支持将 prefill 和 decode 视为两种不同的层间集中度结构。

## 6. Prefill / Decode 聚类

用 40 层归一化 256 维 route 分布拼接成 fingerprint，cosine 为：

|  | math | python | moe | prefill 7k | prefill 32k | prefill 60k |
|---|---:|---:|---:|---:|---:|---:|
| math | 1.00 | 0.58 | 0.50 | 0.28 | 0.27 | 0.28 |
| python | 0.58 | 1.00 | 0.78 | 0.26 | 0.24 | 0.24 |
| moe | 0.50 | 0.78 | 1.00 | 0.29 | 0.25 | 0.24 |
| prefill 7k | 0.28 | 0.26 | 0.29 | 1.00 | 0.93 | 0.87 |
| prefill 32k | 0.27 | 0.24 | 0.25 | 0.93 | 1.00 | 0.99 |
| prefill 60k | 0.28 | 0.24 | 0.24 | 0.87 | 0.99 | 1.00 |

两个主要簇：

1. Prefill cluster：7k / 32k / 60k，cosine 0.87-0.99。
2. Decode cluster：math / python / moe，cosine 0.50-0.78。

decode 内部还可以进一步区分：

- math 与 python / moe 的 cosine 只有 0.50-0.58。
- python 与 moe 的 cosine 为 0.78，二者更接近。

## 7. 话题之间的 top-10 漂移

| 场景对 | Top-10 overlap | Top-26 overlap | Top-10 Jaccard | Fingerprint cosine |
|---|---:|---:|---:|---:|
| math vs python | **2.78/10** | 42.2% | 0.166 | 0.58 |
| math vs moe | **2.70/10** | 39.6% | 0.160 | 0.50 |
| python vs moe | **5.57/10** | 62.8% | 0.397 | 0.78 |
| math vs prefill 60k | **0.57/10** | 10.9% | 0.031 | 0.28 |
| python vs prefill 60k | **0.55/10** | 10.7% | 0.031 | 0.24 |
| moe vs prefill 60k | **0.55/10** | 10.1% | 0.031 | 0.24 |
| prefill 32k vs prefill 60k | **8.53/10** | 89.0% | 0.754 | 0.99 |

对应的 top-10 概率质量：

| 场景对 | 对称 top-10 质量共享 |
|---|---:|
| math vs python | 12.8% |
| math vs moe | 11.9% |
| python vs moe | 22.0% |
| decode vs prefill | 4.4%-4.9% |
| prefill 32k vs 60k | 31.4% |

解释：

- math 与 python / moe 的 top-10 漂移很大。
- python 与 moe 有较高重叠，不是两个完全独立的簇。
- prefill 与 decode 的 top-10 基本不共享，是最强边界。
- prefill 32k 与 60k 几乎共享同一组热点。

## 8. 同一话题内部的时间漂移

将每条 255-token decode 请求拆成 first-85 与 last-85：

| 话题 | 真实 first-85 vs last-85 Top-10 overlap | 同话题随机 85 vs 85 基线 | 差值 |
|---|---:|---:|---:|
| math | 2.98/10 | 6.90/10 | -3.92 |
| python | 2.83/10 | 7.00/10 | -4.18 |
| moe | 3.05/10 | 6.87/10 | -3.82 |

随机基线来自同一话题内部随机抽取两组 85 tokens。真实早/晚窗口 overlap 明显低于随机基线，说明这不是单纯采样噪声，而是真实的生成阶段漂移。

因此：

- top-10 适合描述当前窗口的即时热点。
- top-10 不适合表示整个请求的稳定专家身份。
- 要描述话题身份，建议使用完整分布或 top-10% 加权集合。

## 9. 对 expert cache 的含义

当前数据下的直接含义：

1. 热点存在，但不是极端 10 个专家主导。
2. 只用 top-10 pinning 最多覆盖约 27%-32% route。
3. 用 top-10% pinning 大约覆盖一半 route。
4. 要覆盖 90% route，需要约 36%-43% 专家。
5. 热点集合随以下因素变化：
   - layer
   - prefill 或 decode 模式
   - decode 的话题
   - decode 的生成阶段
6. prefill 和 decode 的 top-10 几乎不共享，因此二者不适合共用单一固定 pin 集合。
7. 更合理的策略是：
   - 按 layer 维护热点集合。
   - 区分 prefill 与 decode。
   - decode 内部考虑 phase-aware 或 topic-aware 的专家集合。
   - 使用加权分布或 top-10% 以上的集合，而不是固定 top-10。

## 10. 局限

1. 本文统计的是 expert route 数量，不是 hidden state activation magnitude。
2. 没有分析 expert 权重向量的表示相似性。
3. 三个 decode 话题各只有 255 tokens，且是 deterministic 单次运行。
4. 三个 decode 请求顺序执行，前序请求可能影响后续状态。
5. Long prefill 的场景之间可能有共同文本或位置统计，prefill cluster 的高相似度不能直接解释为语义任务相同。
6. Top-10 exact overlap 对样本量和 tie-break 敏感，应结合 top-26、cosine 和 weighted mass 使用。

## 11. 结果文件

```text
results/analysis-v1/decode_expert_stats.csv
results/analysis-v1/prefill_expert_stats.csv
results/analysis-v1/decode_per_layer.csv
results/clustering-v1/concentration_summary.csv
results/clustering-v1/layer_concentration.csv
results/clustering-v1/global_concentration.csv
results/clustering-v1/experts_needed_for_coverage_summary.csv
results/clustering-v1/pairwise_similarity.csv
results/clustering-v1/layer_pair_drift.csv
results/clustering-v1/top10_mass_capture.csv
results/clustering-v1/top_experts_by_scenario_layer.csv
results/clustering-v1/temporal_bootstrap_summary.csv
```

图表：

```text
results/clustering-v1/concentration_curves.png
results/clustering-v1/experts_needed_90pct.png
results/clustering-v1/scenario_similarity.png
results/clustering-v1/scenario_dendrogram.png
results/clustering-v1/top10_overlap_heatmap.png
results/clustering-v1/topic_separability_bootstrap.png
```
