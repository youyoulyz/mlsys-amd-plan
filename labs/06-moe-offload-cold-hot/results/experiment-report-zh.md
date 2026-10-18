# Qwen3.6-35B-A3B-NVFP4 单卡 RTX 3080 20GB MoE Offload 实验报告

## 0. 摘要

本报告基于 FreeToken 在单张 RTX 3080 20GB、GPU1、PCIe 4.0 x8 上运行 Qwen3.6-35B-A3B-NVFP4 的实测 trace，分析 hybrid MoE offload 下：

- GPU 内稳定常驻的专家数量
- 每个 decode token 经 PCIe 从 host 拉取的专家数量
- 每个 decode token 留在 CPU 上计算的专家数量
- 逐层和逐 `(layer, expert)` 的常驻、缺失和拉取行为
- long prefill 与 long decode 的专家 cache 行为差异
- 随时间变化的 warm-up、平台期和抖动

本次没有使用 GSM8K。分析复用了上一轮已经跑过的 agent 工作负载：

- long prefill：7,056 / 32,055 / 60,058 prompt tokens
- long decode：三组 thinking 请求，每组 255 completion tokens
  - 数学推理
  - Python list vs dict
  - MoE expert-cache 设计

最重要的结论是：

1. 长 decode 中，稳定 expert cache 会快速吸收高频专家，随后进入低 fetch 平台期。
2. 每 token 从 PCIe 拉取的专家数在不同工作负载之间差异很大：
   - 冷 cache math：22.09 experts/token，37.40 MiB/token
   - 热 cache python：5.85 experts/token，9.91 MiB/token
   - 深热 cache moe：2.98 experts/token，5.05 MiB/token
3. 同一批 miss 中，只有一部分通过 PCIe 拉取，其余直接由 CPU 计算：
   - math：48.95 CPU-computed experts/token
   - python：21.06 CPU-computed experts/token
   - moe：14.61 CPU-computed experts/token
4. 请求内部没有观察到专家槽位驱逐。单个 `(layer, expert)` 在一个 255-token 请求内最多记录到 1 次 PCIe fetch。
5. Long prefill 的稳定 expert cache 利用率极低：
   - 7k prefill route hit：0%
   - 32k prefill route hit：1.85%
   - 60k prefill route hit：2.02%
   - 每个 chunk、每层仍然搬运完整 256 experts，约 0.423 GiB/layer/chunk
6. 当前 hybrid cache 的稳态行为不是“频繁换入换出”，而是“热专家一次拉取并长期常驻，长尾专家留下由 CPU 计算”。

## 1. 实验目标

本次实验回答以下问题：

1. 在 long prefill 和 long decode 场景下，GPU 内有多少专家处于稳定常驻状态？
2. 平均每个 token 从 CPU/host 侧经 PCIe 拉取多少专家？
3. 平均每个 token 有多少 miss 没有拉取，而是留在 CPU 计算？
4. 常驻专家是否随 token 时间明显波动？
5. 高频专家是否长期稳定，长尾专家是否长期缺席？
6. long prefill 和 long decode 的专家 cache 行为是否一致？
7. 专家级统计能否细化到单个 `(layer, expert)`？

## 2. 实验环境与配置

| 项目 | 配置 |
|---|---|
| 仓库 | `/mnt/luyuzhou/hpc/FreeToken` |
| 代码版本 | `71eb1f0` |
| 模型 | `/home/luyzh/models/Qwen3.6-35B-A3B-NVFP4` |
| GPU | RTX 3080 20GB，GPU1 |
| GPU 互联 | PCIe 4.0 x8 |
| MoE 策略 | hybrid |
| Expert cache 容量 | 7,200 个 expert slots |
| 模型层数 | 40 |
| 每层专家数 | 256 |
| Top-k | 8 |
| 单个专家权重 | 1,775,616 bytes |
| 单层完整专家权重 | 约 0.423 GiB |
| 全模型完整专家权重 | 约 16.93 GiB |
| Batch size | 1 |
| Max running requests | 1 |
| CUDA graph | 关闭 |
| Prefill 路径 | full-layer volatile double buffer；trace 中 `prefill_hit_d2d` 关闭 |
| Trace | `experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/traces/trace-410395.tsv` |
| 分析产物 | `experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/analysis-v1/` |

前序硬件路径实测参数：

- PCIe H2D：约 13.5 GB/s
- CPU NVFP4 MoE：约 38.5 GB/s
- hybrid auto fetch fraction：约 35.7%

注意：本报告中的 `fetched/missing` 是按 trace 的 route 统计，不是直接使用配置的 35.7% 推导。核心侧的 unique-expert 选择和 route 级统计并不完全等价。

## 3. 工作负载

### 3.1 Long prefill

复用上一轮长上下文 agent 工作负载，不使用 GSM8K：

| 名称 | Prompt tokens | Prefill chunks | chunk token 数 |
|---|---:|---:|---|
| prefill_7k | 7,056 | 1 | 7,056 |
| prefill_32k | 32,055 | 4 | 8,192 / 8,192 / 8,192 / 7,479 |
| prefill_60k | 60,058 | 8 | 8,192 × 7 / 2,714 |

### 3.2 Long decode

三组 deterministic thinking 请求，每组 255 decode tokens：

| Segment | 主题 | Decode tokens |
|---|---|---:|
| math | 数学推理 | 255 |
| python | Python list vs dict | 255 |
| moe | MoE expert-cache 设计 | 255 |

三组请求在一次 server 生命周期中顺序执行，因此 cache 状态是累积的：

- math 从近乎冷 cache 开始
- python 继承 math decode 后的稳定专家
- moe 继承 python decode 后的稳定专家

这使 python 和 moe 的结果更接近热 cache 行为，而 math 包含完整的 warm-up 过程。

## 4. 统计口径

Trace 中每条 decode layer event 记录：

| 字段 | 含义 |
|---|---|
| `raw` | 当前 token 在该层的 top-8 expert ids |
| `pre_resident` | 在本次处理前，该 route 命中的稳定 GPU cache mask |
| `post_slot` | 经过 hybrid 选择后，该 route 获得的 GPU slot，-1 表示未分配 |
| `fetched` | 分配了 GPU slot 并经 PCIe 从 host 拉取的 route 数 |
| `cpu_routed` | miss 后未分配 GPU slot，留在 CPU 计算的 route 数 |
| `missing` | `fetched + cpu_routed` |
| `resident_count` | 当前层稳定 cache 中的 expert 数 |
| `route_counts` | 当前层每个 expert 的 route 次数 |
| `active_unique` | 当前层实际被路由到的 unique experts |
| `resident_unique` | `active_unique` 中已在稳定 cache 的 experts |
| `missing_unique` | `active_unique` 中不在稳定 cache 的 experts |

本文中：

- “PCIe fetch/token” 指所有 40 层求和后的 `fetched`
- “CPU/token” 指所有 40 层求和后的 `cpu_routed`
- “miss/token” 指所有 40 层求和后的 `missing`
- 每 token 理论 route 数：`40 × 8 = 320`

## 5. Long decode 总体结果

| Workload | Resident start -> end | Resident mean | Resident P95 | Route hit | Miss/token | PCIe fetch/token | CPU/token | Fetch / miss | Fetch bytes/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 106 -> 5,729 | 3,862 | 5,609 | 77.80% | 71.04 | 22.09 | 48.95 | 31.1% | 37.40 MiB |
| python | 5,226 -> 6,719 | 6,295 | 6,694 | 91.59% | 26.92 | 5.85 | 21.06 | 21.8% | 9.91 MiB |
| moe | 6,207 -> 6,967 | 6,808 | 6,957 | 94.50% | 17.59 | 2.98 | 14.61 | 16.9% | 5.05 MiB |

补充统计：

| Workload | Fetch P50 | Fetch P95 | Fetch P99 | Missing P95 | Resident P50 |
|---|---:|---:|---:|---:|---:|
| math | 17 | 51 | 79.84 | 153.90 | 4,407 |
| python | 3 | 22 | 48.38 | 67.30 | 6,365 |
| moe | 1 | 8 | 40.22 | 37.00 | 6,846 |

### 5.1 常驻容量

模型共有：

`40 layers × 256 experts = 10,240` 个 layer-expert。

配置 cache：

`7,200` 个 expert slots。

因此即使 cache 完全装填，也只能容纳：

`7,200 / 10,240 = 70.3%`。

实测中：

- math 结束时占用 5,729/7,200，即 79.6%
- python 结束时占用 6,719/7,200，即 93.3%
- moe 结束时占用 6,967/7,200，即 96.8%

moe workload 已经接近配置容量，但仍平均每 token 有 14.61 个 expert route 留在 CPU。这说明稳态下 CPU fallback 的主要来源是 cache 无法覆盖的长尾专家，而不是明显的 cache 抖动。

### 5.2 请求内驱逐

请求内部观察结果：

- 每层 `resident_count` 在 255 个 decode steps 中没有下降。
- 总 resident 的 step 间增量与当前 step 的 fetch 数一致。
- 单个 `(layer, expert)` 在一个 255-token 请求内最多 fetch 1 次。
- 三条 workload 相加，单个 `(layer, expert)` 的最大 fetch 次数为 3，即每条 workload 最多一次。

因此，本 trace 中 decode 的主要成本是“第一次发现并拉取专家”，不是反复换入换出。

请求边界出现净下降：

- math -> python：5,729 -> 5,226，净减少 503
- python -> moe：6,719 -> 6,207，净减少 512

这发生在短 prefill 过渡附近，和请求内部的常驻行为应分开解释。

## 6. 时间波动分析

### 6.1 分窗口结果

| Workload | 窗口 | Fetch/token | CPU/token | Miss/token | Resident mean |
|---|---|---:|---:|---:|---:|
| math | first 8 | 66.25 | 130.00 | 196.25 | 384 |
| math | first 32 | 49.94 | 98.75 | 148.69 | 977 |
| math | 64-127 | 28.69 | 59.56 | 88.25 | 3,561 |
| math | last 128 | 10.40 | 28.78 | 39.18 | 5,164 |
| python | first 8 | 43.00 | 87.75 | 130.75 | 5,421 |
| python | first 32 | 17.69 | 43.06 | 60.75 | 5,649 |
| python | 64-127 | 3.36 | 17.20 | 20.56 | 6,254 |
| python | last 128 | 2.77 | 15.30 | 18.06 | 6,550 |
| moe | first 8 | 38.13 | 77.00 | 115.13 | 6,380 |
| moe | first 32 | 12.75 | 33.88 | 46.63 | 6,537 |
| moe | 64-127 | 1.48 | 11.23 | 12.72 | 6,797 |
| moe | last 128 | 0.95 | 10.55 | 11.49 | 6,912 |

### 6.2 Warm-up 后的稳定性

| Workload | Rolling-32 fetch min | Rolling-32 fetch max | Post-64 lag-1 corr | 无 fetch step |
|---|---:|---:|---:|---:|
| math | 5.34 | 104 | 0.829 | 0/255 |
| python | 1.81 | 77 | 0.212 | 29/255 |
| moe | 0.63 | 73 | 0.275 | 79/255 |

解释：

- math 在 255 tokens 内没有完全进入稳定平台，fetch 仍然呈下降趋势，因此 post-64 lag-1 仍较高。
- python 和 moe 在 warm-up 后 lag-1 分别只有 0.212 和 0.275，说明 step-to-step 没有强持久震荡。
- moe 最后 128 tokens 平均只有 0.95 fetch/token，且有 79/255 个 step 完全不需要 PCIe fetch。
- 残余波动主要来自路由组合变化，而不是 cache 周期性替换。

## 7. 逐层结果

下表是三组 decode 合并后的逐层平均值。单位均为每个 decode token。

| 层 | Resident mean | Resident P5-P95 | PCIe fetch/token | CPU/token | Route hit |
|---:|---:|---:|---:|---:|---:|
| 0 | 195.2 | 71-229 | 0.340 | 0.941 | 83.99% |
| 1 | 183.4 | 64-220 | 0.324 | 0.858 | 85.23% |
| 2 | 197.0 | 61-236 | 0.340 | 0.872 | 84.85% |
| 3 | 177.5 | 58-211 | 0.305 | 0.835 | 85.75% |
| 4 | 158.5 | 53-192 | 0.286 | 0.769 | 86.81% |
| 5 | 149.2 | 45-181 | 0.264 | 0.742 | 87.42% |
| 6 | 149.8 | 47-191 | 0.281 | 0.693 | 87.83% |
| 7 | 135.9 | 38-176 | 0.267 | 0.647 | 88.58% |
| 8 | 108.3 | 38-140 | 0.213 | 0.593 | 89.92% |
| 9 | 116.2 | 42-153 | 0.227 | 0.661 | 88.89% |
| 10 | 137.6 | 41-172 | 0.261 | 0.778 | 87.01% |
| 11 | 144.6 | 40-181 | 0.276 | 0.732 | 87.40% |
| 12 | 143.6 | 48-179 | 0.268 | 0.699 | 87.91% |
| 13 | 128.5 | 36-162 | 0.244 | 0.664 | 88.64% |
| 14 | 121.8 | 37-157 | 0.244 | 0.703 | 88.15% |
| 15 | 128.5 | 44-155 | 0.239 | 0.698 | 88.28% |
| 16 | 129.1 | 43-161 | 0.250 | 0.697 | 88.17% |
| 17 | 124.5 | 39-159 | 0.244 | 0.654 | 88.77% |
| 18 | 123.1 | 44-152 | 0.230 | 0.658 | 88.91% |
| 19 | 123.3 | 42-157 | 0.239 | 0.643 | 88.97% |
| 20 | 111.2 | 40-137 | 0.216 | 0.631 | 89.41% |
| 21 | 123.5 | 50-147 | 0.224 | 0.650 | 89.08% |
| 22 | 153.4 | 49-187 | 0.268 | 0.663 | 88.37% |
| 23 | 154.7 | 48-184 | 0.269 | 0.702 | 87.86% |
| 24 | 145.7 | 51-176 | 0.261 | 0.723 | 87.70% |
| 25 | 136.1 | 45-169 | 0.251 | 0.680 | 88.37% |
| 26 | 131.8 | 45-167 | 0.250 | 0.722 | 87.86% |
| 27 | 134.2 | 51-166 | 0.244 | 0.667 | 88.61% |
| 28 | 137.3 | 48-166 | 0.242 | 0.669 | 88.61% |
| 29 | 128.5 | 42-162 | 0.237 | 0.697 | 88.33% |
| 30 | 130.5 | 47-161 | 0.237 | 0.668 | 88.69% |
| 31 | 124.5 | 48-155 | 0.229 | 0.705 | 88.33% |
| 32 | 147.9 | 54-173 | 0.251 | 0.695 | 88.17% |
| 33 | 147.2 | 49-178 | 0.258 | 0.725 | 87.71% |
| 34 | 151.4 | 51-180 | 0.268 | 0.676 | 88.20% |
| 35 | 140.8 | 53-168 | 0.244 | 0.682 | 88.42% |
| 36 | 139.8 | 54-168 | 0.247 | 0.689 | 88.30% |
| 37 | 140.9 | 55-167 | 0.251 | 0.659 | 88.63% |
| 38 | 138.9 | 50-168 | 0.241 | 0.635 | 89.05% |
| 39 | 161.4 | 61-192 | 0.277 | 0.732 | 87.39% |

逐层观察：

- 平均常驻最少的层是 layer 8，108.3 experts。
- 平均常驻最多的层是 layer 2，197.0 experts。
- 平均 fetch 最少的层是 layer 8，0.213/token。
- 平均 fetch 最多的层是 layer 0 和 layer 2，0.340/token。
- layer 0-4 通常有更高的常驻量。
- layer 8、20、21 通常有更低的常驻量。
- 所有层的 route hit 都落在约 84%-90% 的窄区间内。

三组 workload 结束时，逐层常驻数量范围：

| Workload | 最终每层常驻范围 |
|---|---:|
| math | 99-210 |
| python | 132-229 |
| moe | 137-236 |

## 8. 逐专家结果

逐专家统计粒度是：

`segment × layer × expert`

共 30,720 条记录。字段包括：

- `routes`
- `misses`
- `fetched`
- `cpu`
- `resident_steps`
- `resident_rate`

完整文件：

`experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/analysis-v1/decode_expert_stats.csv`

### 8.1 常驻覆盖

`resident_rate` 表示该 `(layer, expert)` 在对应 255 个 decode steps 中位于稳定 GPU cache 的比例。

| Workload | 出现过的专家 | 曾经常驻 | 出现过但从未常驻 | 从未常驻 / 活跃 | 平均 resident_rate | resident_rate 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| math | 7,416 | 5,721 | 1,695 | 22.9% | 0.520 | 0.604 |
| python | 6,955 | 5,857 | 1,098 | 15.8% | 0.781 | 1.000 |
| moe | 6,796 | 5,925 | 871 | 12.8% | 0.848 | 1.000 |

分档统计：

| Workload | `resident_rate = 0` | `< 25%` | `> 90%` | `= 100%` |
|---|---:|---:|---:|---:|
| math | 1,695 | 2,177 | 1,404 | 98 |
| python | 1,098 | 1,274 | 4,897 | 4,364 |
| moe | 871 | 929 | 5,565 | 5,165 |

三组 workload 合并：

- 共出现 8,660 个不同的 `(layer, expert)`。
- 5,471 个专家在三组 workload 中都出现。
- 这 5,471 个持续活跃专家中，270 个从未进入稳定 cache，约占 4.9%。

### 8.2 高频稳定专家

下表跨越三条 255-token workload，因此 fetch 最大值可能为 3。

| Layer/Expert | Routes | Misses | PCIe fetch | CPU | Resident rate |
|---|---:|---:|---:|---:|---:|
| L35/E167 | 363 | 3 | 1 | 2 | 0.919 |
| L33/E081 | 323 | 3 | 3 | 0 | 0.988 |
| L20/E224 | 312 | 6 | 3 | 3 | 0.987 |
| L25/E013 | 312 | 3 | 1 | 2 | 0.970 |
| L21/E207 | 308 | 5 | 1 | 4 | 0.962 |
| L27/E206 | 301 | 5 | 1 | 4 | 0.962 |
| L15/E206 | 284 | 0 | 0 | 0 | 1.000 |
| L21/E243 | 278 | 0 | 0 | 0 | 1.000 |

高频专家的特点是路由次数高，并且在时间上长期稳定。L15/E206、L21/E243 在整个 765-step 分析窗口中始终常驻。

### 8.3 高成本长尾专家

这些专家有大量 route，但几乎从不进入稳定 cache，主要由 CPU 计算。

| Layer/Expert | Routes | Misses | PCIe fetch | CPU | Resident rate |
|---|---:|---:|---:|---:|---:|
| L33/E097 | 41 | 41 | 0 | 41 | 0.000 |
| L17/E241 | 86 | 39 | 2 | 37 | 0.631 |
| L36/E246 | 37 | 37 | 0 | 37 | 0.000 |
| L32/E140 | 56 | 37 | 1 | 36 | 0.337 |
| L32/E099 | 35 | 35 | 0 | 35 | 0.000 |
| L14/E058 | 30 | 30 | 0 | 30 | 0.000 |
| L05/E188 | 127 | 35 | 3 | 32 | 0.796 |

### 8.4 单专家 fetch 行为

- 单个 `(layer, expert)` 在一个 255-token workload 内最多 fetch 1 次。
- 三条 workload 合并后，最多 fetch 3 次。
- Top fetched expert 之一是 L05/E188：
  - routes = 127
  - misses = 35
  - fetched = 3
  - CPU = 32
- 这说明当前 trace 中没有观察到反复刷新同一专家权重。
- 当前 expert cache 更接近“热专家进入后保持常驻，长尾专家持续留 CPU”的选择性常驻策略。

## 9. Long prefill 结果

### 9.1 汇总

| Workload | Tokens | Chunks | Resident/layer mean | Active unique/layer | Route hit | 搬运粒度 |
|---|---:|---:|---:|---:|---:|---|
| prefill_7k | 7,056 | 1 | 0.00 | 231.6 | 0.00% | 256 experts/layer/chunk |
| prefill_32k | 32,055 | 4 | 2.51 | 206.7 | 1.85% | 256 experts/layer/chunk |
| prefill_60k | 60,058 | 8 | 2.71 | 199.6 | 2.02% | 256 experts/layer/chunk |

这里的 `Resident/layer mean` 是 prefill 每个 chunk、每层处理前稳定 cache 中的 expert 数，不包括 prefill 的 volatile double buffer。

### 9.2 搬运量

每个 chunk、每层：

`256 × 1,775,616 bytes = 454,557,696 bytes = 0.4233 GiB`

全模型 40 层、每个 chunk：

`0.4233 × 40 = 16.93 GiB`

因此：

| Workload | Chunks | 全模型 chunk 总搬运 |
|---|---:|---:|
| 7k | 1 | 16.93 GiB |
| 32k | 4 | 67.73 GiB |
| 60k | 8 | 135.46 GiB |

按每 token、每层折算：

| Workload | Chunk 平均 bytes/token/layer | 总字节/总 token |
|---|---:|---:|
| 7k | 64.4 KB | 64.4 KB |
| 32k | 56.8 KB | 56.7 KB |
| 60k | 69.5 KB | 60.5 KB |

60k 的 chunk 平均偏高，是因为最后一个 chunk 只有 2,714 tokens，仍然搬运了完整一层。

### 9.3 路由集中度

60k prefill 中最集中的 `(layer, expert)`：

| Layer/Expert | Routes in 60k | Resident before |
|---|---:|---:|
| L38/E209 | 59,942 | 约 1 |
| L34/E108 | 54,643 | 约 0-1 |
| L32/E066 | 53,541 | 约 0-1 |
| L33/E137 | 53,256 | 约 0-1 |
| L36/E081 | 49,437 | 约 0-1 |

L38/E209 几乎每个 token 都被路由一次，但 prefill 仍然几乎全部 fallback 到流式整层搬运。这说明低 route hit 不是路由没有热点，而是 long-prefill 路径没有充分利用稳定 expert cache。

### 9.4 短 prefill 对照

trace 后面还有两个很短的 prefill：

| Workload | Tokens | Resident/layer mean | Route hit |
|---|---:|---:|---:|
| python_prefill | 74 | 131.2 | 62.11% |
| moe_prefill | 80 | 155.5 | 74.15% |

这两个短 prefill 继承了前序 decode 的稳定 cache，因此 route hit 高。它们说明：

- 路由本身有很强的时间稳定性。
- long prefill 的低命中主要来自 prefill 路径使用 volatile full-layer buffer，而不是 expert 路由完全不可缓存。

## 10. 综合分析

### 10.1 Decode

Long decode 的专家 cache 行为可以分为三个阶段：

1. 冷启动阶段：大量 miss，PCIe fetch 和 CPU fallback 都很高。
2. 快速填充阶段：高频专家被拉入稳定 cache，fetch/token 快速下降。
3. 平台阶段：剩余 miss 主要来自低频长尾专家，PCIe fetch 很低，CPU fallback 仍持续存在。

在平台阶段：

- moe workload 平均只有 2.98 PCIe fetch/token。
- 但仍有 14.61 expert routes/token 在 CPU 计算。
- 这表示单纯增大 GPU cache 不一定能消除 CPU 部分，仍受 cache 容量和 admission policy 影响。

### 10.2 Long prefill

Long prefill 在当前 trace 中表现为：

- 稳定 cache 几乎没有服务 prefill route。
- 每个 chunk 都把完整 256-expert layer 搬入 volatile buffer。
- 即使某个专家在几乎每个 token 都被路由，也不会因此形成稳定的 prefill cache hit。
- 因此 long prefill 的成本更接近“每 chunk 扫描完整模型专家层”，而不是“按活跃专家稀疏加载”。

### 10.3 PCIe x8 的含义

从数据看，PCIe x8 的成本并不是线性贯穿所有 decode token：

- cold math 阶段：37.40 MiB/token 的 fetch 流量
- warm moe 阶段：5.05 MiB/token 的 fetch 流量

因此 warm-up 阶段和 long prefill 阶段更容易暴露 x8 带宽压力；稳态 decode 的 PCIe 流量明显更低，但 CPU fallback 计算仍然会限制性能。

## 11. 局限性

1. Probe 关闭 CUDA graph，并在路由和 cache mask 处增加 host synchronization。
2. Probe 状态下 decode 约 14 tok/s，明显低于正常状态的约 80-90 tok/s。
3. 本报告不能把 probe 的延迟直接当作正式吞吐 benchmark。
4. 路由和 cache 计数来自真实执行路径，适合分析专家选择、fetch 和常驻行为。
5. 每条 workload 只做了一次 deterministic run，temperature=0，batch size=1。
6. 三个 decode workload 顺序执行，python 和 moe 不是独立冷启动。
7. 请求边界出现 503 和 512 的 resident 净下降；本报告只记录该现象，未进一步归因于具体 cache transition。
8. Prefill 结果反映的是当前 `prefill_hit_d2d=False` 的路径行为，不能外推到其他配置。
9. 本报告没有使用 GSM8K，也没有把它作为数据集评测结果。
10. 本报告没有修改仓库代码，也没有提交 commit。

## 12. 结论

### 12.1 直接回答

**GPU 内驻留专家情况**

- Decode 每层平均常驻范围约 108-197 experts。
- 常见稳态范围约 120-165 experts/layer。
- moe workload 结束时常驻总数 6,967，接近 7,200 slot 上限。

**平均每 token 从 CPU 拉取多少专家**

| Workload | PCIe fetch/token | 流量 |
|---|---:|---:|
| math | 22.09 | 37.40 MiB |
| python | 5.85 | 9.91 MiB |
| moe | 2.98 | 5.05 MiB |

**平均每 token 有多少专家没有拉取、留在 CPU 计算**

| Workload | CPU-computed experts/token |
|---|---:|
| math | 48.95 |
| python | 21.06 |
| moe | 14.61 |

**时间波动**

- 没有观察到稳态下持续、剧烈的周期性换入换出。
- 主要波动是 warm-up 过程中 fetch 快速下降。
- 请求内部 expert 槽位单调增长。
- 单个专家在一个请求中通常只 fetch 一次。
- moe 最后 128 tokens 平均 fetch 已降到 0.95/token。

**Long prefill**

- 7k/32k/60k 的 route hit 分别为 0%、1.85%、2.02%。
- 每 chunk、每层仍搬运全部 256 experts。
- 稳定 expert cache 对当前 long-prefill 路径几乎没有贡献。

### 12.2 一句话总结

当前单卡 3080 20GB 的 hybrid MoE 行为是：**long decode 下热专家快速常驻、长尾专家长期由 CPU 承担；long prefill 下稳定 expert cache 基本失效，主要成本来自每个 chunk 对完整专家层的流式搬运。**

## 13. 分析产物

目录：

`experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/analysis-v1/`

| 文件 | 内容 |
|---|---|
| `results.json` | 汇总统计和元数据 |
| `decode_per_layer.csv` | 40 层 decode 汇总 |
| `decode_expert_stats.csv` | 30,720 行逐 `(segment, layer, expert)` 统计 |
| `prefill_expert_stats.csv` | 逐 prefill chunk、layer、expert 统计 |
| `prefill_layer_chunk.csv` | 逐 prefill layer/chunk 汇总 |
| `decode_timeseries.png` | PCIe fetch、CPU fallback、常驻总量的时间序列 |
| `decode_per_layer.png` | 逐层 fetch/CPU/resident 图 |
| `decode_expert_heatmap.png` | 逐专家热力图 |
| `prefill_layer_chunk.png` | prefill 的稳定 cache 和 active expert 热力图 |
| `analyze_experts.py` | 分析脚本 |
| `run.log` | 分析执行日志 |
