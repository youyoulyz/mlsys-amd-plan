# 02 · 实测读数据比官方带宽还快 3 倍：卡里藏着一层 96 MB 的"超快缓存"

> **一句话在做什么**：在这张 24 GB 游戏显卡上，控制"每次读多大一块数据"逐档测读取速度，找速度随数据量变化的曲线。实测：数据 ≤ 96 MB 时速度 1.8–2.6 TB/s（官方显存带宽 960 GB/s 的 **2–3 倍**），超过 96 MB **断崖式**掉到 ~800 GB/s——中间没有过渡。

## 实验导读

这张卡（RDNA3 架构）官方标称的显存带宽是 960 GB/s。但 lab 01 实测出来的"等效带宽"有时能超过 960——物理上不可能，除非数据没有走显存。这个实验就是找到那条"不可能数字"的分界线在哪。

跑完（或读完）你会知道：

- 这块 96 MB 的片上缓存（Infinity Cache）不是"大号 L2"，是一条独立的高速层，而且边界是硬的：80 MB → 2083 GB/s，112 MB → 823 GB/s，32 MB 之差差 2.5 倍；
- 什么时候生成速度跑在这层缓存上而不是显存上——单步读出的权重 < 96 MB 且反复读时；
- 为什么你看到的"等效带宽"数字必须标注"测的是哪一层"，否则同一张卡的两个数字可以差 2.5 倍而且都对；
- 反例：4 bit 量化格式为什么吃不到这层缓存的红利（它的瓶颈在"解压权重"的计算，不在读数据）。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| 工作集大小 | 一次读多大的数据（用来找悬崖在哪） | 80 / 96 / 112 MiB 等 |
| 96 MiB | 片上缓存容量（悬崖的分界线） | 这张卡固定 96 MiB |
| Q8_0 / F32 | 权重存储格式（8 bit 量化 / 32 bit 浮点） | 2 种对照 |
| n=1 | 一次只算 1 个 token（单用户生成形状） | 1 |

**主题**：Batch-Decode / 性能与测量 / 校准
**硬件**：AMD RX 7900 XTX（**96 MiB 片上缓存 Infinity Cache**）· ROCm 10.0.0 — plan **T2** 档
**强化**：第 2 周（时间都去哪了：在 AMD 上校准）
**状态**：实测已有（含 profiler traces），自包含脚本可复跑

## 你会怎么猜

RDNA3 有 96 MiB 的 Infinity Cache（profiler 里叫 `GL2C`）。你的第一反应大概是：

- "就是个**大一点的 L2**，对 decode 影响有限，主要还是看 HBM。"
- 用 `等效带宽 = 权重字节 / 时间`，理论峰值取 HBM 的 960 GB/s 去算达成率。

## 实测会看到

96 MiB Infinity Cache **不是"大 L2"，是一条独立的高带宽层**，而且边界是**硬的**：

| 工作集 | 实测等效带宽 |
| --- | --- |
| ≤ ~96 MiB | **1.8–2.6 TB/s**（Infinity Cache） |
| 过 96 MiB | **断崖式**掉到 **~800–890 GB/s**（HBM） |

断点是硬边界，不是渐变（`read_sum` microbench）：

- 80 MiB → **2083 GB/s**
- 96 MiB → **1997 GB/s**
- 112 MiB → **823 GB/s** ← 断在这里

用 llama.cpp 真实 kernel 复现（n=1）：

| 权重 | 79.7 MiB（< 96） | 111.6 MiB（> 96） |
| --- | --- | --- |
| Q8_0 | **2093 GB/s** | **812 GB/s** |
| F32 | 2140 GB/s | 821 GB/s |

**超过 960 GB/s 的"表观带宽"只有用 Infinity Cache 才能解释。** 而单算子权重经常整块放得进这 96 MiB（4B BF16 的 `gate_up` = 95 MiB 正好贴上限；7B 只有 qkv/o 放得下）——所以 **decode（n=1）阶段的"等效权重带宽"很多时候跑在 Infinity Cache 上，不是 HBM 上**。

反例（同样重要）：**Q4_K 的 n=1 没从 Infinity Cache 明显受益**（峰值 1023 → 外部 746 GB/s，只有 1.37×）——因为它的瓶颈是**反量化/计算**，不是访存。这正好解释了 lab 01 里"为什么 Q4_K 贴着 HBM 峰值却仍慢"。

**一句话结论**：用纯 HBM 带宽模型去预测/解释，会让"算存比 / ridge 达成率"的结论偏差 **2–2.5 倍**。任何"等效带宽"数字都必须标注**测的是哪一层**。

## 为什么会这样

decode 每个 token 把同一份权重重读一遍。当这份权重 < 96 MiB 且被反复读，它就驻留在 Infinity Cache，此时服务它的是 Infinity Cache 的 ~2 TB/s 而非 HBM 的 960 GB/s。于是出现"表观带宽 > HBM 峰值"这种纯 HBM 模型里不可能有的数字。把这种"虚高"当成达成率分子去和 HBM ridge 比，会得到偏乐观的达成率；而权重大于 96 MiB 时又是纯 HBM——两层混在同一张图里就不可比了。

## 用了什么工具，为什么

- **为什么自己写微基准而不是跑完整模型**：找"悬崖在哪"需要精确控制"每次读多大"——完整模型的读写模式你控制不了。微基准把变量隔离到只剩"数据量"。
- **为什么还要用 llama.cpp 真实 kernel 复现一遍**：微基准证明悬崖存在，真实 kernel 证明悬崖在真实工作负载里同样存在（Q8_0：79.7 MiB → 2093 GB/s，111.6 MiB → 812 GB/s）。两相对照，结论才不是基准程序的假象。
- **profiler（rocprofv3）是什么**：AMD 官方的性能分析器，能给出每段计算读/写了多少字节、花多长时间——"等效带宽 = 字节 ÷ 时间"的分子分母都从它来。

## 跑法

本地 `code/` 是三组关键脚本，`results/` 是完整数据 + 图 + 报告：

```bash
python code/bench_cache_sweep.py      # 缓存工作集扫描 microbench（画出 96 MiB 悬崖线）
python code/sweep_llama_mulmat.py     # llama.cpp 真实 kernel n=1 复现（2093 -> 812 GB/s）
python code/analyze_ifcache.py        # 数据分析与出图
```
关键结果：`results/cache_sweep.csv`、`results/llama_mulmat_sweep.csv`、`results/weight_footprint.csv`、`results/figures/`；完整报告 `results/report-infinity-cache.md`。
在 T4 上重跑时，MI 系列的 L2/缓存层次不同（Infinity Cache 是 RDNA 专有），预期这条悬崖**不会以同样形态出现**——这本身就是一组值得记录的"消费卡 vs 数据中心卡"差异。

## 来源

- 报告（含结论 + 三组实验设计）：`/mnt/luyuzhou/hpc/moecho/experiments/ifcache_arith_7900xtx_20260924/REPORT.md`
- 实验目录：`/mnt/luyuzhou/hpc/moecho/experiments/ifcache_arith_7900xtx_20260924/`（`scripts/` `traces/` `results/`）
- 硬件/环境：RX 7900 XTX gfx1100，ROCm 10.0.0，PyTorch 2.12.0+rocm10.0.0，llama.cpp `1bc7a5af`，rocprofv3 1.3.5

## 延伸

- 这是 lab 01（roofline）的"续集"：为什么等效带宽会超过 HBM 峰值。
- 校准视角（第 2 周）：把 Infinity Cache 当成每档硬件要拟合的一个**独立带宽参数**，而不是笼统的"利用率"。