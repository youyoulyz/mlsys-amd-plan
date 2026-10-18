# RX 7900 XTX：Infinity Cache（GL2C/L2）与算存比补实验

> 目的：回答“原带宽分析把整个 GPU 当一条 960 GB/s 的 HBM 管子，是否忽略了
> 96 MiB Infinity Cache（IF Cache）”。结论是——**确实忽略了，而且这个忽略会
> 让“算存比 / ridge 达成率”的结论出现 2~2.5 倍的偏差**。本目录用三组实验
> 把 IF Cache 从“总带宽”里切出来单独量化。

## 0. 一句话结论

- 7900 XTX 的 96 MiB IF Cache 不是“大一点的 L2”，而是一条**独立的高带宽层**：
  工作集 ≤ ~96 MiB 时实测等效带宽 **1.8–2.6 TB/s**，超过 96 MiB 后**断崖式**
  掉到 **~800–890 GB/s**（HBM）。
- 断点是**硬边界**，不是渐变：`read_sum` 在 80 MiB = 2083 GB/s，96 MiB = 1997 GB/s，
  112 MiB = 823 GB/s。
- LLM 的**单个算子权重**经常整块放得进这 96 MiB（4B 的 qkv/o/gate_up/down 全部
  放得下；4B BF16 `gate_up` = 95 MiB 正好贴上限；7B 只有 qkv/o 放得下）。
  所以 **decode（n=1）阶段的“等效权重带宽”很多时候是 IF 带宽，不是 HBM 带宽**。
- 用 llama.cpp 真实 kernel 复现：`n=1` 的 Q8_0 权重从 79.7 MiB 的 2093 GB/s
  掉到 111.6 MiB 的 812 GB/s；F32 从 2140 → 821 GB/s。**超过 960 GB/s 的“表观
  带宽”只有用 IF Cache 才能解释。**
- 反例：**Q4_K 的 n=1 并没有从 IF 明显受益**（峰值 1023→外部 746 GB/s，只有
  1.37×），因为它的瓶颈是反量化/计算，不是访存。这解释了原报告里
  “Q4_K n=1 只有 900 GB/s”为何贴着 HBM 峰值却仍慢。
- 因此原报告 §3.2 / §5 / §6 / §8 中所有“等效权重带宽”都必须标注**测的是哪一层
  带宽**；`N=16` 的 ridge 达成率只能给**下界**，真实区间要按 IF/HBM 两档给。

## 1. 为什么原分析会被 IF Cache 带偏

原报告的算法是：

```
等效权重带宽 = 权重字节数 / 端到端时间
ridge 达成率 = 实测等效带宽 ÷ 理论峰值带宽
```

其中“理论峰值带宽”取 960 GB/s（HBM）。问题在于：

- 当算子权重 < 96 MiB 且被反复读取（decode 的每个 token 都要把同一份权重
  重读一遍），**这份权重会驻留在 IF Cache**，此时实际服务它的是 IF 的
  ~2 TB/s，而不是 HBM 的 960 GB/s。
- 于是会出现“表观带宽 > 960 GB/s”这种在纯 HBM 模型里不可能的数字
  （原报告里 Q8_0 n=1 = 1811 GB/s 就是这一类）。
- 反过来，当把这种“虚高”当成达成率分子去和 HBM ridge 比，会得到偏乐观的
  达成率；而当权重大于 96 MiB（7B 的 gate_up/down/lm_head）时又是纯 HBM，
  两者混在同一张图里就不可比了。

## 2. 硬件背景

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX，gfx1100，RDNA3，48 CU |
| 显存 | 24 GiB GDDR6，标称带宽 960 GB/s |
| Infinity Cache | **96 MiB**（6 × MCD × 16 MiB），在 profiler 里就是 `GL2C` 块 |
| ROCm | 10.0.0 |
| PyTorch | 2.12.0+rocm10.0.0 |
| llama.cpp | `1bc7a5af`（2026-09-15），`build-gfx1100-rocm10` |
| rocprofv3 | 1.3.5 |

> 说明：在 RDNA3 的计数器命名里，L2 就叫 `GL2C`，它就是承载 Infinity Cache
> 的那一级缓存。下文 “IF Cache / GL2C / L2” 指同一层。

## 3. 实验方法

### 3.1 实验 A —— 工作集扫描（cache-size sweep）

脚本 `scripts/bench_cache_sweep.py`。对单个数组大小从小往大扫，做四种流式
原语，测“等效逻辑带宽”：

- `read_sum`：读 N 字节
- `write_zero`：写 N 字节
- `copy`：读写各 N 字节（占用 2 份数据）
- `triad_add`：`y = x + y`，读 2N 写 N（占用 2 份数据）

每个计时样本内部循环若干次（`inner`，上限 512），使单样本逻辑流量
≥ 1 GiB，从而摊掉 kernel 启动开销。`torch.cuda.Event` 计时，25 个样本取中位数。

### 3.2 实验 B —— llama.cpp 真实 MUL_MAT 权重体量扫描

decode 的 GEMV 每次 token 都要把整份权重读一遍，是 IF Cache 最典型的受益者。
用 llama.cpp 自带 `test-backend-ops perf` 跑**真实的 ROCm MUL_MAT kernel**：

- 固定 `m = 4096`（输出行数 ⇒ kernel 并行度恒定，避免把“并行度变化”误当成
  “带宽变化”），`n = 1`（decode）与 `n = 16`（并发解码）。
- 扫 `k`（规约维）使权重体量 `|W| = m·k·bpe` 从 8 MiB 走到 512 MiB，跨越
  96 MiB。
- 类型 `f16 / q8_0 / q4_K / f32`；bpe = 2.0 / 1.0625 / 0.5625 / 4.0。

> **对 test-backend-ops 做了本地增补**：upstream 的 perf 形状是硬编码的有限
> 集合，无法扫体量。我们把上面的 `(m=4096, k, n)` 形状加进了
> `tests/test-backend-ops.cpp` 的 `make_test_cases_perf()`（原始文件备份为
> `scripts/test-backend-ops.cpp.orig`），单独重建 `test-backend-ops` 后运行，
> 跑完已还原。改动仅新增测试形状，不触碰 kernel。

### 3.3 实验 C —— 算子权重 footprint 映射

把 4B / 7B 每层主干算子的权重字节数与 96 MiB 对照（从模型 config 直接算，
不依赖模型文件），回答“哪些算子放得进 IF Cache”。

### 3.4 未能完成的部分：硬件计数器

原计划用 `rocprofv3 --pmc GL2C_HIT_sum GL2C_MISS_sum FETCH_SIZE WRITE_SIZE`
直接读命中率和 DRAM 流量。**实测该环境下 GL2C 块的所有计数器恒返回 0**
（用纯 HIP vector-add 和 torch 都验过；同时 `SQ_WAVES_sum` 等 SQ 计数器正常，
说明 profiler 本身在工作，是内存块计数器不可用）。这台机器上没有免密 root，
`perf_event_paranoid=4`，无法用提权绕过。

因此本报告**不使用 GL2C 命中率**，全部结论建立在**时间维度的工作集扫描**上
——即“同一 kernel 在跨过 96 MiB 边界时带宽断崖”这一直接观测量。这条路径不依赖
计数器，结论同样硬。

## 4. 实验 A 结果：带宽 vs 工作集

（图：`results/figures/cache_sweep.png`）

单位 GB/s（中位数）：

| 每次操作数据 (MiB) | read_sum | write_zero | copy | triad_add |
|---:|---:|---:|---:|---:|
| 8 | 565 | 960 | 1466 | 1622 |
| 16 | 920 | 1149 | 1685 | 2267 |
| 32 | 1471 | 1295 | 1825 | **2620** |
| 48 | 1313 | 1362 | 1464 | 2543 |
| 64 | 1750 | 1400 | 772 | 793 |
| 80 | **2083** | 1428 | 789 | 782 |
| 96 | 1997 | 1342 | 820 | 786 |
| 112 | **823** | 892 | 790 | 786 |
| 128 | 837 | 899 | 802 | 790 |
| 256 | 860 | 921 | 785 | 787 |
| 512 | 888 | 919 | 787 | 791 |
| 1024 | 750 | 932 | 776 | 801 |

读法：

- **read_sum（1 份数据）**：≤ 96 MiB 时 1.3–2.1 TB/s，≥ 112 MiB 后钉在
  ~0.82–0.89 TB/s。断点在 96→112 MiB，正好是 IF Cache 容量。
- **copy / triad（2 份数据）**：断点前移到 ~48→64 MiB（footprint = 2×size）。
  triad 在 32 MiB 达到 2620 GB/s 的有效带宽，是 IF 读写合计能力的体现。
- 断崖非常陡：read_sum 从 1997 直接掉到 823，**2.4 倍阶跃**。
- 大于 112 MiB 后所有曲线收敛到 0.78–0.93 TB/s，与独立 HBM microbenchmark
  （read_sum 874、write 933、copy 778、triad 801 GB/s @1024 MiB）一致 ⇒ 这一档
  就是纯 HBM。

> 注意：8–16 MiB 的 `read_sum` 偏低（565/920）不是带宽不够，而是**并行度不足**
> ——小数组喂不满 48 个 CU，被启动/尾部延迟主导。IF 的“满血”区间要从
> 32 MiB 往上看。

## 5. 实验 B 结果：真实 MUL_MAT 的权重带宽 vs 权重体量

（图：`results/figures/llama_mulmat_sweep.png`）

### 5.1 n=1（decode，权重主导）

| 权重体量 (MiB) | f16 | q8_0 | q4_K | f32 |
|---:|---:|---:|---:|---:|
| 8 | 85 | 36 | 53 | 90 |
| 16 | 281 | 206 | 553 | 175 |
| 32 | 1041 | 1160 | 865 | 543 |
| 48 | 1345 | 1645 | 972 | 1395 |
| 64 | 1512 | 1884 | **1023** | 1885 |
| 80 | **1617** | **2093** | 1008 | **2141** |
| 96 | 979 | 1426 | 925 | 1494 |
| 112 | 973 | 812 | 740 | 821 |
| 128 | 800 | 811 | 739 | 831 |
| 192 | 756 | 784 | 739 | 825 |
| 256 | 869 | 870 | 770 | 884 |
| 512 | 850 | 846 | 773 | 889 |

（单位 GB/s）

### 5.2 峰值 / 平台比

| 类型 | IF 内峰值 (GB/s) | 出现位置 (MiB) | >128 MiB 平台中位 (GB/s) | 比值 |
|---|---:|---:|---:|---:|
| f16 (n=1) | 1617 | 80 | 846 | 1.91× |
| q8_0 (n=1) | 2093 | 80 | 835 | 2.51× |
| f32 (n=1) | 2141 | 80 | 848 | 2.52× |
| q4_K (n=1) | 1023 | 64 | 747 | **1.37×** |
| f16 (n=16) | 979 | 64 | 571 | 1.72× |
| q8_0 (n=16) | 802 | 48 | 573 | 1.40× |
| q4_K (n=16) | 329 | 48 | 285 | 1.16× |
| f32 (n=16) | 538 | 64 | 250 | 2.15× |

要点：

1. **n=1 时 f16 / q8_0 / f32 明显受 IF 影响**：峰值都出现在 64–80 MiB，
   越过 96 MiB 后跌到 ~0.8 TB/s。峰值远高于 960 GB/s 标称 HBM，只有 IF 能解释。
2. **峰值在 ~80 MiB 而不是 96 MiB**：权重把 96 MiB 填满后，激活/输出/量化
   scale 没地方放，开始互相驱逐，所以“刚好 96 MiB”就已经在掉。
3. **Q4_K 是例外**：曲线平得多（1023→747）。INT4 的瓶颈是反量化与计算，
   访存不再是主约束，所以 IF 帮不上多少。**这也正是原报告“Q4_K n=1 ≈ 900 GB/s”
   的原因——它不是“HBM 打满”，而是“kernel 打不满 HBM”**。
4. **n=16 整体更低更平**（f16 峰值 979、q8_0 802、q4_K 329）：token 变多后
   权重被复用的收益下降（每字节权重承担 16× FLOPs），算子转向计算受限，
   权重带宽不再是关键路径。**所以“N=16 的 ridge 达成率”用 HBM ridge 去算
   会系统性偏低，用 IF 去算也不对——它本来就不是同一种受限模式。**

## 6. 实验 C 结果：算子权重 footprint vs 96 MiB

（图：`results/figures/weight_footprint.png`；表：`results/weight_footprint.csv`）

单层权重体量（MiB，BF16 / Q4_K）：

| 算子 | 4B BF16 | 4B Q4_K | 7B BF16 | 7B Q4_K |
|---|---:|---:|---:|---:|
| qkv_proj | 30.0 | 8.4 | 31.5 | 8.9 |
| o_proj | 20.0 | 5.6 | 24.5 | 6.9 |
| gate_up_proj | **95.0** | 26.7 | **259.0** | 72.8 |
| down_proj | 47.5 | 13.4 | **129.5** | 36.4 |
| lm_head | **741.9** | **208.7** | **1039.5** | **292.4** |
| 单层四主干合计 | 192.5 | 54.1 | **444.5** | **125.0** |

（4B 为 tied embedding，`lm_head` 与 embedding 共享存储，不额外计一遍。）

读法：

- **4B**：qkv / o / gate_up / down **四个算子的权重单独都放得进 96 MiB**
  （gate_up 95.0 MiB 正好贴上限）。Q4 下更是全部远小于 96 MiB。
  ⇒ 4B decode 的权重访存**大量命中 IF**，这是原报告“4B 等效带宽只有
  491–699 GB/s”需要重新解释的地方（见 §7 第 3 条）。
- **7B**：只有 qkv(31.5) / o(24.5) 放得下；gate_up(259) / down(129.5) /
  lm_head(1039.5) 全部放不下，是纯 HBM。
  ⇒ 7B decode 的权重流量是 IF 与 HBM **混合**的，不能用一个带宽数概括。
- 单层四主干合计：4B BF16 192.5 MiB、7B BF16 444.5 MiB，**任何一层都放不进**
  IF。所以 IF 只在“算子粒度”上有效，在“层粒度”上无效。

## 7. 对原报告 / 博客结论的修正建议

1. **§3.2 “n=1 权重带宽主导”**：必须区分两层。
   - Q8_0 n=1 的 1811 GB/s、F32 886 GB/s、Q4_K 900 GB/s 里，
     Q8_0/F32 呈现的是 **IF 带宽**（>960 就是铁证），Q4_K 呈现的是
     **kernel 计算/反量化上界**。
   - 建议表格加一列 “受限层”：IF / HBM / compute。

2. **§5 batch=1 / §6 并发**：
   - 7B FP16 端到端 797.6 GB/s 这类“权重流等效带宽”，在权重(14 GiB)远大于
     96 MiB 的情形下**确实是 HBM 有效带宽**，可以保留，但要写明它是
     “端到端等效”，并且**已经被 IF 命中悄悄抬高过一部分**（小算子命中）。
   - GPTQ-INT8 b1 只有 91 GB/s 之类，是 kernel/反量化问题，不要归因到带宽。

3. **§8 “模型是不是太小 / 0.5B、4B 测不出带宽”**：原因要补一条——
   小模型的算子权重小到能**整块驻留 IF**，所以测到的是 **IF 路径带宽**，
   本来就不该拿 HBM 峰值去衡量。0.5B/4B 的“低等效带宽”更多是
   **并行度不足 + kernel 效率**，不是 HBM 不够快。

4. **§6 ridge 达成率**：应给**区间**而不是单值——
   - 下界：假设全部走 HBM（960 GB/s）
   - 上界：假设算子权重命中 IF（~2 TB/s），激活仍走 HBM
   - 并注明 N=16 时算子已转向计算受限，带宽 ridge 不再是好的标尺。

5. **新增一段“IF Cache 为什么重要”**：把这 96 MiB 当成**独立的算存比档位**，
   给出 “IF ridge = 算力 ÷ ~2 TB/s” 与 “HBM ridge = 算力 ÷ 960 GB/s”
   两条 ridge 线，而不是一条。

## 8. 局限

- **无硬件命中率**：GL2C 计数器在本机不可用（见 §3.4），因此“命中率 %”是
  推断而非实测。所有 IF 相关结论都由**带宽断崖**支撑，证据强度等同于工作集
  扫描，但不等于逐 kernel 命中率。
- **n=1 小权重段（8–16 MiB）不可用于比较带宽**：并行度不足，数值偏低。
- 实验 B 用的是 patch 后的 `test-backend-ops`；形状是合成的，但 kernel 是
  模型实际使用的 kernel。真实模型还会叠加 attention / norm / 其它访存，
  端到端未命中率只会更高。
- IF 有效容量受其它驻留数据影响，实测“可用”约 80–96 MiB。

## 9. 产物与复现

```
scripts/bench_cache_sweep.py       # 实验 A
scripts/sweep_llama_mulmat.py      # 实验 B 驱动
scripts/test-backend-ops.cpp.orig  # 上游原始文件（备份）
scripts/analyze_ifcache.py         # 汇总、出图、CSV
results/cache_sweep.{json,csv}     # 实验 A 原始数据
results/llama_mulmat_sweep.{json,csv}  # 实验 B 原始数据
results/weight_footprint.csv       # 实验 C
results/ifcache_summary.json       # 关键派生数字
results/figures/*.png              # 三张图
logs/                              # 运行日志
```

复现：

```bash
cd experiments/ifcache_arith_7900xtx_20260924
PY=/home/luyzh/miniconda3/envs/torch/bin/python
OUT=$PWD/results/cache_sweep.json $PY scripts/bench_cache_sweep.py

# 实验 B 需要先给 test-backend-ops 打形状补丁（见 §3.2）并重建
OUT=$PWD/results/llama_mulmat_sweep.json $PY scripts/sweep_llama_mulmat.py

$PY scripts/analyze_ifcache.py
```
