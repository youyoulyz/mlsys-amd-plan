# 01 · 一张卡跑 7B 大模型生成文字：算力只用 1–3%，跟模型多大无关

> **一句话在做什么**：在一张 24 GB 游戏显卡（RX 7900 XTX）上，让 7B 大模型一个字一个字往外蹦（单用户、一次只算 1 个人），测两件事——每个字要从显存读多少字节模型权重、做多少次计算，以及这张卡的理论平衡点在哪。实测：**每个字只做 0.5–3.2 次计算/字节，卡的平衡点是 128 次/字节**——差 40–250 倍，所以这张卡的算力（FP16 峰值 ~123 TFLOPS，每秒 123 万亿次）只用到 1–3%。

## 实验导读

大模型"生成"回答是一字一字往外蹦的，每一步都要把整个模型的权重从显存读一遍。显卡的卖点是算力（每秒多少次计算），这个实验回答：单用户场景下，这点算力到底被用上了多少，为什么用不上。

跑完（或读完）你会知道：

- 为什么"模型越大、越能吃满显卡算力"是错的——瓶颈在"一次只算一行"的形状，不在模型参数量；
- 怎么算"算存比"（每读 1 字节做几次计算），以及这张卡的理论平衡点在哪；
- 为什么只有把多个用户合在一起算（加 batch）才能提高算力利用率，而且收益会被摊薄；
- 量化（INT4/INT8）为什么能加快生成——它把权重读小了，但改不了"形状"。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| 7B 模型 | 参数量 70 亿（16 bit 精度下约 15 GB 显存） | Qwen2.5-7B |
| batch=1 | 一次只服务 1 个用户 | 1 |
| 算存比 | 每读 1 字节权重做几次计算（越低越偏"搬数据"） | 0.5–3.24 |
| 平衡点（ridge） | 卡的算力 ÷ 带宽，算存比低于它就算力用不满 | 128（FP16）/ 256（INT4） |
| 对照模型 | 换更小/更大模型验证"与大小无关" | 0.5B / 4B |

**主题**：Batch-Decode / 性能与测量
**硬件**：AMD RX 7900 XTX（24 GiB 显存，48 个计算单元，96 MiB 片上缓存）· Qwen2.5-7B — plan **T2** 档
**强化**：第 1 周（估算与瓶颈）、第 2 周（时间都去哪了）
**状态**：实测已有，可直接读；脚本可移植到 T4 重跑

## 你会怎么猜

decode 就是把每个 token 的权重读一遍做矩阵乘。直觉上：

- 模型越大 → 单 token 算得越多 → 应该越接近算力上限（compute-bound）。
- 或者：卡算力高 → 利用率应该高。

所以你会猜：**7B 比 0.5B 更接近算力 roof，大模型才"喂饱"高算力卡。**

## 实测会看到

把单请求 decode 的**逻辑算存比（arithmetic intensity，AI）**算出来（AI = 每 token 有用 FLOPs ÷ 每 token 必读权重字节）：

- **7B 单请求 decode 的 AI 只有 0.50–3.24 OP/B**。
- 这张卡的理论 ridge（AI 的分水岭）：FP16/BF16/INT8 ≈ **128.1** OP/B，INT4 ≈ **256.2** OP/B。
- 也就是说，单请求 decode 离 ridge **差一到两个数量级**——它死死钉在**访存受限**一侧。

关键反直觉：

| 事实 | 数字 |
| --- | --- |
| 7B decode 的有效权重读带宽 | **797.6 GB/s** = 实测 HBM 中位数 874.2 GB/s 的 **91%**、官方 960 的 **83%** —— 权重流确实接近带宽上限 |
| 但单请求 AI | 0.50–3.24 OP/B，离 ridge 差 40–250× |
| 为什么 | **batch=1 的形状决定了 AI ≈ 1/位宽**，与模型尺寸无关 |
| 模型太小测不出带宽 | 0.5B 只有 ~90.8 GB/s，4B 699.2 GB/s，7B 才把 7900 XTX 的权重流喂到 91% |
| 提 batch 才拉高 AI | N=16 时 FP16 才到 16.0 OP/B = ridge 的 **12.5%**；GPTQ-INT4 到 50.4 OP/B = ridge 的 **19.7%**；N=8→16 的 batch efficiency 掉到 0.41–0.66 |
| 量化提升逻辑 AI 但实测看 kernel | GPTQ-INT4 **94.1 tok/s** 最快；AWQ-INT4 逻辑 AI 几乎相同（3.156 vs 3.153 OP/B）却只有 **55.6 tok/s**（ROCm 上 AWQ 走整块反量化 Triton，没走融合 W4A16 kernel）；GPTQ-INT8 最慢 11.8 tok/s |
| 硬约束 | FP32 权重 28.28 GB 超 24 GiB 装不下；FP8 在 gfx1100 上不可用 |

**一句话结论**：7B 才把这张卡的**带宽**喂饱，但 decode 的算力利用率始终是 1–3%。"用大模型喂饱高算力卡"这个直觉，在 batch=1 的 decode 上是错的——瓶颈是形状，不是模型大小。

## 为什么会这样

decode 每生成一个 token，要把全部权重从显存读一遍，但只对 **1 行**激活做矩阵乘。所以每读 1 字节权重只做 ~1/位宽 次浮点——AI 被 batch size 卡死在极低位置，跟模型参数量无关。要拉高 AI 只能加大 batch（共享一次权重读取算多个请求），但加 batch 又把权重共享的收益分给 attention/KV/调度/固定开销吃掉（batch efficiency 掉到 0.41–0.66）。这就是为什么"batch 越大越好"也不成立（见 lab 03）。

## 用了什么工具，为什么

- **为什么用 llama.cpp 和 vLLM 两套引擎测**：llama.cpp 是开源的通用推理引擎（CPU/各种显卡都能跑），vLLM 是工业界部署大模型最常用的推理引擎。同一件事两套引擎各测一遍，是为了确认"算力利用率低"不是某个引擎的实现问题，而是"一次只算一行"这个任务形状本身的性质。
- **为什么先单独测显卡的显存带宽**：算"利用率"需要分母——这张卡每秒最多能从显存读多少数据（官方 960 GB/s）。分母不实测，"读到了 91%"这种结论无从谈起。
- **Roofline / 算存比是什么**：Roofline 是一张"性能天花板"图：横轴是算存比（每搬 1 字节做几次计算），纵轴是你能达到的计算速度。工作点落在左侧 = 被"搬数据"卡住，落在右侧 = 被算力卡住。单用户生成永远钉死在左侧，跑什么模型都一样。

## 跑法

本地 `code/` 是关键脚本，`results/` 是原始数据与完整报告：

```bash
# HBM 带宽 microbenchmark（roofline 的分母）：
bash code/bench_hbm.sh                # 或 python code/bench_hbm_bandwidth.py

# decode 带宽 / AI 采集（需 Qwen2.5-7B GGUF + ROCm 构建）：
bash code/run_llamacpp.sh             # llama.cpp 路径
bash code/run_vllm_batch1.sh          # vLLM batch=1
bash code/run_vllm_concurrency.sh     # vLLM batch=8/16（扫 AI(N)）

# 分析与报告数字校验：
python code/analyze_arith_intensity.py
python code/check_report_numbers.py
```
原始结果在 `results/`（`llamacpp/` `vllm/` `vllm_batch1/` `vllm_concurrency/` `common/`），完整报告是 `results/report-arith-intensity-7b.md`。在 T4（Instinct）上重跑同一套口径，对比 7900 XTX 的 ridge 与达成率，就是一份第 2 周的校准作业。

## 来源

- 报告：`/mnt/luyuzhou/hpc/moecho/results/reports/ARITH_INTENSITY_QWEN2.5_7B_7900XTX_20260924.md`
- 配套 4B/0.5B 对照：`ARITH_INTENSITY_QWEN3_4B_7900XTX_20260923.md`
- 实验目录：`/mnt/luyuzhou/hpc/moecho/experiments/arith_intensity_7b_7900xtx_20260924/`
- 环境：llama.cpp `build-gfx1100-rocm10`（commit `1bc7a5af`）；vLLM `0.27.1`；PyTorch `2.12.0+rocm10.0.0`

## 延伸

- Roofline 原典：Williams 等，"Roofline: An Insightful Visual Performance Model"（2009）——plan 第 1 周阅读。
- 下一站：为什么"等效带宽"会突然超过 HBM 峰值？见 [02 · Infinity Cache 悬崖](02-infinity-cache-cliff/)。