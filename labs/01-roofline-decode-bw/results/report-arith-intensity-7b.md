# RX 7900 XTX Qwen2.5-7B 推理算存比实验报告

- 实验日期：2026-09-24
- GPU：AMD Radeon RX 7900 XTX，24 GiB，`gfx1100`，RDNA3，48 CU，96 MiB Infinity Cache
- 模型：**Qwen2.5-7B-Instruct**（`hidden_size=3584`，`intermediate_size=18944`，`num_layers=28`，`vocab_size=152064`，`Q heads=28`，`KV heads=4`，`head_dim=128`，`tie_word_embeddings=false`）
- llama.cpp：`/opt/llamacpp/llama.cpp/build-gfx1100-rocm10`，commit `1bc7a5af0`，build `10973`
- vLLM：`0.27.1.dev5+gf46a9dfe2.d20260827`（Docker），PyTorch `2.12.0+rocm10.0.0`，HIP `7.15.26333`
- 实验目录：`experiments/arith_intensity_7b_7900xtx_20260924`
- 独立并发报告：`VLLM_CONCURRENCY_REPORT.md`；并发原始数据：`results/vllm_concurrency/`
- 4B/0.5B 旧模型的删除清单：`logs/deleted_4b_0.5b_models_manifest.txt`（本轮为腾出磁盘删掉 4B、0.5B 权重共 67.184 GiB；被删的 4B/0.5B **报告与 CSV 作为对照保留**）

## 导出结论（先看这里）

1. **7B 才是能把 7900 XTX 权重流喂饱的量级。** llama.cpp FP16 decode 的有效权重读带宽为 **`797.6 GB/s`**，是本次 HBM 单流读 microbenchmark 中位数 `874.2 GB/s` 的 **91.2%**、官方 960 GB/s 的 **83.1%**。作为对照，4B 同口径只有 `699.2 GB/s`（官方 72.8%），0.5B 的 vLLM FP16 只有约 `90.8 GB/s`。**模型太小的顾虑被证实**：0.5B 完全测不出这张卡的权重带宽，4B 明显偏小，7B 才进入“权重流主导、接近带宽上限”的区间。
2. **但单请求 decode 依然远没有进入计算受限区。** 所有成功路径的单请求逻辑算存比只有 `0.50–3.24 OP/B`，相对理论 ridge（FP32 `63.96`、FP16/BF16/INT8 `128.125`、INT4 `256.25` OP/B）低两个数量级。原因不是模型不够大，而是 **decode 的 batch=1 形状决定了 AI ≈ 1/位宽**，与模型尺寸无关（详见 §8）。
3. **量化提升逻辑 AI，但实测性能取决于 kernel。** vLLM batch=1 中 GPTQ-INT4 达到 `3.153 OP/B`、`94.12 tok/s`，是最快的量化路径；AWQ-INT4 的 AI 几乎相同（`3.156 OP/B`），吞吐却只有 `55.55 tok/s`（ROCm 上 AWQ 走的是整块反量化 Triton 路径，未走融合 W4A16 kernel）；GPTQ-INT8 最慢，`11.76 tok/s`。
4. **并发把 AI 拉高，但 N=16 仍在 ridge 的 12%–23%。** vLLM 的 FP16 / GPTQ-INT4 / AWQ-INT4 / GPTQ-INT8 在 `N=16` 分别达到 `16.00 / 50.44 / 50.49 / 29.11 OP/B`，对应理论 ridge 的 `12.49% / 19.68% / 19.70% / 22.72%`。N=8→16 的 batch efficiency 落到 `0.41–0.66`，权重共享收益已被 attention/KV、调度和固定开销吃掉。
5. **FP32 在 7B 上不可运行，FP8 在 gfx1100 上不可用。** FP32 权重 `28.28 GB`（GGUF 文件 `30.47 GB`）超过 24 GiB 显存，llama.cpp 与 vLLM 全部加载失败；vLLM 的 FP8-dynamic 在加载期调用 `torch._scaled_mm`，报错只支持 CUDA SM 8.9/9.0 或 ROCm MI300+，全部失败。两者都不填任何编造的实测数字。
6. **逻辑 FLOPs 是算法口径，不等于原生低精度指令数。** Q8/Q4、AWQ/GPTQ 都是 weight-only/混合路径，激活或累加仍可能是 FP16；本报告不会把 Q4 的 `246 TOPS` 规格直接当成实际 kernel throughput。

> **口径说明**：`LOGICAL_MATMUL_PARAMS = V*H + L*(2*H*Q_HEADS*HEAD_DIM + 2*H*KV_HEADS*HEAD_DIM + 3*H*I) = 7,070,285,824`，`TOKEN_MATMUL_FLOPS = 2 * LOGICAL_MATMUL_PARAMS = 14,140,571,648`。Qwen2.5-7B 的 Q/O 投影宽度为 `28*128 = 3584`（恰等于 hidden_size），KV 宽度为 `4*128 = 512`。`tie_word_embeddings=false`，因此输出头 `lm_head.weight`（`152064×3584`）与输入 embedding 表是两张独立的表。

---

## 1. 指标定义与实验边界

### 1.1 逻辑算存比

本文统一采用 roofline 的逻辑口径：

```text
有用 GEMM FLOPs       = 2 * m * n * k
单 token 逻辑 FLOPs  = TOKEN_MATMUL_FLOPS = 14,140,571,648
权重-only AI          = FLOPs / 每 token 必读权重字节
理论 ridge            = 硬件峰值算力 / 理论 HBM 带宽
有效权重读带宽        = 每 token 权重字节 * 每秒 token 数
```

decode 阶段同一 batch 内多个请求共享一次权重读取，所以 batch=N 的逻辑 AI 写成：

```text
AI(N) = N * TOKEN_MATMUL_FLOPS / per_token_weight_bytes
```

`per_token_weight_bytes` 的口径是 **transformer 权重 + 输出投影表读一次**：decode 时输入 embedding 只做一次单行查表（可忽略），输出投影则是完整的大矩阵乘（`[1,H]×[H,V]`），必须完整读一次。Qwen2.5-7B 的五个 HF checkpoint 都在文件里同时含 `model.embed_tokens.weight` 与独立的 `lm_head.weight`，因此：

```text
per_token_weight_bytes = (全部 tensor 字节) − (输入 embedding 表字节)
                       = transformer 权重 + 输出投影表（恰好一次）
```

llama.cpp 的 GGUF 统计（`token_embd` 不计、`output.weight` 计一次）与 vLLM 侧 safetensors 统计使用**同一条规则**，两边结果可对齐：FP16 在 llama.cpp 与 vLLM 的 per-token 权重都是 `14.14 GB`、权重-only AI 都是 `1.000 OP/B`。对量化权重，本文仍按“一次乘加 = 两个逻辑 FLOP”计数，不把 Q8/Q4 的数值解释成原生 INT8/INT4 指令数。

### 1.2 三层数据

1. **llama.cpp 算子级**：固定 `m=4096, k=14336`，测 `n=1` 和 `n=512` 的 `MUL_MAT`。
2. **llama.cpp 模型级**：`llama-bench` 分别测 prompt 512 与 decode 128。
3. **vLLM 模型级**：`vllm bench latency`，input 512、output 128；测 batch=1、batch=8，以及并发 N=1/2/4/8/16。

除 HBM microbenchmark 外，本文**没有读取硬件 DRAM counter**。所谓“实测”是由端到端时间/吞吐和逻辑权重字节反推的**等效权重读带宽**，不是 profiler 的物理 DRAM traffic——两者不可混称。

---

## 2. 理论算力、带宽和 ridge point

采用 AMD RX 7900 XTX 产品页规格：HBM 960 GB/s，FP32 vector 61.4 TFLOPS，FP16 matrix 123 TFLOPS，INT8 matrix 123 TOPS，INT4 matrix 246 TOPS；BF16 没有单列，按 FP16 matrix 等效。

| 精度 | 官方峰值 | 理论 HBM | 理论 ridge | 说明 |
|---|---:|---:|---:|---|
| FP32 | 61.4 TFLOPS | 960 GB/s | **63.96 OP/B** | FP32 vector |
| FP16 | 123 TFLOPS | 960 GB/s | **128.125 OP/B** | FP16 matrix |
| BF16 | 123 TFLOPS | 960 GB/s | **128.125 OP/B** | 无单列规格，按 FP16 等效 |
| INT8 / Q8 | 123 TOPS | 960 GB/s | **128.125 OP/B** | 官方 INT8 matrix |
| INT4 / Q4 | 246 TOPS | 960 GB/s | **256.25 OP/B** | 官方 INT4 matrix |
| FP8 | 无可用原生峰值 | 960 GB/s | 不适用 | `gfx1100` 无原生 FP8 WMMA，vLLM 路径失败 |

本次 1024 MiB microbenchmark 的单流读带宽中位数为 `874.23 GB/s`。若把它当作保守 HBM 分母，则 ridge 参考值变为：

| 精度 | 按 874.23 GB/s 的保守 ridge |
|---|---:|
| FP32 | 70.24 OP/B |
| FP16 / BF16 / INT8 | 140.70 OP/B |
| INT4 | 281.39 OP/B |

主表仍使用官方 `960 GB/s`：硬件规格和实际可持续带宽是不同口径，microbenchmark 只用于解释“实测离规格还有多远”。

---

## 3. llama.cpp 算子级实验

### 3.1 运行方法

```bash
/opt/llamacpp/llama.cpp/build-gfx1100-rocm10/bin/test-backend-ops \
  perf -b ROCm0 -o MUL_MAT \
  -p 'type_a=(f32|f16|bf16|q8_0|q4_K),type_b=f32,m=4096,n=1,k=14336'
```

n=1 接近 decode 的权重带宽主导形状；n=512 接近 prompt/prefill 的计算密集形状。Q8_0 权重字节按 `34 B / 32 weights` 计算，Q4_K 按 `144 B / 256 weights` 计算。

### 3.2 n=1：权重带宽主导

| 类型 | 权重 | 时间 | 逻辑算力 | 逻辑 AI | 表观权重读带宽 | 相对 960 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| F32 | 224.0 MiB | 265.00 us | 443.16 GFLOPS | 0.500 | 886.30 GB/s | 92.3% |
| F16 | 112.0 MiB | 144.07 us | 815.16 GFLOPS | 1.000 | 815.16 GB/s | 84.9% |
| BF16 | 112.0 MiB | 144.52 us | 812.61 GFLOPS | 1.000 | 812.61 GB/s | 84.6% |
| Q8_0 | 59.5 MiB | 34.45 us | 3410.0 GFLOPS | 1.882 | 1810.97 GB/s | 188.6% |
| Q4_K | 31.5 MiB | 36.68 us | 3200.0 GFLOPS | 3.556 | 900.45 GB/s | 93.8% |

**关键 caveat：**这是重复执行同一个权重矩阵的微基准。Q8_0 的 `62.4 MB` 和 Q4_K 的 `33.0 MB` 工作集都能整体放进 96 MiB Infinity Cache，因此 Q8_0 的“表观权重读带宽”可以超过 HBM 960 GB/s。它不能被解释为 HBM 流量。F32 的 224 MiB、F16/BF16 的 112 MiB 工作集远大于或接近 Infinity Cache，与 HBM 的相关性更强。

### 3.3 n=512：计算更密集，但峰值利用率仍不高

| 类型 | 时间 | 逻辑算力 | 逻辑 AI | 对应峰值 | 峰值利用率 |
|---|---:|---:|---:|---:|---:|
| F32 | 2585.15 us | 23.26 TFLOPS | 256.0 OP/B | 61.4 TFLOPS | 37.9% |
| F16 | 946.92 us | 63.50 TFLOPS | 512.0 OP/B | 123 TFLOPS | 51.6% |
| BF16 | 879.77 us | 68.35 TFLOPS | 512.0 OP/B | 123 TFLOPS | 55.6% |
| Q8_0 | 1003.72 us | 59.91 TFLOPS | 963.8 OP/B | 123 TOPS | 48.7% |
| Q4_K | 1044.34 us | 57.58 TFLOPS | 1820.4 OP/B | 246 TOPS | 23.4% |

n=512 时 AI 已远高于对应 ridge，工作负载在理想 roofline 上进入计算受限区；但实际峰值利用率只有 `23.4%–55.6%`。Q4_K 尤其说明“逻辑 4-bit AI 很高”不等于“原生 INT4 峰值跑满”，因为实际路径包含解包、反量化和 FP32 输入处理。

![llama.cpp 算子 roofline](results/common/figures/operator_roofline.png)

---

## 4. llama.cpp 模型级实验

### 4.1 运行方法

五个 GGUF 都使用：

```bash
/opt/llamacpp/llama.cpp/build-gfx1100-rocm10/bin/llama-bench \
  -m <gguf> -r 5 -p 512 -n 128 -ngl 99 -b 2048 -ub 512 -fa on -o json
```

### 4.2 存储、算存比与吞吐

| 精度 | 文件 MiB | 每 token 权重 MiB | 实际 bit/逻辑参数 | 权重-only AI | prompt 512 tok/s | decode 128 tok/s | 有效权重读带宽 | 逻辑 TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 29056.95 | 26972.27 | 32.0016 | 0.49998 | **超显存，失败** | **超显存，失败** | — | — |
| FP16 | 14531.95 | 13486.77 | 16.0016 | 0.99991 | 4457.67 | 56.40 | 797.55 GB/s | 0.7975 |
| BF16 | 14531.95 | 13486.77 | 16.0016 | 0.99991 | 4443.42 | 56.07 | 792.99 GB/s | 0.7929 |
| Q8_0 | 7723.36 | 7165.44 | 8.5016 | 1.88202 | 4458.35 | 93.00 | 698.75 GB/s | 1.3151 |
| Q4_K_M | 4466.13 | 4168.09 | 4.9453 | 3.23541 | 4192.18 | 128.90 | 563.38 GB/s | 1.8228 |

**FP32 是实测失败样本，不是缺测。** FP32 GGUF 文件 `30,468,420,160 B = 28.28 GiB`（tensor 数据 `28.28 GB`），远超 24 GiB 显存；`llama-bench` 在加载阶段直接退出：

```text
llama_bench: error: failed to load model
  '.../models/gguf/qwen2.5-7b-instruct-f32.gguf'
```

因此本表只保留它的**存储统计与理论算存比**（AI `0.49998 OP/B`），decode/prompt 吞吐留空，不外推、不补数。若要拿到 7B FP32 decode 数据，需要 CPU/多卡分层或换更大显存的卡。

对 decode，按 960 GB/s 的理想权重流上限分别为：

| 精度 | 960 GB/s 理论 token/s 上限 | 实测 decode tok/s | 达到理论上限 |
|---|---:|---:|---:|
| FP32 | — | — | — |
| FP16 | 67.88 | 56.40 | 83.1% |
| BF16 | 67.88 | 56.07 | 82.6% |
| Q8_0 | 127.77 | 93.00 | 72.8% |
| Q4_K_M | 219.65 | 128.90 | 58.7% |

观察：

- **FP16/BF16 已经把权重流跑到 797.6 / 793.0 GB/s。** 相对微基准实测读带宽 `874.23 GB/s` 已达 **91.2%**，相对官方 `960 GB/s` 也有 **83.1%**——这是本实验“7B 够大、能喂满带宽”的核心证据。
- **Q8_0、Q4_K_M 的 token/s 更高，但没有按压缩比提升。** 相对 FP16，Q4_K_M 权重约 3.23 倍小、decode 快约 2.29 倍（128.90 vs 56.40）；剩下的是反量化和混合 kernel 开销，有效带宽因此降到 698.7 / 563.4 GB/s。
- **prompt 512 吞吐远高于 decode**（FP16 `4458 tok/s` vs `56.4 tok/s`），因为 prefill 的 GEMM 形状有更高复用；它不能与 decode tok/s 混为同一个指标。

![模型级 AI 与理论 ridge](results/common/figures/model_ai_vs_ridge.png)

---

## 5. vLLM batch=1 与 batch=8

### 5.1 运行方法

所有 vLLM 测试都在 Docker 内执行，包装脚本 `scripts/run_vllm_in_docker.sh`，等效命令：

```bash
vllm bench latency \
  --model <model> --input-len 512 --output-len 128 \
  --batch-size <1|8> --num-iters 5 --num-iters-warmup 2 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager
```

`--enforce-eager` 关闭了 torch.compile/CUDAGraph。端到端延迟包含 prefill 512、额外输出头以及 128 个输出 token 的生成；因此这里的 tok/s 是整段 benchmark 的等效吞吐，不是纯 decode kernel 速率。

### 5.2 batch=1

| 路径 | 每 token 权重 MiB | 延迟 s | 总 tok/s | 权重-only AI | 等效权重读带宽 | 逻辑 TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 13486.14 | 2.176 | 58.82 | 1.000 | 831.85 GB/s | 0.8318 |
| FP16 | 13486.14 | 2.174 | 58.88 | 1.000 | 832.67 GB/s | 0.8326 |
| FP32 | 26972.27 | 失败 | — | — | — | — |
| GPTQ-INT4 | 4277.50 | 1.360 | 94.12 | 3.153 | 422.17 GB/s | 1.3310 |
| AWQ-INT4 | 4273.18 | 2.304 | 55.55 | 3.156 | 248.91 GB/s | 0.7855 |
| GPTQ-INT8 | 7413.31 | 10.887 | 11.76 | 1.819 | 91.39 GB/s | 0.1663 |
| FP8-dynamic | 7265.79 | 失败 | — | — | — | — |

### 5.3 batch=8

| 路径 | 延迟 s | 总 tok/s | 每请求 tok/s | batch-scaled AI | 等效每请求权重带宽 | 逻辑 TFLOPS | AI/ridge |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 3.553 | 288.17 | 36.02 | 8.000 | 509.39 GB/s | 4.0749 | 6.24% |
| FP16 | 3.228 | 317.25 | 39.66 | 8.000 | 560.79 GB/s | 4.4861 | 6.24% |
| FP32 | 失败 | — | — | — | — | — | — |
| GPTQ-INT4 | 2.108 | 485.80 | 60.72 | 25.221 | 272.37 GB/s | 6.8695 | 9.84% |
| AWQ-INT4 | 3.036 | 337.24 | 42.16 | 25.247 | 188.89 GB/s | 4.7688 | 9.85% |
| GPTQ-INT8 | 13.514 | 75.77 | 9.47 | 14.553 | 73.64 GB/s | 1.0715 | 11.36% |
| FP8-dynamic | 失败 | — | — | — | — | — | — |

观察：

- **GPTQ-INT4 是 7B 上最强的量化路径**：batch=1 延迟 `1.360 s`，batch=8 延迟 `2.108 s`，总吞吐 `485.80 tok/s`。
- **AWQ-INT4 没有把 W4A16 的优势兑现**：AI 与 GPTQ-INT4 基本相同，但 batch=1 慢 69%、batch=8 慢 44%。当前 ROCm 路径走 Triton/整块反量化，而不是 RDNA3 融合 W4A16 kernel。
- **GPTQ-INT8 最慢**：权重只比 FP16 小约 45%，但 batch=1 延迟是 FP16 的 5.0 倍，说明 `ConchLinearKernel` 实现开销主导。
- **FP16/BF16 batch=1 已把权重流打到 832 GB/s**，甚至略高于 llama.cpp decode 的 797.6 GB/s。两者 benchmark 边界不同（vLLM 含 prefill 512 与 5 次迭代平均），不能直接当作同一 kernel 的胜负。
- **batch=8 的 AI 提升来自权重共享，不是硬件吞吐自动变成 8 倍。** 总吞吐约提升 `3.6–5.2` 倍，明显次线性。

---

## 6. vLLM 并发 N=1/2/4/8/16 的访存比

本节数据来自独立静态并发扫描 `results/vllm_concurrency/vllm_concurrency.csv`，每个 N 用独立进程运行；完整解释、图和 caveat 见 `VLLM_CONCURRENCY_REPORT.md`。

### 6.1 权重-only AI（OP/B）

| 路径 | N=1 | N=2 | N=4 | N=8 | N=16 | 理论 ridge | N=16 达成度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 1.000 | 2.000 | 4.000 | 8.000 | 15.999 | 128.125 | 12.49% |
| GPTQ-INT4 | 3.153 | 6.305 | 12.611 | 25.221 | 50.443 | 256.25 | 19.68% |
| AWQ-INT4 | 3.156 | 6.312 | 12.623 | 25.247 | 50.494 | 256.25 | 19.70% |
| GPTQ-INT8 | 1.819 | 3.638 | 7.276 | 14.553 | 29.105 | 128.125 | 22.72% |
| FP8 | 失败 | 失败 | 失败 | 失败 | 失败 | 不适用 | 不适用 |

### 6.2 Byte/FLOP（1/AI）

| 路径 | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
| FP16 | 1.0000 | 0.5000 | 0.2500 | 0.1250 | 0.0625 |
| GPTQ-INT4 | 0.3172 | 0.1586 | 0.0793 | 0.0396 | 0.0198 |
| AWQ-INT4 | 0.3169 | 0.1584 | 0.0792 | 0.0396 | 0.0198 |
| GPTQ-INT8 | 0.5497 | 0.2749 | 0.1374 | 0.0687 | 0.0344 |

### 6.3 聚合吞吐（tok/s）

| 路径 | N=1 | N=2 | N=4 | N=8 | N=16 | N=16/N=1 | batch efficiency N=16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 59.02 | 111.53 | 205.54 | 317.75 | 525.71 | 8.91x | 0.557 |
| GPTQ-INT4 | 91.65 | 171.15 | 293.89 | 477.65 | 602.36 | 6.57x | 0.411 |
| AWQ-INT4 | 55.03 | 105.30 | 194.81 | 336.59 | 527.72 | 9.59x | 0.599 |
| GPTQ-INT8 | 11.71 | 22.56 | 42.24 | 75.28 | 124.10 | 10.59x | 0.662 |

并发趋势的核心结论：

- AI 随 N 近似严格线性，因为模型级口径假设同一 batch 内一次权重读取服务所有请求；这是 roofline 理想值，不是 profiler 实测 DRAM 字节。
- N≤4 的吞吐放大接近线性；N=8→16 明显放缓，说明 attention/KV、调度和 kernel 效率开始吃掉权重共享收益。
- **7B 的 batch efficiency 比 4B 更低**（FP16 N=16：`0.557` vs 4B `0.573`；GPTQ-INT4：`0.411` vs 4B `0.649`）。7B 每 token 权重、KV cache 和 attention 开销都更大，N=16 时 KV/attention 已与权重流争夺带宽和显存。
- 即使 N=16，所有路径仍在 ridge 的 `12%–23%`，没有一条进入真正的计算受限区。

---

## 7. 实测 vs 理论总表

| 精度路径 | 理论峰值 | 理论 ridge | llama.cpp n=1 AI | llama.cpp n=512 峰值利用率 | llama.cpp decode AI | vLLM b1 AI | vLLM b8 AI | vLLM N=16 AI | N=16/ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 61.4 TFLOPS | 63.96 | 0.500 | 37.9% | 超显存 | 超显存 | 超显存 | 未扫 | 不适用 |
| FP16 | 123 TFLOPS | 128.125 | 1.000 | 51.6% | 1.000 | 1.000 | 8.000 | 15.999 | 12.49% |
| BF16 | 123 TFLOPS | 128.125 | 1.000 | 55.6% | 1.000 | 1.000 | 8.000 | 未扫 | 6.24%* |
| INT8 / Q8 | 123 TOPS | 128.125 | 1.882 | 48.7% | 1.882 | 1.819 | 14.553 | 29.105 | 22.72% |
| INT4 / Q4 | 246 TOPS | 256.25 | 3.556 | 23.4% | 3.235 | 3.153 | 25.221 | 50.443 | 19.68% |
| AWQ-INT4 | 246 TOPS | 256.25 | — | — | — | 3.156 | 25.247 | 50.494 | 19.70% |
| FP8 | 不适用 | 不适用 | 不适用 | 失败 | 不适用 | 失败 | 失败 | 失败 | 不适用 |

\* BF16 与 FP32 的 N=16 未做并发扫描；BF16 一栏为 batch=8 的 AI/ridge。FP32 三个模型级口径全部因为 `28.28 GB > 24 GiB` 无法运行。

这张表最需要注意的是：

- **decode 阶段的 AI 比 n=512 算子小几百倍。** n=512 的 GEMM 可以进入计算受限区（峰值利用率 `23%–56%`），但端到端 decode 的 batch 和小序列形状远远不够。
- **Q4/Q8 的高 n=512 AI 是逻辑量，不等于原生低精度利用率。** Q4_K 峰值利用率只有 `23.4%`，Q8_0 为 `48.7%`。
- **理论峰值是上限，不是当前 kernel 的实际上限。** llama.cpp FP16/BF16 decode 已经跑到实测 HBM 读带宽的 ~91%，而 Q8/Q4 的端到端吞吐受反量化和调度约束，只能到 72.8% / 58.7%。
- **逻辑 AI 与模型尺寸几乎无关。** 表中 FP16 的 AI 在 0.5B、4B、7B 都是 `≈1.000`：per-token 权重字节与 per-token FLOPs 同比例随模型增大。模型变大改变的是**绝对吞吐和带宽利用率**，不是 AI 本身。

---

## 8. 7B 与 4B / 0.5B 的对照：模型是不是太小

用户的核心疑问是“是不是模型太小了、测不出这张卡的能力”。本轮把 4B/0.5B 权重删除、换成 7B 重做，直接对照如下。

| 指标 | Qwen2.5-0.5B | Qwen3-4B | Qwen2.5-7B | 7B/4B |
|---|---:|---:|---:|---:|
| 逻辑矩阵参数量 | 0.499B | 4.022B | 7.070B | 1.76x |
| 每 token 逻辑 FLOPs | 0.999 GFLOP | 8.045 GFLOP | 14.141 GFLOP | 1.76x |
| FP16 每 token 权重 MiB | 942.3 | 7672.6 | 13486.8 | 1.76x |
| llama.cpp FP16 decode tok/s | 未测同口径 | 86.91 | 56.40 | 0.65x |
| llama.cpp FP16 有效权重读带宽 | 未测同口径 | 699.2 GB/s | **797.6 GB/s** | 1.14x |
| llama.cpp FP16 占官方 960 GB/s | — | 72.8% | **83.1%** | +10.3 pt |
| vLLM FP16 b1 有效权重读带宽 | **90.8 GB/s** | 491.4 GB/s | **834.6 GB/s** | 1.70x |
| vLLM FP16 b1 tok/s | — | 62.22 | 58.88 | 0.95x |
| vLLM FP16 N=16 聚合吞吐 | 1061 tok/s | 560.3 tok/s | 525.7 tok/s | 0.94x |
| vLLM GPTQ-INT4 N=16 聚合吞吐 | 1449 tok/s | 685.1 tok/s | 602.4 tok/s | 0.88x |
| vLLM FP16 AI @N=1 / N=16 | 1.011 / 16.176 | 1.000 / 15.999 | 1.000 / 15.999 | 基本不变 |
| vLLM GPTQ-INT4 AI @N=1 / N=16 | 2.175 / 34.799 | 3.013 / 48.211 | 3.153 / 50.443 | 1.05x |
| vLLM AWQ-INT4 AI @N=1 / N=16 | 1.367 / 21.878 | 3.018 / 48.281 | 3.156 / 50.494 | 1.05x |
| vLLM GPTQ-INT8 AI @N=1 / N=16 | 1.562 / 24.989 | 1.787 / 28.599 | 1.819 / 29.105 | 1.02x |
| FP16 batch efficiency @N=16 | 0.722 | 0.573 | 0.557 | 0.97x |

结论：

1. **0.5B 太小、4B 偏小、7B 才够。** 0.5B 的 FP16 有效权重读带宽只有 `90.8 GB/s`，4B 升到 `500.5 GB/s`（vLLM）/`699.2 GB/s`（llama.cpp decode），7B 进一步升到 `834.6 GB/s`（vLLM b1）/`797.6 GB/s`（llama.cpp decode）。7B 的 llama.cpp decode 已达微基准实测读带宽的 **91.2%**，说明权重流已经成为主导瓶颈。
2. **"AI 随模型变大而升高"是误解。** FP16 的 AI 在 0.5B→4B→7B 基本恒为 `≈1.000 OP/B`，因为 per-token 权重字节与 per-token FLOPs 同比例增长。AWQ/GPTQ 的小幅上升来自“embedding 不量化 + 输出头口径”被更大的模型摊薄，而不是模型尺寸本身提高 AI。
3. **模型变大提高带宽利用率，但不提高绝对吞吐。** 7B 每 token 权重/FLOP 是 4B 的 1.76 倍，N=16 聚合吞吐只有 4B 的 `0.88–0.94` 倍。所以 7B 的收益是“测得更准”，代价是“吞吐更低、KV 更紧”。
4. **不能把 7B 的结果外推成 70B 的绝对吞吐。** 7B 足够回答“这张卡权重流能跑多快”，但更大模型需要重新做形状、KV cache 和 batch 扫描。

---

## 9. FP8 与 FP32：两个“如实失败”的样本

### 9.1 FP8-dynamic

FP8-dynamic checkpoint 的磁盘统计：

- tensor 总量：`7,618,731,008 B = 7265.79 MiB`
- 类型：`BF16 2,183,437,312 B + F8_E4M3 6,525,288,448 B`
- 统一 per-token 口径：`7265.79 MiB`（该 checkpoint 有独立 `lm_head.weight`）
- 理论等价位宽：`8.6206 bit/逻辑参数`

vLLM 失败发生在模型加载阶段：

```text
Selected ChannelWiseTorchFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
RuntimeError: torch._scaled_mm is only supported on CUDA devices with
compute capability >= 9.0 or 8.9, or ROCm MI300+
```

因此：

- gfx1100/RDNA3 不在该 FP8 路径的支持列表；
- vLLM batch=1、batch=8、并发 N=1/2/4/8/16 全部 RC=1；
- llama.cpp 本轮也没有生成可用的 FP8 `MUL_MAT` 数据；
- FP8 的理论 ridge、实测算存比和端到端吞吐均记为“不适用/失败”，不做模拟补数。

### 9.2 FP32

- GGUF 文件 `30,468,420,160 B = 28.28 GiB`，per-token 权重 `26972.27 MiB`，理论权重-only AI `0.49998 OP/B`；
- llama.cpp：`llama_bench: error: failed to load model`，因为 28.28 GB 权重装不进 24 GiB 显存；
- vLLM：`base_fp32`（BF16 checkpoint 上转 FP32，`26972.27 MiB`）引擎初始化失败，RC=1；
- 结论：**7B 的 FP32 推理在单张 7900 XTX 上不可行**，本报告只保留其存储与理论算存比。

---

## 10. HBM 带宽 microbenchmark

方法：PyTorch `torch.cuda.Event`，256/512/1024 MiB 等档位，每档 warmup 和 30 次计时；`copy` 按读写总字节计，`triad_add` 按 2 read + 1 write 计。下表取 1024 MiB：

| 操作 | 逻辑字节 | 中位带宽 | 960 GB/s 利用率 |
|---|---:|---:|---:|
| `read_sum` | 1 GiB read | **874.23 GB/s** | 91.1% |
| `write_zero` | 1 GiB write | 933.38 GB/s | 97.2% |
| `copy` | 1 GiB read + 1 GiB write | 778.43 GB/s | 81.1% |
| `triad_add` | 2 GiB read + 1 GiB write | 801.34 GB/s | 83.5% |

实践含义：

- 单流读实际可持续 `874.23 GB/s`，不是 960 GB/s。用实测值当分母，FP16/BF16/INT8 的保守 ridge 约 `140.7 OP/B`，INT4 约 `281.4 OP/B`，比官方口径更难达到。
- llama.cpp FP16 decode 的 `797.6 GB/s` 已经达到 microbenchmark 读带宽的 **91.2%**；vLLM FP16 b1 的 `834.6 GB/s` 达 **95.5%**。这说明 7B 的 decode 确实被权重流主导。
- Q8_0/Q4_K 的表观带宽低于 FP16，是因为权重变小后反量化 kernel 与固定开销占比升高，不再是纯带宽问题。

---

## 11. 适用边界和 caveat

1. **"实测"= 端到端时间反推的等效权重读带宽，不是 DRAM counter。** 本文没有 `rocprof`/hw counter，不能声称“实际从 HBM 读了这么多字节”。
2. **逻辑 FLOPs ≠ 原生低精度指令数。** Q8/Q4/AWQ/GPTQ 是 weight-only，激活仍可能是 FP16；逻辑 AI 仅用于和 roofline ridge 对齐。
3. **vLLM 含 prefill。** `vllm bench latency` 的延迟包含 512 prefill + 128 decode，tok/s 是整段等效吞吐；llama.cpp decode 是纯 decode。两者不要直接比 kernel 速度。
4. **`--enforce-eager`** 关闭了 CUDAGraph/compile，测的是 eager 路径的保守数字。
5. **microbenchmark 的重复访存会被 96 MiB Infinity Cache 放大**，Q8_0/Q4_K 的小工作集尤其明显；跨 cache 的 F32/F16/BF16 才更接近 HBM。
6. **per-token 权重口径**：transformer 权重 + 输出投影表一次，输入 embedding 查表不计。7B 的 5 个 checkpoint 都含独立 `lm_head.weight`，所以 `per_token = 全部 tensor − 输入 embedding 表`。
7. **FP32 不可运行是被显存限制卡住的**，不是 kernel 问题；换卡或 CPU 分层才能测。
8. **并发 N=1/2/4/8/16 是本实验的静态 batch 口径**，不是真实在线流量的到达率。

---

## 12. 产物和复现入口

### 数据

| 文件 | 内容 |
|---|---|
| `results/common/theoretical_roofline.csv` | 官方峰值与理论 ridge |
| `results/common/hbm_bandwidth.csv` | HBM microbenchmark 全档位 |
| `results/common/operator_perf.csv` | llama.cpp `MUL_MAT` n=1 / n=512 |
| `results/common/llamacpp_model.csv` | llama.cpp 模型级（含 FP32 失败行） |
| `results/common/vllm_model_storage.csv` | vLLM checkpoint 存储与 bpw |
| `results/common/vllm_latency.csv` | vLLM batch=1/8 延迟（含失败行） |
| `results/common/analysis_summary.json` | 汇总元数据 |
| `results/common/figures/operator_roofline.png` | 算子级 roofline 图 |
| `results/common/figures/model_ai_vs_ridge.png` | 模型级 AI vs ridge 图 |
| `results/vllm_concurrency/vllm_concurrency.csv` | 并发扫描 25 行原始结果 |
| `results/vllm_concurrency/vllm_concurrency_summary.json` | 并发扫描汇总 |

### 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/download_models.sh` / `download_remaining.sh` | 下载 5 个 7B HF checkpoint |
| `scripts/run_convert.sh` | HF → GGUF f16/bf16/f32 |
| `scripts/run_quantize.sh` | fp16 GGUF → Q8_0 / Q4_K_M |
| `scripts/run_llamacpp.sh` | llama.cpp 模型级 `llama-bench` |
| `scripts/bench_hbm_bandwidth.py` | HBM microbenchmark |
| `scripts/run_vllm_in_docker.sh` | Docker 内跑 vLLM bench |
| `scripts/run_vllm_batch1.sh` / `run_vllm_matrix.sh` | vLLM batch=1 / batch=8 矩阵 |
| `scripts/run_vllm_concurrency.sh` | 并发 N=1/2/4/8/16 扫描 |
| `scripts/analyze_arith_intensity.py` | 生成 common CSV/图/汇总 |
| `scripts/analyze_vllm_concurrency.py` | 生成并发 CSV/图/汇总 |
| `scripts/check_vllm_concurrency_consistency.py` | 并发数据自校验（275 项） |
| `scripts/check_report_numbers.py` | 报告数字与 CSV 交叉校验 |

### 一键复核

```bash
PY=/home/luyzh/miniconda3/envs/torch/bin/python
$PY scripts/analyze_arith_intensity.py
$PY scripts/analyze_vllm_concurrency.py
$PY scripts/check_vllm_concurrency_consistency.py
$PY scripts/check_report_numbers.py
```

### 理论规格参考

- AMD Radeon RX 7900 XTX 产品页：HBM 960 GB/s、FP32 61 TFLOPS、FP16/INT8 123 TFLOPS、INT4 246 TOPS。
- 本次 microbenchmark 单流读中位数：`874.23 GB/s`（1024 MiB）。
