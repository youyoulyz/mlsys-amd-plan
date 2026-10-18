# 09 · 同样 100 万字上下文，"记忆"占显存从 549 MB 压到 0.1 MB：三种压缩各压掉多少

> **一句话在做什么**：不读论文、不装 GPU，用 ~50 行 PyTorch（CPU 就能跑）把三种"对话记忆压缩"方法（MLA / CSA / HCA，来自 DeepSeek-V3/V4）自己搭一遍，实测同样上下文下占多少字节、算多少次 attention。100 万字（1M）上下文：不压 MLA 要 549 MB → 再加 CSA 30 MB → 再加 HCA **0.1 MB**——压了约 5000 倍。

## 实验导读

大模型对话时要把每个字的"注意力记忆"（KV cache）留在显存里，随对话长度线性涨——长上下文时它是显存杀手。DeepSeek 连出三代压缩（MLA → CSA → HCA），论文里都说"省 90%+"。这个实验把口号变成可复现的数字：每代到底压掉多少字节、多少计算。

跑完（或读完）你会知道：

- 三种压缩压的是不同维度：MLA 压"每个字记忆的宽度"（特征维），CSA/HCA 压"记多少个字"（序列维）——所以 CSA/HCA 的压缩比与上下文长短无关，是恒定比例；
- 为什么 1M 上下文下 HCA 能压到 0.1 MB（序列维每 128 个字合并成 1 个记忆条目）；
- attention 计算量（FLOPs）随压缩怎么变：CSA 把"看全部历史"变成"只看最相关的 top-k 个"，计算量随上下文长度加速下降（5.15e10 → 2.15e9 → 9.44e6 FLOPs）；
- "省 90%+"这种论文口号，怎么拆成"字节省了多少、计算省了多少、精度换来了什么"三笔账。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| MLA | DeepSeek-V3 的记忆压缩（压宽度） | d_c=256 |
| CSA / HCA | V4 的序列维压缩（每 4 字 / 每 128 字合并 1 个） | m=4 / m'=128 |
| 上下文长度 | 对话有多长 | 256 → 4096 实测，外推 1M |
| 精度 | 计算用的数 | FP16 |

**主题**：KVCache / 训练与推理系统
**硬件**：任意（**纯 PyTorch，可 CPU 跑**，无需 GPU / 自定义 kernel）— 全档位通用
**强化**：第 3 周（显存决定方案）
**状态**：✅ 本机已验证（2026-10-17，RX 7900 XTX 上跑通，数字与上游一致，日志在 `results/local_run.log`）

## 你会怎么猜

KV cache 随上下文线性增长，是长上下文推理的显存杀手。MLA（DeepSeek-V3）已经用低秩投影把 KV 压到 d_c + d_rope 维度。V4 又加了 CSA（沿序列每 m=4 token 压成 1 个）和 HCA（m'=128）。你的直觉：

- "再压一次，收益应该边际递减。"
- 或者："这些压缩会不会把精度压没了？"

## 实测会看到

自建的 `nn.Module` 实现（scaled `d_model=1024`，32 heads，d_c=256），seq 256–4096，FP16：

**KV 体积（每 token：MLA = 576 元素，CSA/HCA 按 m 压缩）：**

| 序列 | V3 MLA | V4 CSA | V4 HCA | CSA 降幅 | HCA 降幅 |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.14 MB | 0.03 MB | 0.00 MB | 77.8% | 98.6% |
| 4096 | 2.25 MB | 0.50 MB | 0.03 MB | 77.8% | 98.6% |

**外推到 1M 上下文：**

| | V3 MLA | V4 CSA | V4 HCA |
| --- | --- | --- | --- |
| KV 体积 | ~549 MB | **~30 MB（−94.4%）** | **~0.1 MB（−99.9%）** |

**attention FLOPs（seq=4096）：**

| | V3 MLA | V4 CSA | V4 HCA |
| --- | --- | --- | --- |
| FLOPs | 5.15e10 | 2.15e9（**23.99×**） | 9.44e6（**5461×**） |

**反直觉**：降幅在 256→4096 之间**几乎恒定**（CSA 稳定 77.8%、HCA 稳定 98.6%）——因为 CSA/HCA 是**沿序列维**压缩，比例与序列长无关；而 1M 外推时 HCA 把 KV 压到 0.1 MB，比直觉的"边际递减"凶得多。CSA 把 attention 从 O(seq²) 降到 O(seq × top-k)，所以 FLOPs 降幅随序列长**加速**（256 时 5.99× → 4096 时 23.99×）。

**一句话结论**：用 ~50 行 PyTorch 就能把"KV 压缩省多少"从论文口号变成可复现的数字：同样 1M 上下文，KV 从 549 MB 压到 0.1 MB（~5000×），attention FLOPs 从 5e10 压到 1e7 量级。

## 为什么会这样

- **MLA**：只在**特征维**压缩（低秩，d_c=512 + d_rope=64 = 576 元素/token），不压序列维。
- **CSA**：沿**序列维**每 m=4 token 聚成 1 个压缩条目，再配 top-k 稀疏（Lightning Indexer），把 attention 从全序列降到 top-k。
- **HCA**：m'=128，更激进的序列维压缩，KV 几乎压没。

序列维压缩的比例与序列长无关（所以降幅恒定），而它让 attention 的有效长度变短，FLOPs 降幅随序列长放大。

## 用了什么工具，为什么

- **为什么自己用 PyTorch 搭一遍而不是直接跑 DeepSeek**：跑 V3/V4 要 H 卡 + 几百 GB 显存 + 定制 kernel。本 lab 要回答的是"压缩省多少"这个量级问题，用 ~50 行的 `nn.Module` 把三种方法的形状（宽度压到多少、序列维每几个字合并 1 个）搭出来，CPU 上几分钟就能出数字——把论文口号变成可复现、可改参数重跑的计算。
- **为什么不测延迟**：本 lab 的结论是"字节体积 + FLOPs"这两个量（它们决定显存需求和计算需求），延迟依赖具体硬件和 kernel 实现，不在纯 PyTorch 参考实现的可比范围内。
- **MLA / CSA / HCA 是什么**：三种让"注意力记忆"更省空间的架构设计——MLA 把每个字的记忆从"多个向量"压成"一个低秩向量"；CSA/HCA 把连续若干个字的记忆合并成一个（HCA 更激进，每 128 字才留 1 个）。

## 跑法

```bash
# 纯 PyTorch，可 CPU 跑（脚本内置 device=cpu 回退）
python code/benchmark_kv_cache_efficiency.py     # KV 体积 + 延迟 + 理论 FLOPs + 1M 外推
python code/visualize_kv_cache_benchmark.py      # 生成 PNG 到 results/images/

# 注：上游仓库另有 benchmark_kv_cache.py（torch.cuda 门控，需 GPU），本 lab 只复制了可 CPU 跑的主脚本
```
依赖：`torch`、`numpy`（可视化另需 `matplotlib`）。无 flash-attention / triton / 自定义 kernel。
本地 `results/` 已含结果表（`BENCHMARK_README.md`）、生成图片（`results/images/`）与本机运行日志（`results/local_run.log`，2026-10-17 在 RX 7900 XTX 上跑通，输出数字与上游一致）。

## 来源

- 脚本：`/mnt/luyuzhou/hpc/dsv4/benchmark_kv_cache_efficiency.py`（717 行，主脚本）
- 说明 + 结果表：`/mnt/luyuzhou/hpc/dsv4/BENCHMARK_README.md`
- 可视化：`/mnt/luyuzhou/hpc/dsv4/visualize_kv_cache_benchmark.py`、`images/`
- 背景报告：`/mnt/luyuzhou/hpc/dsv4/DeepSeek_V4_AI_Infra_Report.md`（1M 上下文 KV 10%、单 token FLOPs 27%/10% 的出处）

## 延伸

- 第 3 周阅读：Kwon 等，"PagedAttention"（2023）——KV 的显存管理视角。
- KV 装不下时"存盘 vs 重算"的决策见 [10 · KV 搬运 vs 重算](10-kv-transfer-vs-recompute/)。