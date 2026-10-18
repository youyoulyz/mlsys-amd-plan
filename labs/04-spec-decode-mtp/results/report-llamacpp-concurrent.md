# llama.cpp 长上下文并发 Decode 与 Prefix KV Cache 实验报告

日期：2026-09-08  
环境：`/mnt/luyuzhou/hpc/moecho`

## 1. 实验目的

本实验针对本地 llama.cpp API 在 OpenCode 场景下的两个问题进行测量：

1. 两个约 40K token 的请求并发生成时，是否能有效进行 decode batching，以及实际吞吐是否仍接近此前观察到的 60–80 tok/s。
2. 两个请求是否真正共享公共 prompt 的 KV cache，而不是仅仅因为 slot 的 `n_ctx` 容量相同而产生误判。

## 2. 服务配置

服务端为：

```text
/opt/llama.cpp/llama.cpp/build/bin/llama-server
-ngl 99 -sm tensor -c 500000 --parallel 4
--spec-type draft-mtp --spec-draft-n-max 2
-fa on -ctk q4_0 -ctv q4_0 --kv-unified
```

OpenCode 通过以下 API 访问：

```text
http://127.0.0.1:8080/v1
model: qwen3.8-27b
```

运行时 `/slots` 显示 4 个 slot，每个 slot 的实际 `n_ctx` 为 262,144。命令行中的 `-c 500000` 没有把模型上下文上限提升到 500K。

## 3. 实验一：双请求、约 700 output tokens

### 3.1 方法

- 同时提交两个约 39,666 token 的长 prompt。
- 两个 prompt 具有公共前缀，但末尾请求标识不同。
- 每个请求要求生成约 700 tokens。
- 记录服务端 `prompt eval`、`eval`、MTP acceptance 和总耗时。

### 3.2 结果

| slot | prompt tokens | prompt prefill | output tokens | decode 速度 | MTP acceptance | MTP mean len |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 39,669 | 36.31 s | 700 | 12.62 tok/s | 0.49218 | 1.98 |
| 3 | 39,669 | 72.05 s | 700 | 34.83 tok/s | 0.51971 | 2.04 |

两个请求约在 93.5 秒内完成。prefill 阶段存在明显竞争：一个 slot 的 prompt 处理约 36 秒，另一个约 72 秒；decode 速度也出现 12.6/34.8 tok/s 的明显不均衡。

### 3.3 观察

- 700 token 的短输出不足以摊薄 40K prompt prefill 成本，因此端到端延迟主要由长 prompt 处理决定。
- 并发请求没有表现为两个独立 slot 各自维持单请求速度；GPU 计算和 KV/显存带宽在 prefill 阶段发生竞争。
- MTP acceptance 约 49–52%，低于长输出测试，说明短请求下统计样本较少，且 prefill/调度干扰对观测值影响更大。

## 4. 实验二：双请求、约 7000 output tokens

### 4.1 方法

- 使用与实验一相同规模的两个约 39,666 token prompt。
- 两请求并发生成，各要求 7000 tokens。
- 记录完整生成期间的 slot timing、decode 吞吐和 MTP 指标。

### 4.2 结果

| slot | prompt eval | output tokens | decode 时间 | decode 速度 | MTP acceptance | MTP mean len |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 513 tokens（缓存后缀） | 7,000 | 171.25 s | 40.87 tok/s | 0.71481 | 2.43 |
| 1 | 513 tokens（缓存后缀） | 7,000 | 183.25 s | 38.19 tok/s | 0.60954 | 2.22 |

端到端耗时约为 175.9 s 和 187.9 s。由两个请求的 decode 速度相加得到：

```text
40.87 + 38.19 ≈ 79.06 tok/s
```

端到端合计吞吐约为：

```text
14,000 / 187.9 ≈ 74.5 tok/s
```

### 4.3 观察

- 两请求并发时，合计 decode 吞吐约 79 tok/s，端到端吞吐约 74.5 tok/s，基本回到此前记忆的 60–80 tok/s 区间。
- 单个请求的速度约 38–41 tok/s；并发后不是简单的 2 倍单请求速度，而是共享同一 GPU 计算/带宽资源后的总吞吐。
- 相比 700 token 实验，长输出显著摊薄了 prompt prefill 成本，因此吞吐表现稳定得多。
- MTP acceptance 分别为 71.5% 和 61.0%，平均 draft 长度为 2.43 和 2.22；MTP 正在工作，但没有阻止多 slot 的 decode batching。

## 5. Prefix KV Cache 对照验证（A/B/C）

为了排除“slot 大小相同”这一误判，另外进行了严格的顺序对照：

- A：全新公共前缀 + 后缀 A；
- B：与 A 完全相同的公共前缀，仅修改末尾后缀；
- C：完全不同的新公共前缀 + 后缀 C。

每个 prompt 约 40K tokens，仅生成 64 tokens。

| 请求 | prompt tokens | `cached_tokens` | 请求耗时 |
|---|---:|---:|---:|
| A | 40,042 | 0 | 38.5 s |
| B | 40,042 | 39,526 | 1.9 s |
| C | 40,043 | 0 | 34.9 s |

服务端在 B 请求中记录：

```text
selected slot by LCP similarity, f_sim_best = 1.000, f_keep = 0.998
prompt eval ... 516 tokens
```

而 A/C 都执行了完整约 40K token 的 prompt eval。因此可以确认：

1. B 复用了 A 的公共 token 前缀，命中约 39.5K tokens；
2. C 没有误命中 A 的缓存；
3. 这是按 token 前缀进行的逻辑 KV/prompt-cache reuse，而不是 slot 容量相同造成的假象。

C 到来时还出现了：

```text
removing oldest entry (size = 944.243 MiB)
removing oldest entry (size = 2112.031 MiB)
```

说明 prompt cache 容量有限，新的长前缀会触发旧条目淘汰。

## 6. 综合结论

1. **并发长输出 decode 是有效的。** 两个 7000-token 请求的合计 decode 吞吐约 79 tok/s，端到端约 74.5 tok/s；MTP 没有占据或禁用多请求 batching。
2. **短输出请求看起来“很慢”主要是长 prompt prefill 竞争。** 在 700-token 实验中，约 40K prompt 的处理时间远大于 decode 时间，并发导致两个 slot 的 prefill 时间明显不均衡。
3. **公共前缀确实被缓存复用。** API 的 `cached_tokens=39,526`、LCP 日志和 B 请求仅处理 516 个新 token 三者相互吻合。
4. **KV cache 存在淘汰。** 8 GiB prompt-cache 限制会在多个长前缀并存时触发 eviction；因此 OpenCode 如果频繁改变长 prompt，可能反复承担 prefill 成本。
5. **当前证据证明了逻辑上的 prefix cache reuse，但不单独证明底层一定是同一物理 KV 地址。** llama.cpp 可以通过引用、复制或 allocator 管理实现该复用；对性能而言，API 命中数和省去的 prefill 已足以证明复用生效。

## 7. 复核指标

运行中的请求可用以下命令观察实际缓存 token 数：

```bash
curl -s http://127.0.0.1:8080/slots | jq '.[] | select(.is_processing) |
  {id, n_prompt_tokens, n_prompt_tokens_processed, n_prompt_tokens_cache}'
```

请求完成后 slot 的缓存计数可能归零；最终判断应优先使用 API 返回的 `usage.prompt_tokens_details.cached_tokens` 和服务端 LCP/prompt-eval 日志。
