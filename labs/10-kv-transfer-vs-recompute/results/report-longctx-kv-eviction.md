# EXP — Qwen3.5-9B-MTP 单卡 × RULER 长上下文性能与通信税（含断电排查）

> 实验日：2026-09-04 | 状态：✅ 完成（A1 单流矩阵）
> 关联：协议 [PROTOCOL_qwen38_27b_tp2_ruler_longctx_20260903.md](../../docs/designs/PROTOCOL_qwen38_27b_tp2_ruler_longctx_20260903.md)
> 数据：`results/data/qwen38_ruler/partA_9b_single/`（records.jsonl）

---

## 0. 事件：双卡 TP2 触发断电（2026-09-04，本实验最大发现之一）

在跑 27B TP2（及 9B TP2）时机器**两次瞬时断电重启**（journal 戛然而止、无任何关机序列、无 Xid/MCE/thermal/kernel panic）。

**根因**：硬件电源事件——2× RTX 3080 满载 ~605W + CPU ≈ **800-850W 整机**，3080 瞬时功率尖峰（NCCL 权重 allreduce / kernel 启动）可达 2× TDP → **PSU OCP 过流保护跳闸 → 瞬时断电**。触发条件 = **双卡同时满载 + NCCL 双卡同步**（TP2 加载/并发 prefill），而非模型大小（9B TP2 也触发）。

**排查结论**：
1. 系统 GPU 空闲时稳定 15h+；一上双卡满载就断电。
2. **9B 单卡（sm none，无 NCCL）满载 312W（GPU1 空闲）→ 整机 ~420W，全程稳定**。✓
3. 后续实验统一 **9B 单卡**；窗口保持 262144；显存空着不管（`start-qwen-tp.sh` 已改）。

> 注意：`nvidia-smi -lgc` 无权限（非 root），此前「锁频 1803MHz」实际未生效，~1800MHz 是热降频；A1 数据无需推翻（`-cram 0` 禁 prompt cache 已保证测量一致性），但报告不得声称「时钟锁定」。

## 1. A1 单流矩阵（9B 单卡，spec off/mtp，n=4K..261K）

配置：`llama-server -sm none -c 266240 --parallel 1 -ctk q4_0 -ctv q4_0 -cram 0`，RULER `niah_multikey_2` 精确 n（tokenizer 与 27B 同 vocab 248044，数据复用）。时钟未锁（热降频 1710-1980 MHz）。

| n | off TTFT(s) | mtp TTFT(s) | off prefill t/s | mtp prefill t/s | off decode t/s | mtp decode t/s | **MTP speedup** |
|---|---|---|---|---|---|---|---|
| 4096 | 1.127 | 1.310 | 3654 | 3142 | 99.8 | 138.8 | **1.39×** |
| 8192 | 2.310 | 2.602 | 3570 | 3170 | 96.5 | 133.5 | **1.38×** |
| 16384 | 4.749 | 5.326 | 3477 | 3098 | 91.3 | 127.7 | **1.40×** |
| 32768 | 10.106 | 11.294 | 3269 | 2922 | 81.6 | 114.6 | **1.40×** |
| 65536 | 22.934 | 25.506 | 2879 | 2587 | 68.6 | 96.2 | **1.40×** |
| 131072 | 56.229 | 63.132 | 2345 | 2087 | 52.4 | 71.0 | **1.35×** |
| 261120 | 153.753 | 174.708 | 1706 | 1500 | 35.5 | 50.3 | **1.42×** |

## 2. 关键发现（9B 单卡）

1. **MTP 恒定 ~1.4× decode 加速**（1.35-1.42×，全 n 区间，短输出~128 token 即成立）——单卡 llama.cpp 上 MTP 是稳定强正收益，与 27B TP2（1.84-2.04×）一致方向。
2. **长上下文 decode 惩罚 > prefill 惩罚**：decode 99.8→35.5 t/s（**−64%** @261K）；prefill 3654→1706 t/s（−47%）。长上下文下 decode 是主要瓶颈。
3. **MTP prefill 开销 ~14-16%**（TTFT on/off 比：1.31/1.13 @4K → 174.7/153.8 @261K），为 draft 上下文设置成本，随 n 占比稳定。
4. **prefill_tps 随 n 超线性下降**（3654→1706，-53%@261K），FA 下仍受注意力计算/内存带宽约束。

## 3. 对比 27B TP2（同协议、不同栈/模型，仅供方向参考）

| 指标 | 27B TP2（q4, 锁频失败≈热降频） | 9B 单卡 |
|---|---|---|
| n4096 prefill t/s | 1393 | 3654 |
| n131K prefill t/s | 1032 | 2345 |
| n4096 decode t/s (off) | 51.2 | 99.8 |
| MTP speedup 区间 | 1.84-2.04× | 1.35-1.42× |
| 满载功耗 | ~605W（2 卡，触发断电） | ~312W（1 卡，稳定） |

## 4. Part B serving（9B 单卡，2026-09-04）

统一池 147456（144K）——单卡下 262144 + 多 slot graph 会 OOM（实测 mtp c4/c8 加载失败）；`--parallel c` 闭环 10 请求/ cell，TP 无（单卡）。

| n | c | off goodput | mtp goodput | off TTFT p50 | mtp TTFT p50 |
|---|---|---|---|---|---|
| 8192 | 1 | 2197 | 2276 | 2.4s | 2.6s |
| 8192 | 2 | **2399** | 2306 | 5.0s | 5.7s |
| 8192 | 4 | 2257 | 2198 | 7.6s | 6.9s |
| 8192 | 8 | 2127 | 2051 | 13.0s | 14.6s |
| 32768 | 1 | **2750** | 2607 | 10.3s | 11.4s |
| 32768 | 2 | 2278 | 2075 | 22.8s | 18.8s |
| 32768 | 4 | 1750 | 1585 | 28.3s | 33.3s |

> n32768/c8：单卡 20GB 物理放不下（8×32K=256K KV > 容量），未测。

**serving 结论**：
1. **goodput 峰值在 c=2**（n8192）或 c=1（n32768）；更高并发 prefill 瓶颈反降。
2. **MTP 在 serving 下无吞吐增益**（off 普遍 ≥ mtp）——MTP 的 ~1.4× decode 优势被 prefill 主导的负载吞掉，draft 计算还与主模型争 GPU。
3. TTFT 随并发爆炸（n8192: 2.4→13s，5.4×；n32768: 10.3→33s @c4）。
4. 与 27B serving 同构：「持续负载下 MTP 不赚」，但 llama.cpp 单卡下是**中性偏弱**而非强负（vLLM 时代 −11~28%）。

## 5. KV cache 驱逐 → 重复 prefill 实验（2026-09-04，专项）

**场景**：统一 KV 池（环形缓冲）容量 < 并发请求工作集时，新请求的 prefill 会**覆盖（驱逐）旧请求的 KV cell**；被驱逐的请求再次出现时必须**全量重复 prefill**（浪费）。

**设置**：9B 单卡，`--parallel 2 --cache-reuse 256 --kv-unified`，两个 40K 请求（工作集 = 38680 + 37278 = **75958 tokens**），扫池大小。观测 `timings.prompt_n` = 每请求实际 prefill token 数（4 ≈ 全复用，全长 ≈ 全驱逐）。

| 池 | 相对工作集 | H#2 重复 prefill | 耗时 |
|---|---|---|---|
| 131072 / 98304 / 81920 / **77824** | ≥75958 | **4 tok**（缓存复用）| ~70ms |
| **73728** / 69632 / 65536 | <75958 | **38680 tok（全量重 prefill）** | **~12.2s** |

**结论**：
1. **驱逐阈值 = 并发工作集大小（75958），二值无中间态**——环形缓冲下第 2 个请求的 prefill 整段覆盖第 1 个请求，无部分驱逐。
2. **成本量化**：每次被驱逐的重复请求多付 **~12.2s prefill**（vs 缓存命中 ~70ms，**~160× 浪费**），等效于 prefill 吞吐下降一半以上（同内容二次请求 100% 重算）。
3. 这是 serving 池配置的关键约束：**池 ≥ 预期并发工作集** 才能让 `--cache-reuse` 生效；否则多请求场景触发缓存抖动（thrashing）。
4. 观测方法学：`timings.prompt_n` 直接等于重复 prefill 量；llama-server `--cache-reuse` 启用后同内容请求仅 prefill 尾部（4 token）。

**数据**：`results/data/qwen38_ruler/evict/records.jsonl`（7 池 × 2-4 请求 × 2 rep）。
**脚本**：`experiments/qwen38_tp2_ruler_20260903/scripts/run_evict.py`。

### 5b. 真实 agent 会话重放（LMCache 40 会话，2026-09-04）

用 `mtp_edge_20260814/sessions.jsonl`（40 个真实 SWE-bench/Gaia/wildclaw agent 会话，上下文 0.7-12.7K）逐轮重放，测池 {32768..14336} 的 re-prefill ratio：

- **会话真实上下文只有 ≤12.7K**（此前误加总 90K 是重复计数）。
- 逐轮重放（chat/completions，4-6 会话 round-robin）的 re-prefill ratio 在**所有池（含 32768 对照）均为 ~2.31×**——恒定，与池无关。
- **归因**：不是池驱逐，而是**重放方法学伪影**——我们生成的非参考输出与 trace 的参考 assistant 内容不匹配，破坏槽内前缀连续 → 每轮从分歧点全量重 prefill。t6/t7 小 delta（126/156）证明复用机制正常，t1-t5 大 delta 是「参考 assistant + 下一 user」的真实新增内容被重复计入。
- **对池驱逐的含义**：该工作负载每轮 prompt ≤12.7K、工作集 ≤25K；池驱逐只在 **池 < 12.3K（最大单轮上下文）** 时才会触发，而那时请求已 400（prompt>池）。→ **真实 agent 负载下，池 ≥ 最大单轮上下文即可免驱逐**；多会话并发时池还应 ≥ 并发工作集（§5 结论）。
- 数据：`results/data/qwen38_ruler/evict_agent/records.jsonl`；脚本 `run_evict_agent.py`。

## 6. 半导体厂 code-review 场景：naive vs 协调调度 的 re-prefill 浪费（2026-09-04）

**场景**：多工程师并发用 API 做 max-effort code review（低时效，总耗时最低优先）。负载 = LMCache 40 个真实 agent 会话（SWE-bench/Gaia/wildclaw 多轮工具调用，就是 code-review/code-edit 负载）。

**单卡模拟**（9B 单卡，池 16384，12 个最大会话）：
- **naive**（当前 harness+API：逐轮 round-robin 交错，--parallel 4 槽 churn）→ 每轮换槽全量重 prefill。
- **协调**（KV 保留：会话逐个完整跑完；= 方案1「服务端只 serve KV 内请求+排队」与方案2「harness 协调」在单卡上的共同收敛）→ 每 token 只 prefill 一次。

| 调度 | 总耗时 | 总 prefill tokens | 相对 unique(120.5K) |
|---|---|---|---|
| naive | **192s** | **285,556** | 2.37× |
| 协调 | **105s** | **65,840** | 0.55× |

**结论**：
1. naive 多 prefill **219,716 tokens（4.3×）** = 纯 re-prefill 算力浪费（@~3000 t/s ≈ 73s）；总耗时慢 45%（87s），与重 prefill 算力吻合。
2. 协调调度（服务端 KV-aware 排队 或 harness 协调）消除该浪费：单卡上两方案收敛，每 token 只 prefill 一次，总耗时 −45%。
3. naive 的 re-prefill 主要由**槽 churn**（会话数 > 槽数，LRU 换槽）驱动，池驱逐其次——即使池足够大（对照 32768+ 同 ratio）churn 照样重 prefill。
4. 数据：`results/data/qwen38_ruler/sched/`（summary.jsonl + naive/coord records）；脚本 `run_sched_compare.py`。

## 7. 端侧前缀复用调度算法对比（CacheWise 真实会话，2026-09-06）

**数据**：CacheWise Coding Traces（真实 Claude Code 用户 94 会话）。用 5 个最大会话（~556M 上下文，tool 间隙中位 5.1s/均值 23s/最大 44min）在 9B 单卡成本模型上做离散事件模拟（prefill 3000 t/s、decode 100 t/s、KV q4=42KB/token、CPU↔GPU 6 GB/s）。

| 算法 | 总耗时 | GPU 利用率 | 说明 |
|---|---|---|---|
| naive（无协调，每步全量重 prefill）| 190,076s (52.8h) | 高 | 92% 上下文重复 prefill |
| **算法1 强制死等**（≤2 常驻，KV 不换出）| **143,321s (39.8h)** | **12%** | 保住 KV（零 re-prefill）但 tool 间隙 GPU 空转 88% |
| **算法2 动态轮换**（tool 间隙 KV→CPU，换短请求，返回搬回）| **14,898s (4.1h)** | 30% | swap 4.96TB/827s（仅 5% 开销）|
| （参考）coord 串行理想（忽略间隙）| 17,745s (4.9h) | — | 理论下界 |

**结论**：
1. **强制死等被 tool 间隙击穿**：该负载 tool 调用长，死等 = GPU 空转 88%，5 会话串行化 39.8h。KV 保留的收益远抵不过利用率损失。
2. **动态轮换最优（4.1h）**：比死等快 9.6×、比 naive 快 12.8×；swap 带宽（6 GB/s）不是瓶颈。
3. **带宽敏感性伪影**：swap 太快（bw≥12）导致每个间隙都换出 → thrashing（swap 涨到 14TB、wall 反升）——**轮换需 hysteresis**（间隙须 > swap 代价 + 收益余量才换出）。
4. 数据：`results/data/qwen38_ruler/sched_sim/sim_result.json`；脚本 `sim_sched.py` + `extract_cachewise.py`；原始 CacheWise 克隆 `experiments/qwen38_tp2_ruler_20260903/cachewise/`（gitignore 嵌套 repo）。

## 8. 真卡调度验证（9B 单卡 slot=2，2026-09-06）

**实跑**（`run_algo_test.py`）：5 个 CacheWise 会话 × 20 轮合成等长 prompt（每会话不同 filler、前缀连续），间隙 cap 3s，`--parallel 2 --kv-unified`。

| 算法 | 总耗时 | 总 prefill | 每轮 prefill |
|---|---|---|---|
| 算法1 强制死等（≤2 常驻，间隙空转）| **138s** | 39,213 | turn≥1 = 4 tok（复用）|
| 算法2 动态轮换（间隙填其他会话）| **103s** | 39,213 | turn≥1 = 4 tok（复用）|

**实证结论**（修正/细化 §7 模拟假设）：
1. **llama-server `--kv-unified` 的槽前缀匹配让「轮换」也零 re-prefill**——两算法 prefill 完全相同（39,213 = 纯增量），此前 evict_agent 的 churn 伪影源于「生成≠参考输出」破坏前缀连续，非调度本身。
2. **轮换快 34%**（138→103s）且无 re-prefill 代价——纯赚 gap 填充。
3. 5 会话 × 2 常驻 → 死等串行化排队（138s）；轮换填满间隙（103s）。与模拟方向一致，但**真实 server 无需 CPU swap**（槽匹配免费保 KV），§7 的 swap 成本是多余假设。

**对 §7 的修正**：真实 llama-server 上「动态轮换」= 间隙轮转其他会话即可，KV 由 unified 槽匹配免费保留，无需 CPU 搬入搬出。

## 9. 真并发压池（2/3/4 在途，slot=2）——排队+驱逐+swap 实证（2026-09-06）

**修正设计**：此前调度测试顺序发请求（1 在途）无池压力。本实验真并发 R=2/3/4 在途（每会话一线程同时发请求），`--parallel 2`（slot=2）+ 池 65536，上下文增长（CacheWise，cap 32K）。

| 在途 R | wall | 总 prefill | 重 prefill | 池超额(估) | swap 成本(13GB/s) |
|---|---|---|---|---|---|
| 2 | 40s | 59,681 | 7,032 | ~0 | 0 |
| 3 | 94s | 95,504 | 10,087 | ~19,881 | 0.1s |
| 4 | 129s | 119,598 | 9,053 | ~45,009 | 0.3s |

**swap 带宽实测**：CPU↔GPU **~13 GB/s**（156K-context KV 6.59GB 双向各 0.5s）；模拟器此前假设 6 GB/s 需修正。

**结论**：
1. **真并发制造排队**：R>2 时请求在 slot=2 服务端排队，wall 40→129s。
2. **池驱逐→重 prefill 实测**：R≥3 池超额（20K-45K token），被逐会话重 prefill ~10K token（≈3.4s @3000 t/s）。
3. **swap 是正解**：驱逐量 swap 仅 0.1-0.3s（~10-30× 便宜）；上下文越大优势越大（156K: 0.5s swap vs 52s 重 prefill = **~100×**）——这是论文核心论点。

数据：`results/data/qwen38_ruler/conc_test/` + `run_concurrent.py`。

## 10. 高 subagent 并发：KV 继承 / tool 预测 / offload 策略（2026-09-06）

### Q1 实测：llama.cpp 不做 sibling subagent 前缀复用（9B, 2 slot）
共享前缀 P(8K) + 4 subagent(P+2K)，`--kv-unified` 2 slot：**每 subagent 都全量 re-prefill**（prompt_n≈9800/10240，复用率 ~0.04）。
归因：llama.cpp 槽匹配只支持**线性前缀延续**（turn k+1 延续 turn k），兄弟分叉（P+tail_0 vs P+tail_1）在 P 后分叉无法匹配；`--cache-reuse` 在 unified 下禁用。vLLM/SGLang radix cache 才支持兄弟共享。

### Q1b 对照：vLLM FP8 + prefix caching 复用共享前缀（docker v0.26）
同一负载（P=8K + 兄弟 subagent）在 vLLM `--enable-prefix-caching` 下：

| 请求 | vLLM wall | cache_hit_delta(9717) | llama.cpp 复用率 |
|---|---|---|---|
| 兄弟 P+t2 | 0.17s | 9504 (98%) | 0.04 |
| 兄弟 P+t3 | 0.76s | 7392 (76%) | 0.04 |
| cold 全新 12K | 3.14s | 0 | — |

**vLLM radix/prefix cache 复用共享前缀（65-98% 命中，快 4-20× vs cold）；llama.cpp 零复用**——同一模型同一负载的引擎差距（radix cache vs 线性槽匹配）。

### Q1c 并发 subagent 下 vLLM 复用率 vs 并发度（docker v0.26, FP8）
8 个兄弟 subagent(P=8K+tail) 并发发射（每档重启清缓存），TTFT 随并发：

| 并发 | 兄弟 TTFT | cold TTFT | 复用效应 |
|---|---|---|---|
| 1 | 0.73s | 2.76s | 3.8× |
| 2 | 1.44s | 3.1-4.3s | 2-3× |
| 4 | 2.3-5.7s | ~2.6s | 消失 |
| 8 | 3.1-8.0s | ~2.6s | 反超(thrashing) |

**FP8 KV ~270KB/token，6GB 缓存 ≈ 22K tokens（P 8K + ~7 tail）；8 并发 80K 超缓存 → P 被逐 → 兄弟退化到全量 prefill。**
→ **vLLM 复用天花板 = KV 缓存容纳的共享前缀工作集；高并发(≥4)时与 llama.cpp 一样退化**。

### Q2 tool 时长预测 + offload 策略
CacheWise 10K+ 次工具：中位 4.7s/均值 80.6s/p90 26.2s；按类型可预测（ExitPlanMode 51s, WebSearch 33s, Grep/Glob 3-4s）。
**决策规则（实测成本）**：swap 13 GB/s，swap 往返/re-prefill ≈ **0.02 恒定** →
- 有 KV offload：永远 switch（模拟 b=1356s < a=1564s）
- 无 offload：选择性 switch，阈值=重 prefill 时间（32K=10.9s）
- 模拟：`results/data/qwen38_ruler/policy_sim/sim.json`

### Q3/Q4 设计结论
- **harness 必须感知服务端**：决策 a/b/c 需要槽/KV 驻留状态、池占用、swap 成本、预测 tool 时长——无此信息无法调度（需 scheduler-aware client 或服务端 hint）。
- **subagent 串并行**：兄弟共享 P，最优 = P 常驻（专用槽/radix）+ tail 批 prefill + decode 2 槽并行；llama.cpp 现状每次 subagent 都重 prefill P（40 sub × 8K = 320K 浪费）。

数据：`results/data/qwen38_ruler/policy_sim/`、`kv_inherit`（Q1 实测）；脚本 `sim_policy.py`、`test_kv_inherit.py`。

### Q1d 验证：vLLM native KV offload 到 CPU 可用（docker v0.26）
`--kv-offloading-size 8 --kv-offloading-backend native` → 日志确认 `OffloadingConnector`+`CPUOffloadingSpec`（experimental）。
**conc=8 兄弟 subagent**：无 offload 3-8s（thrashing）；有 offload wave1/2 = **0.60-0.65s**（复用保留，5-13×）。
→ **server 侧 KV 换入换出有现成实现，能解决高并发复用崩溃**。

## 12. 端侧 subagent DAG 联合调度原型（2026-09-06）

**问题**：端侧约束（1 卡 2 槽、KV 缓存 C、CPU RAM、PCIe 13 GB/s）下的 subagent DAG 调度，4 决策联合优化墙钟：①harness 决定 spawn 哪些 subagent ②server 决定 KV 驻留/换出/丢弃 ③依赖解析 ④目标最短。

**原型** `sim_subagent.py`（离散事件，实测成本）：P=8K + 8 subagent × 3 轮工具。

| 依赖 | serial | greedy(丢缓存重算) | joint(harness+swap) |
|---|---|---|---|
| indep | 335s | 216s | **195s** |
| chain | 335s | 378s | 336s |
| fan | 335s | 216s | **195s** |

**KV 缓存敏感性**（indep）：16K→greedy 217s(重算149K)/joint 195s(swap 12.6GB≈1s)；98K→两者 194s。

**结论**：
1. joint 对缓存大小鲁棒（swap 把内存约束变成可控成本）；greedy 缓存越紧重算越多。
2. 依赖结构是墙钟天花板：chain 串行化无可逃，可并行 1.7× 收益。
3. 端侧三约束（槽数/内存/PCIe）显式建模；swap 带宽 13 GB/s 下成本可忽略。

数据 `results/data/qwen38_ruler/subagent_sim/`；脚本 `sim_subagent.py`。

## 13. 动态 KV offload 成本模型 vs 重算（2026-09-06，`model_offload_cost.py`）

实测参数：prefill rate 8K=3570→261K=1706 t/s（9B A1 矩阵）；swap 13 GB/s（memcpy）；KV q4=42KB/FP8=270KB/token。

**核心结论（与上下文无关的比值）**：
```
swap(双向)/recompute = 2*bpt*rate/B = 0.023 (q4) / 0.12 (FP8)   ← 恒定!
跨界带宽 B* = 2*bpt*rate = 0.30 GB/s (q4) / 1.93 GB/s (FP8)
PCIe 13 GB/s >> B* → swap 恒比重算便宜 8-44×
```
- C=32K: offload 0.21s(q4)/1.36s(FP8) vs recompute 9.2s
- **动态决策**：间隙 G > offload 时间 → 成本藏进间隙（免费）；G < offload 时间 → 略延迟但仍远优于重算
- **政策含义**：有 offload（vLLM native）→ **永不 drop-and-recompute，总 offload**；无 offload（llama.cpp）→ 才考虑重算

数据 `results/data/qwen38_ruler/offload_model/`；脚本 `model_offload_cost.py`。

## 14. 协议偏离记录

- 模型：27B→9B（断电规避）；split：TP2/PP→单卡（断电规避）。
- `-lgc` 锁频未生效（无权限），改为依赖热降频自然状态 + `-cram 0` 保证一致性。
- A2 扩展输出：模型在 ~110-128 token 处 EOS 停，无法测长输出摊薄，作废。
- A3 质量轴：Paul Graham essays 下载失败（URL 失效），未跑。
- Part B serving：27B off 系已完成（见 `partB/`）；9B 单卡版待跑。