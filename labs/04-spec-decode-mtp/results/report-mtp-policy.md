# P2 实验：并发感知 MTP 策略（Policy-A）主实验 (2026-08-13)

> 目标：验证「低并发关 MTP、高并发开 MTP(dm=2)」的并发感知策略在 commodity dual-GPU vLLM 上的 goodput 收益；对照 Always-MTP / Never-MTP / Oracle（逐请求事后选优上界）。
> 环境：vLLM 0.26.0（华为云 docker 镜像）、TP=2、Qwen3.6-35B-A3B-NVFP4-Fast（compressed-tensors，lm_head 已解量化回 bf16）、无 P2P 配方（`NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce`）、`--max-model-len 16384`。

## 0. 结论（TL;DR）

1. **serving 持续负载下 MTP 处处不赚（本栈：vLLM 0.26 TP=2 NVFP4）**：固定并发闭环 c=1..16 净亏 −11~−28%（c=1 −28%、c=8 −12%、c=16 −11%）；泊松到达 λ≤16 时 goodput=offered load（ON/OFF 打平），**λ=24 饱和时 Always-MTP goodput 腰斩（618 vs 1222，−49%）**。
2. **旧「c=8 MTP 反超 +14%」是 lockstep 单轮协议伪影**：n=256、c=8 一次性并发（8 请求同时到达、纯 decode 锁步）下 MTP +10%（91.2→100.5 t/s/req）；同负载持续闭环下 **−7%**（95.2→88.6）；短请求混合闭环 −13%。**评估协议决定 MTP 符号**。
3. **Policy-A(C*=8) 在本负载下 ≈ Always-MTP**：λ≥12 时 inflight 常超阈值 → ~78~99% 请求路由 ON → 高负载塌陷。**该栈该负载的最优阈值是「永不开启」**：Never-MTP 距 Oracle（逐请求事后选优上界）不足 1%。
4. 塌陷机制（λ=24）：Always-MTP 的 agg t/s 与 OFF 相同（1204 vs 1203），**全部损失在 TTFT**（ON p99 1.26s vs OFF 0.11s）——spec-decode 每步固定 CPU 开销在饱和时挤压 prefill 排队；TPOT 不受影响（0.024 vs 0.025 s/token）。

> 对论文的影响：P2 claim 需调整——「并发感知策略防止 MTP 步膨胀损失」成立（策略=关 MTP 时保护 goodput，避免 −49% 塌陷）；「高并发开 MTP 赚钱」在当前 serving 负载与实现下不成立。建议把「lockstep vs 持续到达的协议依赖」写进方法学贡献（这本身是可发表的判别结果）。

## 1. 实验协议（固定，写进论文）

### 1.1 负载

| 负载 | 参数 | 说明 |
|---|---|---|
| 固定并发（闭环） | c ∈ {1,2,4,8,16}，45 s/rep × 2 rep | c 个 worker 持续循环发请求，保持并发≈c（serving-like 持续到达，非一次性并发） |
| 泊松到达 | λ ∈ {0.5,1,2,4,8,12,16,24} req/s，60 s/rep × 2 rep | 到达时刻运行前预生成，跨 server 配置逐 slot 对齐（同 seed） |

- 请求池：32 条短 prompt（medical/code/finance/general 各 8 条）+ 2 条长 prompt；
- 输出长度分布（固定）：n=32 占 60%、n=64 占 30%、n=128 占 10%（短请求为主 + 少量长请求）；
- `temperature=0.6`、`stream=true`（流式拿 TTFT）；warmup 1 请求不计入；
- 每个条件：负载核验（loadavg）、双 rep、rep 间 2 s 间隔。

### 1.2 指标与 SLA（固定）

- SLA（每请求）：**TTFT ≤ 500 ms 且平均 TPOT ≤ 50 ms**；满足即 good。
- goodput：good 请求的 token 数 / 忙窗口（最后一个完成 − 第一个到达），另报 goodput(req/s)。
- 辅指标：agg t/s、TTFT mean/P99、TPOT mean/P99、MTP-on 比例（Policy 下路由到 ON 的请求占比）。
- 忙窗口口径：`max(t_arrival + t_done) − min(t_arrival)`。

### 1.3 策略定义

- **Policy-A（阈值）**：请求到达时观测客户端 inflight（已发送未完成，外置控制器可见量）；`inflight < C*` → 路由 OFF（MTP off）；`inflight ≥ C*` → 路由 ON（dm=2）。扫 C* ∈ {2,4,6,8}，主报 C*=8。
- **离线混合模拟**：OFF/ON 两个纯配置 runs 逐 slot 对齐（同 seed 同调度）；事件驱动重放调度，按 Policy 规则选择每请求的完成时刻/延迟（取对应 server 测得的逐 slot 延迟），重算全部指标。Always-OFF/Always-ON 模拟用于校验（应与实测一致）。Oracle = 逐 slot 取两 server 中更优（TTFT/TPOT 取 min，good 判定取任一满足 SLA）。
- 实现路径 v1 = 外置控制器（客户端旁路按 inflight 路由），论文写 admission/controller；引擎内动态 `num_speculative_tokens∈{0,2}`（epoch 切换）留作 v2。

### 1.4 已知机制（承接 EXP_VLLM_MTP_ISOLATION_20260813）

- vLLM MTP 每步固定引擎开销 ~6–8 ms + Marlin draft/verify 微 kernel 风暴（93→1,090 kernel/步），单流净亏 −10~22%；
- 一次性并发（lockstep）下 c=8 反超 +14% 的旧结论为**单轮协议伪影**（见 §3 判别实验）；
- 流式 SSE 在 MTP 下每事件含多个接受 token，**必须用真实 tokenizer 数 token**（否则 tokens/TPOT 低估 ~2×）。

## 2. 主结果

### 2.1 固定并发（短请求混合）

闭环持续到达（c worker 循环发请求，并发≈c），n 分布 32/64/128 = 60/30/10%，45 s × 2 reps：

| c | Never (OFF) | Always (ON) | Policy-A C*=8 | Oracle | MTP 净效果 |
|---|---|---|---|---|---|
| 1 | 150.1 | 108.1 | 150.1 | 150.3 | **−28%** |
| 2 | 241.2 | 195.2 | 241.2 | 241.3 | −19% |
| 4 | 392.1 | 334.7 | 392.1 | 393.5 | −15% |
| 8 | 605.1 | 534.5 | 534.5 | 605.6 | −12% |
| 16 | 909.2 | 806.9 | 806.9 | 910.3 | −11% |

（goodput t/s；SLA = TTFT≤500ms & TPOT≤50ms，本表所有请求均满足 SLA，goodput=agg；c=8 双 rep 波动 <1%，空闲窗口交错复测一致：c=8 OFF 74.7~76.4 / ON 65.9~66.3 t/s/req。）

- 所有 c 下所有请求都满足 SLA（TTFT p99 c=16 仍 <0.2s），OFF/ON 差异是 TPOT/吞吐差异：c=8 ON tpot_mean 13.3ms vs OFF 12.4ms、TTFT mean ON 105ms vs OFF 52ms。
- **MTP 亏损随 c 单调收窄（28%→11%）但从未翻转**——旧翻转点 c=8 在持续到达下不出现。
- Policy-A(C*=8)：c≤4 正确关 MTP（=Never）；c≥8 开 MTP（=Always），**反而受损**。

### 2.2 泊松到达

| λ | mean inflight(OFF) | Never | Always | Policy C*=8 | Oracle | Policy MTP-on 比例 |
|---|---|---|---|---|---|---|
| 0.5 | 1.1 | 22.8 | 22.8 | 22.8 | 22.8 | 0.0 |
| 1 | 1.3 | 44.3 | 44.1 | 44.3 | 44.3 | 0.0 |
| 2 | 1.7 | 98.9 | 99.3 | 98.9 | 99.5 | 0.0 |
| 4 | 3.0 | 206.8 | 206.1 | 206.8 | 207.3 | 0.012 |
| 8 | 6.1 | 406.5 | 406.8 | 406.7 | 408.0 | 0.19 |
| 12 | 11.4 | 633.1 | 631.0 | 631.1 | 633.6 | 0.78 |
| 16 | 17.2 | 822.6 | 822.6 | 822.6 | 825.0 | 0.95 |
| 24 | 34.0 | 1222.1 | **617.8** | **617.8** | 1224.8 | 0.995 |

（goodput t/s；60 s × 2 reps，模拟与实测逐 slot 对齐后一致。）

- λ≤16：goodput≈offered load（大部分请求满足 SLA），ON 与 OFF 打平——泊松突发形成锁步批，MTP 的 verify 批摊薄，与闭环固定 c 的持续亏不同。
- **λ=24（饱和）：Always-MTP 崩溃**：agg t/s 与 OFF 相同（1204 vs 1203），TTFT mean 0.42s / p99 1.26s vs OFF 0.07/0.11s → 63% 请求（goodput 617 vs 1222）因 TTFT 超 SLA 被计为不达标。
- Policy-A(C*=8) 在 λ≥12 时把 ~78~99.5% 请求路由到 ON → 高负载行为 = Always-MTP。C*∈{2,4,6,8} 敏感度平坦（inflight 分布双峰，阈值几乎不影响路由比例）。
- Oracle（逐请求选 OFF/ON 更优）≈ Never：差距 0.1~0.8%。**上界证明「关 MTP 无损失」**。

## 3. 判别实验：一次性并发 vs 持续到达（旧翻转点复核）

c=8、n=256（同 prompt、temp=0.6）、空闲窗口（loadavg<1.7）交错复测：

| 协议 | OFF（t/s/req） | ON（t/s/req） | MTP 净效果 |
|---|---|---|---|
| 一次性并发（lockstep，8 请求同刻到达） | 91.2 | 100.5 | **+10.2%** |
| 闭环持续到达 | 95.2 | 88.6 | **−6.9%** |
| 闭环持续到达 + 短请求混合 | 75.5 | 66.1 | **−12.5%** |

- OFF 对协议不敏感（91~95）；**ON 对协议高度敏感**（one-shot 100.5 vs 闭环 66~89）：持续到达时序列相位漂移 + 持续 prefill 打断 spec 批，draft/verify 批无法维持 lockstep 摊薄。
- 与 08-13 隔离实验的表（c=8 +14%）对比：其「c=8 赚」来自一次性协议；本文闭环协议下同点反号。

## 4. 与论文 claim 的映射

- ✅ 可写：**concurrency-aware 策略（外置控制器 v1）防止 MTP 的 step-inflation 损失**——本负载下策略学到的就是「关」，Always-MTP 在高负载亏 11~49%，Never/POLICY 距 Oracle <1%。
- ⚠️ 需调整：原「高并发（c≥8）MTP 反超」在 serving 持续负载下不成立；仅 lockstep 一次性基准成立（+10%）。论文需把协议依赖讲清楚，否则审稿人按 Dynamo/LLMEdger 的相位切 TP 类比时会被「你们自己的曲线没翻转」质疑。
- 🔬 新方法学贡献候选：**推测解码的批处理基准（一次性并发）高估 MTP 收益，持续到达 + SLA-goodput 才会暴露 TTFT 塌陷**。
- 下一步（若坚持 MTP 收益 claim）：找 MTP 真正赚钱的 serving 条件（更长请求、纯 decode 期、batch 阶段感知调度），或修 vLLM spec 引擎每步固定开销；引擎内动态 `num_speculative_tokens∈{0,2}`（epoch 切换）留作 v2。

## 5. 文件与复现

- 脚本：`experiments/mtp_policy_20260813/{launch_vllm.sh,loadgen.py,analyze_policy.py,plot_policy.py,run_all.sh}`
- 数据：`experiments/mtp_policy_20260813/data/`（load_*.json + metrics_*.json）
- 结果：`experiments/mtp_policy_20260813/results/`（policy_results.json + p2_fixed_c.png + p2_poisson.png + p2_protocol_dependence.png）
- 判别实验原始数据：`experiments/mtp_policy_20260813/data_diag/`（oneshot 输出在 memo §3）

## 6. 方法与口径注意（写进论文的 appendix）

- **SSE 流式下 MTP 每个事件含多个接受 token**：必须以真实 tokenizer 数 token（本 repo 用 tokenizer.json），否则 tokens/TPOT 低估 ~2×（早期 20s 冒烟曾得到 tokens≈n/2 的假象）。
- 多进程负载生成器的 `t_arrival_s` 各自基于进程内单调钟：**跨 rep 合并前必须先按 seed 归一**，否则 busy window 错算（曾导致 goodput 高估 ~2×）。
- 共享机噪声：MTP（CPU-bound）对 loadavg 敏感（±30% 极端），结论都经空闲窗口（loadavg<1.7）交错复测；主表 per-seed 波动 <1.5%。
