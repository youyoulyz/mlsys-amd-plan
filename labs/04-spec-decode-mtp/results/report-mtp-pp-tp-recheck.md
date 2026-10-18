# 复核 memo:PP vs TP 跨卡通信对 MTP 的影响(官方路径,2026-08-09)

> 触发: 矩阵核心结论(PP 零代价、MTP 栈内 TP 反转)基于 mtp-probe 探针客户端,已被证伪(见 `EXP_MTP_NOP2P_DUALGPU.md` §6.7)。本次用**官方最新 master(08659901c)+ 官方 llama-cli + NCCL 构建**复核 PP/TP 的跨卡通信对 MTP 的影响。
> 判据: llama-cli `Generation: X t/s`(官方 A/B 口径,llama-cli 不打印接受率)。

## 结论先行

1. **PP 的 MTP 收益(+26~34%)是 TP(+15.3%)的两倍**——TP 的跨卡 all-reduce 通信税吃掉一半 MTP 收益。**机制:TP 下 MTP draft 步也要跨卡(draft ctx 权重 tensor 分片,每步 all-reduce);PP 下 draft 只跑最后一张卡(blk.40 在 GPU1),零跨卡**——矩阵时代假设的"PP 结构优势"在官方路径上兑现。
2. **探针版"MTP 栈内 TP 反转(TP 快 6.4%)"结论作废**:官方路径 MTP 开时 PP(170-182)大幅优于 TP(149.0),PP 高 14-22%。
3. **无 MTP 时 PP 略优于 TP**(135.2 vs 129.2,+4.6%)——与矩阵/E 实验方向一致(无 P2P 下 TP 通信税,但幅度比探针版温和)。
4. **跨卡通信 × MTP 的净效应**:TP 的 MTP 增量收益仅 PP 的一半,PP+MTP(170-182 t/s)为全部配置最高——**无 P2P 双卡 + MTP 的最佳部署 = PP(layer split)+ MTP**。

## 数据(官方最新 master,NCCL 构建,双卡,Q3_K_XL,同 prompt "Common cold symptoms and treatment:",n=64,temp=0.6)

| 配置 | 无 MTP | MTP | MTP 收益 |
|------|--------|-----|----------|
| 单卡 | **134.1** | 装不下(20GB OOM) | — |
| PP(layer split,默认) | 135.2 / 135.5 | **170.2 / 175.0 / 182.1** | **+26~34%** |
| TP(tensor 0.5/0.5,`NCCL_P2P_DISABLE=1`) | 129.2 | **149.0** | **+15.3%** |
| 单卡 llama-bench tg1(参考,带同步开销) | ~75(13.3ms/tok) | — | — |

- 每次配置 2-3 次运行取稳定值;PP MTP 三次(170.2/175.0/182.1)均值 ~176
- TP 配置 NCCL 无 P2P(host 中转 all-reduce,与 E 实验同配方)

## 与探针版矩阵的对比(作废 vs 保留)

| 结论 | 探针版(矩阵) | 官方路径(本次) | 判定 |
|------|-------------|---------------|------|
| PP 双卡 ≈ 单卡(零代价) | steady −1.2%,tps +1.3% | PP 135 vs 单卡 tg1 ~75 t/s(**PP 大幅更快**,单 token 时双卡并行摊层) | 方向修正:PP 不是"零代价"而是"正收益"(单流) |
| MTP 栈内 TP > PP | TP steady 快 6.4% | **PP MTP(176) >> TP MTP(149),PP 高 18%** | **反转,探针伪影** |
| TP 无 MTP 负收益 | E 实验 −14.4%(batch16) | TP 129.2 vs PP 135.2(−4.6%) | 方向一致,幅度温和 |
| Q3 ≈ Q4 | tps 83.1 vs 83.0 | Q4 MTP 收益 +40.7% > Q3 +30% | **反转,量化精度影响 draft** |

## 机制解读(跨卡通信 × MTP)

- **PP**: target 步跨界 1 次 4KB hidden state;draft 步(blk.40)全在 GPU1——draft 零跨卡。MTP 收益 = 纯计算收益(draft 便宜 + verify batch 摊薄)。
- **TP**: 每层 2 次 all-reduce(attn + MoE)× 40 层,target 步 ~80 次;draft 步(draft ctx 同样 tensor 分片)也 ~2 次 all-reduce——**draft 步的跨卡税与 target 同比例**,吃掉的正是 MTP 的增量收益(+15.3% vs +26-34%)。
- 量级:无 P2P 下 4KB 级 all-reduce ~10-30µs × 80 次/步 ≈ 1-2ms,对 7-8ms 步长是 15-25% 税——与 TP MTP 收益减半吻合。
- **推论**: 若未来有 P2P(13.55GB/s),TP 的 all-reduce 税大降,TP+MTP 可能反超——P2P 是 TP 的解锁器(与 E 实验"TP 是带宽绑定"一致)。

## 并发补充(2026-08-09 深夜):req 打满后单卡=双卡,瓶颈在 CPU 调度

llama-server(--parallel)压测(每请求 128 tokens,聚合吞吐):

| 配置 | 4 并发 | 8 并发 |
|------|--------|--------|
| 单卡 | 223.0 t/s | 260.4 t/s |
| 双卡 PP | 226.6 t/s | 257.7 t/s |
| 双卡 PP + MTP | 226.4 t/s(接受率 **64-72%**,均值 2.4 tokens/步) | — |

- **并发打满后单卡=双卡=双卡+MTP**(4 并发 223/227/226;8 并发 260/258)——双卡算力翻倍的优势完全没兑现
- **瓶颈定位:CPU/调度侧,不是 GPU**——4 并发压测时双卡 SM 利用率仅 mean 30%/peak 46%(n=3 采样),GPU 大量闲置但吞吐已与单卡持平;llama-server 的 slot 调度/采样链在 CPU
- **MTP 在并发下增量消失**(226.4 vs 226.6),但接受率正常(64-72%,与社区一致)——MTP 的投机收益被 CPU 调度瓶颈掩盖
- 单流结论不变:MTP 是唯一显著增益(+30-40%),并发下所有配置收敛
- 待验证: 若用更高并发(16+)或低 CPU 开销后端(vLLM/SGLang),双卡优势可能显现——llama-server CPU 调度是上限,不是 GPU;**更高并发不会减少每步 launch 次数,突破须换后端或改 ggml**

## 并发单卡 profile 结论(2026-08-09 深夜,nsys,8 并发压测段 8.6s)

- **GPU busy 64.6%**(5.56s/8.6s)——比单流(12-15%)大幅提升,但未满;两卡完全并行(2.77+2.78s,无串行化)
- **剩余 35% = CPU 图调度(launch-bound),不是 GPU/跨卡/采样**:
  - 压测段 **712,990 次 cudaLaunchKernel**(≈ kernel 数),CPU launch 总耗时 ~3.5s(×5µs)≈ GPU 空洞 3.0s——**ggml 逐节点串行 launch 是共享瓶颈,与 GPU 数量无关**(解释单卡=双卡=双卡+MTP 的全部收敛)
  - 同步率 0.04/kernel(并发批处理稀释,非同步问题)
  - 跨卡 cudaMemcpyPeerAsync 仅 348 次(40/s,PP 边界,非瓶颈)
  - 间隙分布:每卡 ~390 次 >50µs 空洞(平均 22ms 一次),最大 ~1ms
- **推论(修正版)**: 双卡算力要兑现,须绕过 ggml CPU 调度。**注意:"大 batch 单步"已由 `--parallel N` 实现(server 把多 slot 合并进同一次 llama_decode),它摊薄 launch 开销但不减少 launch 次数/步(600-800 次,与 batch 无关)**;真正降 launch/步只有两条路——CUDA graph(已被实验否定,见功耗诊断)或 ggml kernel 融合(大改核心)。现实路径: 换低 CPU 开销后端(vLLM/SGLang 有自己的调度器)

## vLLM 对照(2026-08-09 深夜):NVFP4-Fast + MTP,并发完胜 llama.cpp

**环境**: vLLM 0.26.0(华为云镜像)+ unsloth/Qwen3.6-35B-A3B-NVFP4-Fast(compressed-tensors,23GB)+ TP=2 + `NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce` + MTP(`{"method":"mtp","num_speculative_tokens":2}`)。
**踩坑**: ① vLLM 0.22 无法加载量化 lm_head(`weight_scale` 不识别)→ 需 ≥0.25;② 0.25/0.26 有 Humming 内核 bug(`ParallelLMHead` 无 `output_partition_sizes`)→ **hack:lm_head 解量化回 bf16 写回模型**(脚本 `/home/luyzh/models/hack_lmhead.py`,config targets 去 lm_head)——成功;③ NVFP4 在 Ampere 走 Marlin 后端(README 警告 2x 慢)。

**吞吐对比**(每请求 128 tokens):

| 配置 | 1 并发 | 4 并发 | 8 并发 |
|------|--------|--------|--------|
| **vLLM TP=2 NVFP4+MTP** | **108** | **405** | **671** |
| llama.cpp 双卡 PP + MTP | 175-182 | — | — |
| llama.cpp 双卡 PP 无MTP | 135 | 227 | 258 |

**结论**:
1. **并发下 vLLM 完胜**(4 并发 +78%,8 并发 +160%)——**llama.cpp 的 CPU 图调度瓶颈被 vLLM 调度器绕过**,vLLM 并发缩放近线性(108→671,6.2×),llama.cpp 卡在 ~260(CPU 上限,与并发单卡 profile 结论一致)
2. **单流 llama.cpp MTP 更快**(175-182 vs 108)——vLLM TP=2 的跨卡通信税 + Marlin 后端(2x 慢)吃掉单流优势
3. **MTP 接受率一致**: vLLM 61-67%(server SpecDecoding metrics),与 llama.cpp 官方路径(~70%)同量级
4. **架构启示**: 无 P2P 双卡 + 并发场景选 vLLM(TP=2);单流选 llama.cpp PP+MTP;两条路各有归属
5. 待优化: vLLM 换 non-Marlin MoE 后端(cutlass)可进一步提速(README: Marlin 2x 慢)

## 40K 长上下文:TP vs PP(2026-08-10,跨框架)

| 维度 | vLLM TP=2(NVFP4-Fast+MTP) | llama.cpp PP(Q3_K_XL+MTP) |
|------|---------------------------|---------------------------|
| 输入 | 42,497 tokens | ~39,000 tokens |
| prefill | **4301 t/s**(~9.9s) | 3441 t/s(~11.3s) |
| 生成 | ~51 t/s | **168.8 t/s** |
| 生成 vs 8K ctx | 108 → 51(**塌一半**) | 175-182 → 169(**几乎不掉**) |

**结论(含 8-10 判别实验修正)**:
1. **vLLM TP 的 40K 塌陷(108→51)与 TP 通信无关**——判别实验:llama.cpp 同款 TP=2(同样跨卡 all-reduce)40K 下 **156.7 t/s 不塌**(8K 149);塌陷是 vLLM 特有。候选机制:FLASHINFER attention 长上下文开销(llama.cpp 用自家 flash-attn 不塌)、MTP 的 `max_num_scheduled_tokens=2048` 调度警告(日志已提示 suboptimal)
2. **PP 生成侧优势仍成立但幅度修正**:llama.cpp PP 40K 168.8 vs TP 40K 156.7——**同框架下 PP 与 TP 在 40K 几乎持平**(+7.7%),不是 3.3×;3.3× 是 vLLM TP vs llama.cpp PP 的跨框架差,归因于 vLLM 特有塌陷,非 PP 结构优势
3. **prefill 侧 vLLM 胜(+25%)**:vLLM 连续批处理调度更强
4. **架构结论补全**:短上下文(8K)并发选 vLLM TP、单流选 llama.cpp PP+MTP;长上下文(40K)生成侧 llama.cpp(PP/TP 皆可)显著优于 vLLM TP;vLLM 40K 慢 = flashinfer attention + MTP 调度待深挖
5. **vLLM PP 不可用**:vLLM 0.26 的 Qwen3_5MoeMTP 未实现 `SupportsPP` 接口(`NotImplementedError`),PP 只能走 llama.cpp layer-split
6. 技术细节:llama.cpp 40K 的 KV 仅 3.4GB(kv_heads=2,80KB/token),双卡轻松;vLLM TP=2 40K 显存 ~15GB/卡(权重 11.4+KV 1.7+graph 1.1+激活 2.2)

## 限制与后续

- 单 prompt 单轮(加载 2-4 分钟/次限制样本量),方向性结论可靠,精确幅度需多 prompt 复核
- llama-cli 无接受率输出,只能 t/s 判据;接受率复核需修 mtp-probe 循环 bug 或用 llama-server 日志
- 单卡 MTP 无法测(20GB 装不下 Q3+MTP 缓冲),PP vs 单卡比较用无 MTP tg1 参考
- 后续: ① 多 prompt × 多 rep 的 PP/TP × MTP 全矩阵(官方路径)② llama-server 拿接受率 ③ 长上下文(32K)复核 ④ mtp-probe 循环 bug 定位

## 数据与复现

- 命令: `CUDA_VISIBLE_DEVICES=0,1 /home/luyzh/llama.cpp-clean/build-nccl/bin/llama-cli -m /home/luyzh/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf -ngl 99 [--split-mode tensor --tensor-split 0.5/0.5] [--spec-type draft-mtp --spec-draft-n-max 2] -p "..." -n 64 --temp 0.6 -c 256 -no-cnv`(TP 需 `NCCL_P2P_DISABLE=1`)
- 构建: `/home/luyzh/llama.cpp-clean/build-nccl`(最新 master 08659901c,GGML_CUDA + GGML_NCCL,sm_86)
- 模型本地副本: `/home/luyzh/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`(NAS d1 坏盘池,先拷本地)
