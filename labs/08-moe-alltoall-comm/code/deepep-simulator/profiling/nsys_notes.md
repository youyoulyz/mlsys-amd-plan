# 用 Nsight Systems / CUPTI 拆解 dispatch 与 combine

目标：把端到端 dispatch/combine 时间拆成 **P2P/集合通信、拷贝、permutation、GEMM**
四块，验证代价模型中 bytes / latency / imbalance 三项的相对贡献。

## 抓取

```bash
source env.sh
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  -o results/dispatch_combine \
  "$DEEPEPSIM_PYTHON" -m torch.distributed.run \
      --nproc_per_node=2 --standalone \
      -m deepep_sim.cli bench --config configs/default.yaml
```

## 建议的 NVTX 分段

在 `bench.run_rank` 里给 `do_dispatch` / `do_combine` 包 `torch.cuda.nvtx.range`，
再把 `all_to_all_single`（集合通信）、`index_select`（permutation/拷贝）、
`grouped_gemm_ref`（GEMM）分别打点，即可在 timeline 上目视拆解。

## 要看的指标

- **all_to_all 段**：是否随 token 数线性增长？`rank_skew` 下是否出现 rank 间等待台阶。
- **permutation 段**：稳定排序 + gather 的开销占比，是否随 topk 增长。
- **GEMM 段**：`grouped_gemm_bmm` 的 padding 浪费 vs 逐 expert 循环。
- **overlap**：多 stream 下 all_to_all 与 GEMM 是否真正交叠，还是被同步点串行化。

## 与模型对照

把 timeline 上的通信段时间除以元素字节数，得到有效带宽，回填到
`cost_model.LinkParams(bandwidth_gbps, latency_us)`，再让 `cli sweep` 重新预测，
比较 **实测 vs 模型** 的偏差。V100 上拟合出的参数只用于相对趋势，不要外推。
