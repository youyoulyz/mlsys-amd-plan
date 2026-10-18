# vLLM 单卡 Batch Decode（关闭 MTP）

这个实验测量单卡 vLLM 在 dense 3B 模型上的批量解码吞吐：横轴为并行度（同时提交的请求数 / batch size），纵轴为最终总输出 token 速度（output tokens/s）。默认模型为 `/mnt/public/models/Qwen2.5-3B-Instruct`。

## 运行

```bash
# `vllm` 环境当前仅含 PyTorch，实际的 ROCm vLLM 安装在 rocm7.14。
conda activate rocm7.14
cd /mnt/luyuzhou/hpc/moecho/experiments/vllm_batch_decode_nomtp_20260913
bash run.sh
```

可覆盖参数，例如减少生成长度或指定模型：

```bash
bash run.sh --model /mnt/public/models/Qwen2.5-3B-Instruct --max-new-tokens 128 --batches 1,2,4,8,16
```

本次正式扫描使用你提供的 ROCm 容器，并扩展到 batch 256：

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \\
  -v /mnt/luyuzhou/hpc/moecho/experiments/vllm_batch_decode_nomtp_20260913:/exp \\
  -v /mnt/public/models/Qwen2.5-3B-Instruct:/model:ro -w /exp \\
  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0 \\
  python bench_decode.py --model /model --batches 1,2,4,8,16,32,64,128,256 --max-new-tokens 128 --repeats 2
```

结果写入 `results/`：`decode_results.csv` 为原始数据，`decode_throughput.png` 为曲线，`run_config.json` 保存实验配置和运行前的 `amd-smi` 状态。脚本使用 vLLM 离线批量 API，未设置 `speculative_model` / `speculative_config`，因此 MTP 关闭。主机由 `amd-smi list` 确认只暴露一张 GPU；该 ROCm 构建设置 `*_VISIBLE_DEVICES=0` 反而会导致零设备，因此不设置该变量。

## 口径

每个 batch size 先执行一次 warmup，再执行指定重复次数；计时从 `LLM.generate` 调用前到返回后，吞吐为该批次所有请求实际生成 token 数除以墙钟时间。固定 prompt 和 greedy decoding（temperature=0）降低比较噪声。首次加载模型时间不计入 decode 时间。当前单卡可运行上限实测到 batch 256；batch 128 达到最高约 5703 output tok/s，batch 256 回落至约 5387 output tok/s。
