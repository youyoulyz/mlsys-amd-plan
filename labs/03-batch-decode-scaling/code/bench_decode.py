#!/usr/bin/env python3
"""Single-GPU vLLM batch decode throughput sweep with MTP disabled."""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/public/models/Qwen2.5-3B-Instruct")
    p.add_argument("--batches", default="1,2,4,8,16,32")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--input-text", default="Explain how batching improves language model decoding throughput in two sentences.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    if any(x <= 0 for x in batches) or args.repeats <= 0:
        raise ValueError("batch sizes and repeats must be positive")
    from vllm import LLM, SamplingParams

    # No speculative_model/speculative_config is passed: MTP is disabled.
    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, seed=args.seed)
    sampling = SamplingParams(temperature=0.0, top_p=1.0,
                              max_tokens=args.max_new_tokens,
                              ignore_eos=True)
    prompt = args.input_text
    rows = []
    for batch_size in batches:
        prompts = [prompt] * batch_size
        llm.generate(prompts, sampling, use_tqdm=False)  # warmup
        samples = []
        for rep in range(args.repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            samples.append((elapsed, output_tokens))
            rows.append({"batch_size": batch_size, "repeat": rep,
                         "elapsed_s": elapsed, "output_tokens": output_tokens,
                         "output_tokens_per_s": output_tokens / elapsed})
        mean_elapsed = sum(x[0] for x in samples) / len(samples)
        mean_tokens = sum(x[1] for x in samples) / len(samples)
        print(f"batch={batch_size:>3}  tokens={mean_tokens:.1f}  throughput={mean_tokens / mean_elapsed:.2f} output tok/s")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "decode_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    gpu_info = subprocess.run(
        ["amd-smi", "metric", "-g", "0", "--usage", "--mem-usage", "--json"],
        capture_output=True, text=True, check=False).stdout
    config = vars(args) | {"batch_sizes": batches, "mtp_enabled": False,
                           "mtp_disabled_by": "No speculative_model or speculative_config supplied to LLM",
                           "tensor_parallel_size": 1,
                           "device_selection": "One GPU visible on host (amd-smi list)",
                           "amd_smi_before_run": json.loads(gpu_info) if gpu_info else None}
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
