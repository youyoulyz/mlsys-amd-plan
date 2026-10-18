#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); values = defaultdict(list)
    with open(a.input, newline="") as f:
        for row in csv.DictReader(f): values[int(row["batch_size"])].append(float(row["output_tokens_per_s"]))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = sorted(values); ys = [sum(values[x]) / len(values[x]) for x in xs]
    plt.figure(figsize=(7, 4.5)); plt.plot(xs, ys, "o-", linewidth=2)
    plt.xlabel("Parallelism / batch size"); plt.ylabel("Total output tokens/s")
    plt.title("Single-GPU vLLM decode throughput (MTP disabled)"); plt.grid(True, alpha=.3); plt.xticks(xs)
    plt.tight_layout(); plt.savefig(a.output, dpi=160)

if __name__ == "__main__": main()
