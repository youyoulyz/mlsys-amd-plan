#!/usr/bin/env python3
"""
并发压测:llama-server 多请求吞吐(单卡 vs 双卡 PP × MTP)
用法: python3 mtp_concurrency_test.py --url http://127.0.0.1:8080 --concurrency 4 --n-tokens 128
"""
import argparse, json, time, statistics
from concurrent.futures import ThreadPoolExecutor
import urllib.request

PROMPT = "The history of the Roman Empire is a long and complex story."

def one_request(url, n_tokens, model):
    body = json.dumps({"model": model, "prompt": PROMPT, "max_tokens": n_tokens, "temperature": 0.6,
                       "stream": False, "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    n = data.get("usage", {}).get("completion_tokens", 0)
    return {"elapsed": dt, "tokens": n, "tps": n / dt if dt > 0 else 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-NVFP4-Fast")
    args = ap.parse_args()

    all_res = []
    for rep in range(args.repeat):
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(one_request, args.url, args.n_tokens, args.model)
                    for _ in range(args.concurrency)]
            res = [f.result() for f in futs]
        all_res.extend(res)

    total_tokens = sum(r["tokens"] for r in all_res)
    total_time = max(r["elapsed"] for r in all_res)  # 最后一请求完成时间
    agg_tps = total_tokens / total_time
    elapseds = [r["elapsed"] for r in all_res]
    per_req_tps = [r["tps"] for r in all_res]
    print(json.dumps({
        "concurrency": args.concurrency,
        "n_requests": len(all_res),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "aggregate_tps": round(agg_tps, 1),
        "per_req_tps_mean": round(statistics.mean(per_req_tps), 1),
        "per_req_tps_std": round(statistics.stdev(per_req_tps), 1) if len(per_req_tps) > 1 else 0,
        "per_req_latency_mean_s": round(statistics.mean(elapseds), 2),
        "per_req_latency_p95_s": round(sorted(elapseds)[int(len(elapseds)*0.95)-1], 2),
        "errors": sum(1 for r in all_res if r["tokens"] == 0),
    }, indent=2))

if __name__ == "__main__":
    main()
