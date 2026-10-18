from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "probe"
DEFAULT_MODEL = "/home/luyzh/models/Qwen3.6-35B-A3B-NVFP4"
MODEL_NAME = "Qwen3.6-35B-A3B-NVFP4"
POLICIES = ["baseline", "noop", "top85", "top90", "floor", "combined", "layer"]
PROMPTS = [
    (
        "math",
        "一个三位数满足：各位数字之和为 17，百位比十位大 3，个位是偶数。请逐步求所有可能值，最后把答案放进 boxed。",
    ),
    (
        "python",
        "请逐步分析 Python 中 list 与 dict 的缓存局部性差异，并给出一个高并发服务中会选择哪种结构的具体例子。",
    ),
    (
        "moe",
        "设计一个在小显存 GPU 上服务 MoE 模型的 expert cache 淘汰策略。先分析问题，再给出算法、复杂度和失败模式。",
    ),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_ready(origin: str, proc: subprocess.Popen, log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-40:])
            raise RuntimeError(f"server exited during startup\n{tail}")
        try:
            health = get_json(f"{origin}/health", timeout=3)
        except (OSError, ValueError):
            time.sleep(1)
            continue
        if health.get("maintenance") == "serving":
            return
        time.sleep(1)
    raise TimeoutError(f"server not ready after {timeout}s")


def stream_chat(
    origin: str,
    prompt: str,
    max_tokens: int,
    *,
    enable_thinking: bool = True,
    ignore_eos: bool = True,
) -> dict:
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = None
    stamps: list[float] = []
    pieces: list[str] = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            item = json.loads(payload)
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    stamps.append(now)
                    pieces.append(text)
    end = time.perf_counter()
    text = "".join(pieces)
    steps = max(0, len(stamps) - 1)
    decode_s = stamps[-1] - stamps[0] if steps else 0.0
    return {
        "text": text,
        "text_sha1": hashlib.sha1(text.encode()).hexdigest(),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "events": len(stamps),
        "ttft_ms": (first - t0) * 1000 if first else None,
        "decode_tok_s": steps / decode_s if decode_s > 0 else None,
        "ms_per_token": decode_s / steps * 1000 if steps > 0 else None,
        "wall_s": end - t0,
    }


def serve_cmd(model: str, gpu: str, port: int, cache: int, decode: int, hybrid_fetch: int) -> list[str]:
    return [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", model,
        "--gpu", gpu,
        "--host", "127.0.0.1", "--port", str(port),
        "--text-model-only",
        "--moe-strategy", "hybrid",
        "--moe-cache-size", str(cache),
        "--moe-hybrid-max-fetch", str(hybrid_fetch),
        "--max-running-requests", "1",
        "--max-seq-len-override", str(8192 + decode),
        "--memory-ratio", "0.9",
        "--cuda-graph-max-bs", "1",
    ]


def stop_server(proc: subprocess.Popen) -> None:
    for sig, timeout in ((signal.SIGTERM, 60), (signal.SIGKILL, 20)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(2)


def run_policy(policy: str, args: argparse.Namespace, output) -> None:
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_name = tempfile.mkstemp(prefix=f"policy-{policy}-", suffix=".log")
    os.close(fd)
    log_path = Path(log_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROBE}:{REPO / 'python'}:{env.get('PYTHONPATH', '')}"
    env["FREETOKEN_MOE_POLICY"] = "" if policy == "baseline" else policy
    env["FREETOKEN_GPU"] = args.gpu
    cmd = serve_cmd(args.model, args.gpu, port, args.cache, args.decode, args.hybrid_fetch)
    print(f"[bench] policy={policy} port={port} log={log_path}", flush=True)
    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pump = threading.Thread(target=lambda: log_f.write(proc.stdout.read()), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            for prompt_id, prompt in PROMPTS:
                warm = stream_chat(origin, prompt, args.decode)
                measured_rows = [stream_chat(origin, prompt, args.decode) for _ in range(args.repeats)]
                for repeat, measured in enumerate(measured_rows):
                    row = {
                        "policy": policy,
                        "prompt_id": prompt_id,
                        "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
                        "repeat": repeat,
                        "warm": warm,
                        "measured": measured,
                        "gpu": args.gpu,
                        "cache": args.cache,
                        "decode": args.decode,
                        "hybrid_fetch": args.hybrid_fetch,
                        "server_log": str(log_path),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    m = measured
                    print(
                        f"  {prompt_id:6s} r{repeat} ttft={m['ttft_ms']:.1f}ms "
                        f"tok/s={m['decode_tok_s']:.2f} ms/tok={m['ms_per_token']:.3f} "
                        f"sha1={m['text_sha1'][:12]}",
                        flush=True,
                    )
        finally:
            stop_server(proc)
            pump.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("FREETOKEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--gpu", default=os.environ.get("FREETOKEN_GPU", "1"))
    parser.add_argument("--cache", type=int, default=7200)
    parser.add_argument("--decode", type=int, default=255)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hybrid-fetch", type=int, default=-1)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-timeout", type=float, default=900)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    for policy in policies:
        if policy not in POLICIES:
            raise SystemExit(f"unknown policy {policy}; choices={POLICIES}")
    with args.output.open("w") as output:
        for policy in policies:
            run_policy(policy, args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
