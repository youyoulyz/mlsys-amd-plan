from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from run_policy_bench import (  # noqa: E402
    POLICIES,
    REPO,
    PROBE,
    free_port,
    serve_cmd,
    stop_server,
    stream_chat,
    wait_ready,
)


SYSTEM_STYLE = (
    "请直接作答，不要输出分析过程。最后一行必须严格为 `ANSWER: <答案>`，"
    "答案后不要再写任何内容。\n\n"
)

TASKS = [
    {
        "id": "math_add",
        "family": "math",
        "prompt": "17 + 25 等于多少？",
        "expected": "42",
        "scorer": "integer",
    },
    {
        "id": "math_expert_capacity",
        "family": "math",
        "prompt": (
            "一块 GPU 剩余 2.5 GiB 显存，每个专家占 0.003 GiB。"
            "最多还能放多少个完整专家？向下取整。"
        ),
        "expected": "833",
        "scorer": "integer",
    },
    {
        "id": "math_expert_total",
        "family": "math",
        "prompt": "一个 MoE 有 40 层，每层 256 个专家。总专家实例数是多少？",
        "expected": "10240",
        "scorer": "integer",
    },
    {
        "id": "python_square_sum",
        "family": "python",
        "prompt": "Python 表达式 sum([i*i for i in range(5)]) 的结果是多少？",
        "expected": "30",
        "scorer": "integer",
    },
    {
        "id": "python_append_len",
        "family": "python",
        "prompt": "执行 x=[1,2,3]; x.append(4) 后，len(x) 是多少？",
        "expected": "4",
        "scorer": "integer",
    },
    {
        "id": "python_lookup",
        "family": "python",
        "prompt": (
            "在 100 万键上反复做 membership test，list 和 dict 哪个平均复杂度更低？"
            "只回答 list 或 dict。"
        ),
        "expected": "dict",
        "scorer": "label",
        "aliases": ["dict", "dictionary", "字典"],
    },
    {
        "id": "moe_policy_choice",
        "family": "moe",
        "prompt": (
            "小显存 MoE 专家缓存中，如果近期访问频率能可靠预测后续复用，"
            "LRU 和 random replacement 优先选哪个？只回答 LRU 或 random。"
        ),
        "expected": "LRU",
        "scorer": "label",
        "aliases": ["lru", "leastrecentlyused", "最近最少使用"],
    },
    {
        "id": "moe_high_floor",
        "family": "moe",
        "prompt": (
            "Top-8 路由中，在卡专家权重总和已经达到 0.85。"
            "此时单个未在卡专家权重能否大于 0.30？只回答 yes 或 no。"
        ),
        "expected": "no",
        "scorer": "label",
        "aliases": ["no", "否", "不能"],
    },
    {
        "id": "moe_topk_routes",
        "family": "moe",
        "prompt": "单个 token 经过 40 层 MoE，每层选 8 个专家，一共产生多少条 expert route？",
        "expected": "320",
        "scorer": "integer",
    },
]


def answer_text(text: str) -> str:
    matches = re.findall(r"ANSWER\s*[:：]\s*(.*)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def normalize(text: str) -> str:
    return re.sub(r"[\s`'\"。，,.;；:：()\[\]{}]", "", text).lower()


def score(answer: str, task: dict) -> tuple[bool, str]:
    kind = task["scorer"]
    expected = task["expected"]
    if kind == "integer":
        values = [int(x) for x in re.findall(r"-?\d+", answer.replace(",", ""))]
        return (bool(values) and values[-1] == int(expected), str(values[-1]) if values else "")
    if kind == "digit_set":
        values = [int(x) for x in re.findall(r"\d{3}", answer)]
        want = [int(x) for x in expected.split(",")]
        return (set(values) == set(want), ",".join(str(x) for x in sorted(set(values))))
    if kind == "int_pair":
        values = [int(x) for x in re.findall(r"\d+", answer)]
        want = [int(x) for x in expected.split()]
        return (values[:2] == want, " ".join(str(x) for x in values[:2]))
    if kind == "label":
        norm = normalize(answer)
        for alias in task.get("aliases", [expected]):
            if normalize(alias) in norm:
                return True, answer
        return False, answer
    raise ValueError(kind)


def run_policy(policy: str, args: argparse.Namespace, output) -> None:
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_name = tempfile.mkstemp(prefix=f"accuracy-{policy}-", suffix=".log")
    os.close(fd)
    log_path = Path(log_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROBE}:{REPO / 'python'}:{env.get('PYTHONPATH', '')}"
    env["FREETOKEN_MOE_POLICY"] = "" if policy == "baseline" else policy
    env["FREETOKEN_GPU"] = args.gpu
    cmd = serve_cmd(args.model, args.gpu, port, args.cache, args.max_tokens, args.hybrid_fetch)
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    print(f"[accuracy] policy={policy} port={port} log={log_path}", flush=True)
    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pump = threading.Thread(target=lambda: log_f.write(proc.stdout.read()), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            for task in tasks:
                prompt = SYSTEM_STYLE + task["prompt"]
                warm = stream_chat(
                    origin,
                    prompt,
                    args.max_tokens,
                    enable_thinking=args.thinking,
                    ignore_eos=False,
                )
                for repeat in range(args.repeats):
                    measured = stream_chat(
                        origin,
                        prompt,
                        args.max_tokens,
                        enable_thinking=args.thinking,
                        ignore_eos=False,
                    )
                    answer = answer_text(measured["text"])
                    passed, predicted = score(answer, task)
                    row = {
                        "policy": policy,
                        "task_id": task["id"],
                        "family": task["family"],
                        "repeat": repeat,
                        "expected": task["expected"],
                        "predicted": predicted,
                        "answer_text": answer,
                        "passed": passed,
                        "warm": warm,
                        "measured": measured,
                        "gpu": args.gpu,
                        "cache": args.cache,
                        "max_tokens": args.max_tokens,
                        "hybrid_fetch": args.hybrid_fetch,
                        "server_log": str(log_path),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        f"  {task['id']:20s} r{repeat} pass={int(passed)} "
                        f"answer={predicted!r} sha1={measured['text_sha1'][:10]}",
                        flush=True,
                    )
        finally:
            stop_server(proc)
            pump.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("FREETOKEN_MODEL", "/home/luyzh/models/Qwen3.6-35B-A3B-NVFP4"))
    parser.add_argument("--gpu", default=os.environ.get("FREETOKEN_GPU", "1"))
    parser.add_argument("--cache", type=int, default=7200)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hybrid-fetch", type=int, default=-1)
    parser.add_argument("--policies", default="baseline,top85,top90,floor,combined,layer")
    parser.add_argument("--tasks", default="", help="comma-separated task ids; empty means all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-timeout", type=float, default=900)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    args.tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    for policy in policies:
        if policy not in POLICIES:
            raise SystemExit(f"unknown policy {policy}; choices={POLICIES}")
    with args.output.open("w") as output:
        for policy in policies:
            run_policy(policy, args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
