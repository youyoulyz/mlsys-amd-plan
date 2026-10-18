import os

from freetoken.server.args import parse_args

_orig_parse_args = parse_args


def _parse_args(*args, **kwargs):
    server_args, run_shell = _orig_parse_args(*args, **kwargs)
    return server_args, run_shell


import freetoken.server.args as args_mod
args_mod.parse_args = _parse_args
from freetoken.server.launch import launch_server

if __name__ == "__main__":
    launch_server(argv=[
        "--model", os.environ.get("FREETOKEN_MODEL", "/home/luyzh/models/Qwen3.6-35B-A3B-NVFP4"),
        "--gpu", os.environ.get("FREETOKEN_GPU", "1"),
        "--host", "127.0.0.1",
        "--port", "1919",
        "--text-model-only",
        "--served-model-name", "Qwen3.6-35B-A3B-NVFP4",
        "--moe-strategy", "hybrid",
        "--moe-cache-size", "7200",
        "--num-tokens", "65536",
        "--max-running-requests", "1",
        "--cuda-graph-max-bs", "0",
    ])
