"""Experiment configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


# Bytes per element for the supported activation dtypes.
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp8": 1,
}

# torch dtype names, kept as strings so torch is an optional import here.
DTYPE_TORCH = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp8": "float8_e4m3fn",
}


@dataclass
class SimConfig:
    """Shape / distribution knobs for one MoE EP experiment."""

    # Model shape
    num_tokens: int = 4096          # total tokens across all ranks
    hidden: int = 4096              # hidden size
    num_experts: int = 64           # total experts across all ranks
    topk: int = 4                   # experts selected per token
    world_size: int = 2             # EP ranks (== GPUs for single-node)
    dtype: str = "bf16"
    hidden_out: Optional[int] = None  # expert output dim, defaults to hidden

    # Load distribution: "uniform" | "hotspot" | "rank_skew"
    distribution: str = "uniform"
    hotspot_frac: float = 0.5       # fraction of tokens sent to hot experts
    hotspot_gain: float = 6.0       # logit bias magnitude for hot experts
    rank_skew: float = 0.0          # 0 = balanced tokens/rank, >0 skews token counts

    # Benchmark control
    warmup: int = 10
    iters: int = 50
    seed: int = 0
    backend: str = "nccl"           # "nccl" on GPU, "gloo" for CPU smoke tests
    device: str = "cuda"            # "cuda" or "cpu"
    tag: str = "run"
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; pick one of {list(DTYPE_BYTES)}")
        if self.num_experts % self.world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")
        if self.topk > self.num_experts:
            raise ValueError("topk cannot exceed num_experts")
        if self.hidden_out is None:
            self.hidden_out = self.hidden

    # ---- derived quantities -------------------------------------------------
    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.world_size

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def torch_dtype(self) -> str:
        return DTYPE_TORCH[self.dtype]

    def tokens_per_rank(self) -> list[int]:
        """Token counts per rank, honoring `rank_skew`."""
        base = self.num_tokens // self.world_size
        counts = [base] * self.world_size
        counts[-1] += self.num_tokens - base * self.world_size
        if self.rank_skew > 0 and self.world_size > 1:
            # Push extra tokens onto rank 0 to emulate a hot rank.
            shift = int(base * min(self.rank_skew, 0.9))
            take = min(shift, counts[-1])
            counts[0] += take
            counts[-1] -= take
        return counts

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        if unknown:
            kwargs.setdefault("extras", {}).update(unknown)
        return cls(**kwargs)


def load_config(path: str) -> SimConfig:
    """Load a YAML (preferred) or JSON config file."""
    with open(path) as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    return SimConfig.from_dict(data or {})
