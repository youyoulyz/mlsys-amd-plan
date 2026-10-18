"""deepep_sim: a minimal, readable MoE expert-parallel (EP) dispatch/combine simulator.

This package re-implements the *semantics* of DeepEP's dispatch/combine path in
plain PyTorch so that the communication/cost behaviour can be studied on small
clusters (e.g. 2x V100) without the Hopper/Blackwell-only CUDA kernels.

It is a teaching / method-prototype tool, NOT a DeepEP replacement. Absolute
bandwidth/latency numbers must not be extrapolated to modern hardware.
"""

from .config import SimConfig
from .routing import route_tokens, make_routing_logits
from .permutation import group_by_expert, scatter_back
from .cost_model import CostModel

__all__ = [
    "SimConfig",
    "route_tokens",
    "make_routing_logits",
    "group_by_expert",
    "scatter_back",
    "CostModel",
]

__version__ = "0.1.0"
