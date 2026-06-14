"""TileSight integration for TileLang.

This package is intentionally independent from the legacy ``TileSight/`` tree.
It exposes a structured TileLang TIR graph extractor and a refactored cost
model whose public entry only needs:

1. a :class:`KernelGraph`
2. a :class:`HardwareSpec`
"""

from __future__ import annotations

from .arch import HardwareSpec
from .cost_model import CostModelResult, estimate_cost
from .extractor import extract_kernel_graph
from .graph import KernelGraph


def analyze_module(mod, target):
    """Extract a TileSight graph from ``mod`` and run the cost model.

    The cost model itself only sees the graph and hardware specification; this
    wrapper is the pipeline-facing adapter.
    """

    graph = extract_kernel_graph(mod)
    hardware = HardwareSpec.from_target(target)
    result = estimate_cost(graph, hardware)
    return graph, result


__all__ = [
    "CostModelResult",
    "HardwareSpec",
    "KernelGraph",
    "analyze_module",
    "estimate_cost",
    "extract_kernel_graph",
]
