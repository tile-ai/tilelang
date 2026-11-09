"""Configuration for the experimental Ascend graph compiler."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GraphConfig:
    """Immutable options captured by a compiled FX graph.

    Import ``tilelang.graph`` to register ``backend="tilelang_ascend"`` with
    Torch Dynamo. Pass these fields through ``torch.compile(options=...)`` or
    construct a backend with :func:`tilelang.graph.make_backend`.
    """

    target: str = '{"kind": "ascend", "mcpu": "dav-3510"}'
    execution_backend: str = "auto"
    tile_size: int = 1024
    num_cores: int = 32
    threads: int = 128
    vector_width: int = 0
    fusion: str = "off"
    min_speedup: float = 0.05
    max_regression: float = 0.02
    benchmark_repeats: int = 3
    benchmark_backend: str = "event"
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be a nonempty target name or JSON string")
        if self.execution_backend not in ("auto", "tvm_ffi", "pto"):
            raise ValueError("execution_backend must be 'auto', 'tvm_ffi', or 'pto'")
        for name in ("tile_size", "num_cores", "threads", "benchmark_repeats"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.vector_width, bool) or not isinstance(self.vector_width, int) or self.vector_width not in (0, 1, 4):
            raise ValueError("vector_width must be 0 (automatic), 1, or 4")
        if self.fusion not in ("off", "profile"):
            raise ValueError("fusion must be 'off' or 'profile'")
        if self.benchmark_backend not in ("event", "msprof"):
            raise ValueError("benchmark_backend must be 'event' or 'msprof'; graph wall time is measured separately")
        if self.benchmark_repeats < 3:
            raise ValueError("benchmark_repeats must be at least 3")
        for name in ("min_speedup", "max_regression"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or value >= 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be a bool")
