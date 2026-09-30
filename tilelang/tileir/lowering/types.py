"""Internal data types shared by TileIR lowering helpers."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import tirx


@dataclass(frozen=True)
class TileIRLoweringOptions:
    """CUDA Tile IR codegen knobs threaded into the structured TileIR lowering.

    ``fast_math``, ``opt_level`` and ``disable_tma`` are compile directives sourced
    from ``pass_configs`` (global to a compile, like the CUDA backend treats
    ``tl.enable_fast_math`` / ``tl.disable_tma_lower``). ``disable_tma`` mirrors the
    CUDA backend: when set it forces ``allow_tma=False`` on every copy's load/store
    hint, OR-ed with the per-copy ``T.copy(..., disable_tma=True)`` annotation.
    ``num_ctas`` and ``occupancy`` are launch-level CUDA Tile IR entry hints sourced
    from ``T.Kernel`` annotations, so an autotuner config can sweep them by passing
    them as ordinary kernel arguments.

    ``num_ctas`` maps to the entry's ``num_cta_in_cga`` (the CTAs in a thread-block
    cluster / CGA); ``occupancy`` maps to the per-SM CTA occupancy hint;
    ``num_worker_warps`` maps to the entry's ``num_worker_warps_per_cta`` hint
    (only 4 or 8 are valid per spec 13.3). ``None`` leaves the choice to the
    assembler.

    ``hints`` is the per-arch ``optimization_hints`` dictionary sourced from the
    ``T.Kernel(tileir_hints=...)`` annotation, stored as a hashable nested tuple
    ``((arch_key, ((hint_key, value), ...)), ...)`` — e.g.
    ``(("sm_100", (("num_cta_in_cga", 2),)), ("sm_120", (("num_cta_in_cga", 4),)))``.
    ``allow_tma`` values are Python bools; all other hint values are ints.
    Mutually exclusive with the single-value knobs above (``_lowering_options``
    raises on conflict). ``None`` means no per-arch hints.
    """

    fast_math: bool = False
    opt_level: int = 3
    num_ctas: int | None = None
    occupancy: int | None = None
    num_worker_warps: int | None = None
    disable_tma: bool = False
    hints: tuple[tuple[str, tuple[tuple[str, int | bool], ...]], ...] | None = None


@dataclass(frozen=True)
class TileLayout:
    forward_vars: tuple[tirx.Var, ...]
    forward_indices: tuple[tirx.PrimExpr, ...]
    output_shape: tuple[int, ...]
    output_strides: tuple[int, ...]
