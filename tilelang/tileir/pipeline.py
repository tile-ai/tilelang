"""End-to-end lowering from TileLang TIR to CUDA Tile IR MLIR.

The pipeline converts TIR to SemanticIR, lowers it to typed TileIR, runs the
TileIR passes, and emits a ``cuda_tile`` MLIR module. Entry arguments retain
the order and identity of ``root.params`` throughout lowering and emission.
"""

from __future__ import annotations

import re
from typing import Any
from dataclasses import replace

from tilelang.tileir.errors import TileIRLoweringError
from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.ir.types import MemSpace
from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
from tilelang.tileir.lowering.sem_to_ir import lower_kernel
from tilelang.tileir.lowering.mlir_emit import emit_module
from tilelang.tileir.passes.base import PassContext, run_pipeline
from tilelang.tileir.passes.dataflow import dataflow_pass
from tilelang.tileir.passes.token_order import token_order_pass
from tilelang.tileir.passes.loop_carry import loop_carry_pass
from tilelang.tileir.passes.gemm_orientation import gemm_orientation_pass
from tilelang.tileir.scratch import scratch_bytes_for_primfunc

__all__ = ["build_tileir_module", "lower_single_kernel_to_tileir"]

# Per-architecture hint keys omit Hopper and Blackwell architecture suffixes.
_HINT_ARCH_PREFIX_RE = re.compile(r"^(sm_\d+)")


def _normalize_hint_arch_key(arch: str) -> str:
    """Normalize a compile-time arch string to the bare ``sm_XX`` form the
    ``tileir.hints`` per-arch dictionary keys use (``_HINT_ARCH_KEYS`` in
    ``lowering/__init__.py``)."""
    match = _HINT_ARCH_PREFIX_RE.match(arch)
    return match.group(1) if match else arch


def _validate_hints_cover_arch(hints: tuple, arch: str) -> None:
    """Require an exact or default optimization-hint entry for *arch*.

    ``hints`` is the frozen ``((arch_key, ((hint_key, value), ...)), ...)``
    representation produced from ``T.Kernel(tileir_hints=...)``. Validation
    prevents a per-architecture hint dictionary from becoming a silent no-op.
    """
    arch_key = _normalize_hint_arch_key(arch)
    covered = tuple(entry[0] for entry in hints)
    if arch_key in covered or "default" in covered:
        return
    raise TileIRLoweringError(
        f"TileIR backend: tileir.hints covers arch(es) {covered!r} but this compile's "
        f"target arch is {arch_key!r} (from {arch!r}); none of the hints dictionary's "
        "per-arch keys apply (no exact match and no 'default' fallback), so the hints "
        f"would have no effect on this compile. Add a {arch_key!r} entry or a "
        "'default' entry to the tileir.hints dictionary, or drop it if it is not meant "
        "for this arch."
    )


def build_tileir_module(
    prim_func: Any,
    *,
    kernel_name: str | None = None,
    arch: str | None = None,
    num_cta: int | None = None,
    occupancy: int | None = None,
    num_worker_warps: int | None = None,
    hints: tuple | None = None,
    fast_math: bool = False,
    disable_tma: bool = False,
) -> Any:
    """Convert a TileLang ``PrimFunc`` to a ``cuda_tile`` MLIR module.

    Parameters
    ----------
    prim_func :
        A TVM TIR ``PrimFunc`` produced by the TileLang frontend (i.e.
        traced via ``T.prim_func`` / ``tilelang.jit``).
    kernel_name : str | None
        Symbol name for the entry function.  Defaults to the
        ``global_symbol`` attribute of *prim_func*, falling back to
        ``"kernel"``.
    arch : str | None
        Target SM architecture string (e.g. ``"sm_90a"``).  Pass
        ``None`` (the default) to omit the arch hint — useful for tests
        that do not target a real GPU.
    num_cta : int | None
        Number of CTAs in a CGA (thread-block cluster).  Forwarded to
        ``emit_module`` as the ``num_cta`` entry hint.  ``None`` omits
        the hint.
    occupancy : int | None
        Per-SM CTA occupancy hint.  Forwarded to ``emit_module``.
        ``None`` omits the hint.
    num_worker_warps : int | None
        Number of worker warps per CTA, in {4, 8}. Forwarded to
        ``emit_module`` as the ``num_worker_warps`` entry hint.  ``None``
        omits the hint.
    hints : tuple | None
        Per-arch ``optimization_hints`` dictionary in the frozen
        ``((arch_key, ((hint_key, value), ...)), ...)`` form produced by
        ``_lowering_options`` (from ``T.Kernel(tileir_hints=...)``).
        Forwarded to ``emit_module``; mutually exclusive with the
        single-value ``num_cta`` / ``occupancy`` / ``num_worker_warps``
        knobs.  ``None`` omits the per-arch dictionary.

    Returns
    -------
    cuda_tile.ModuleOp
        The emitted MLIR module.

    Raises
    ------
    ImportError
        If ``cuda_tile._mlir`` Python bindings are unavailable.
    ValueError
        If *prim_func* does not contain exactly one kernel. Multi-kernel
        programs must use ``lower_primfunc_to_tileir``, which splits and
        assembles every kernel.
    TileIRLoweringError
        If *hints* and *arch* are both given and *hints* has no entry
        covering *arch* (nor a ``"default"`` fallback) -- see
        ``_validate_hints_cover_arch``.
    """
    if hints is not None and arch is not None:
        _validate_hints_cover_arch(hints, arch)

    program = tir_to_sem(prim_func)
    if len(program.kernels) != 1:
        raise ValueError(
            "build_tileir_module requires exactly one kernel; "
            f"tir_to_sem found {len(program.kernels)}. "
            "Use lower_primfunc_to_tileir for multi-kernel programs."
        )
    kernel = program.kernels[0]

    if kernel_name is None:
        attrs = getattr(prim_func, "attrs", None)
        if attrs is not None and hasattr(attrs, "get"):
            gs = attrs.get("global_symbol", "")
            kernel_name = str(gs) if gs else None
        if not kernel_name:
            kernel_name = "kernel"

    builder = IRBuilder()
    root = lower_kernel(kernel, builder, program=program, fast_math=fast_math)

    # Distinct alias bits prevent unrelated buffers from acquiring false token
    # dependencies. CUDA entry buffers retain their 16-byte alignment contract.
    param_constraints: dict[int, dict] = {}
    for pv in root.params:
        if hasattr(pv, "type") and getattr(pv.type, "space", None) == MemSpace.GLOBAL:
            param_constraints[pv.id] = {"div_by": 16}
    for av in getattr(root, "alloc_buffers", []):
        if hasattr(av, "type"):
            space = getattr(av.type, "space", None)
            if space in (MemSpace.SHARED, MemSpace.REGISTER):
                param_constraints[av.id] = {"div_by": 1, "alias_distinct": True}

    pass_ctx = PassContext()
    pass_ctx.results["param_constraints"] = param_constraints
    pass_ctx.results["target_arch"] = arch
    pass_ctx = run_pipeline(
        root,
        [gemm_orientation_pass, dataflow_pass, token_order_pass, loop_carry_pass],
        ctx=pass_ctx,
    )

    entry_args = [(value.name or "arg", value.type) for value in root.params]

    # The pass result lets emission use precise RAW/WAW dependencies instead of
    # a conservative per-buffer token chain.
    token_plan = pass_ctx.results.get("token_order")
    return emit_module(
        root,
        kernel_name=kernel_name,
        entry_args=entry_args,
        arch=arch,
        num_cta=num_cta,
        occupancy=occupancy,
        num_worker_warps=num_worker_warps,
        hints=hints,
        token_plan=token_plan,
        disable_tma=disable_tma,
        fast_math=fast_math,
    )


def lower_single_kernel_to_tileir(
    prim_func: Any,
    target: Any,
    toolchain: Any | None = None,
    pass_configs: dict[str, Any] | None = None,
) -> Any:
    """Lower and assemble a single-kernel TileLang PrimFunc through TileIR.

    This is the route for all TileIR PrimFuncs.  It calls ``build_tileir_module``
    (TIR → SemanticIR → TileIR Block → passes → MLIR) and then delegates to
    the shared ``assemble_tileir_module`` helper from
    ``tilelang.tileir.lowering`` so the cubin + launch metadata are packaged
    correctly.
    """
    # ``lowering.__init__`` imports this module and owns the assembly helpers.
    from tvm.target import Target  # type: ignore[import]
    from tilelang.tileir.checks import check_tileir_available
    from tilelang.tileir.launch import extract_launch_metadata, _argument_names
    from tilelang.tileir.lowering import assemble_tileir_module

    if toolchain is None:
        toolchain = check_tileir_available()

    if isinstance(target, str):
        target = Target(target)

    arch: str | None = None
    target_arch_raw = getattr(target, "arch", None)
    if target_arch_raw is None:
        attrs = getattr(target, "attrs", None)
        if attrs is not None and "arch" in attrs:
            target_arch_raw = str(attrs["arch"])
    if target_arch_raw:
        arch = str(target_arch_raw)

    cfg = pass_configs or {}

    attrs = getattr(prim_func, "attrs", None)
    if attrs is not None and hasattr(attrs, "get"):
        gs = attrs.get("global_symbol", "")
        kernel_name = str(gs) if gs else "kernel"
    else:
        kernel_name = "kernel"

    # Option validation is centralized in ``_lowering_options``.
    from tilelang.tileir.lowering import _lowering_options

    options = _lowering_options(prim_func, cfg)
    num_cta = options.num_ctas
    occupancy_hint = options.occupancy
    num_worker_warps_hint = options.num_worker_warps
    hints = options.hints
    disable_tma = options.disable_tma
    fast_math = options.fast_math
    opt_level = options.opt_level

    mlir_module = build_tileir_module(
        prim_func,
        kernel_name=kernel_name,
        arch=arch,
        num_cta=num_cta,
        occupancy=occupancy_hint,
        num_worker_warps=num_worker_warps_hint,
        hints=hints,
        fast_math=fast_math,
        disable_tma=disable_tma,
    )

    launch_metadata = extract_launch_metadata(prim_func)
    argument_names = _argument_names(prim_func)

    artifact = assemble_tileir_module(
        mlir_module,
        kernel_name=kernel_name,
        target=target,
        toolchain=toolchain,
        launch_metadata=launch_metadata,
        argument_names=argument_names,
        opt_level=opt_level,
    )
    return replace(artifact, scratch_bytes_per_block=scratch_bytes_for_primfunc(prim_func))
