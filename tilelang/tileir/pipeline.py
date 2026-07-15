"""End-to-end TileIR pipeline.

Provides a single public function that converts a TileLang ``PrimFunc``
into a ``cuda_tile`` MLIR module, bridging all five TileIR layers:

    TIR → SemanticIR → TileIR Block → passes → MLIR

Usage::

    module = build_tileir_module(prim_func)
    print(str(module))          # MLIR text

Entry-arg / root.params alignment
----------------------------------
``emit_module`` requires ``root.params[i] ↔ entry_args[i]`` to be
position-aligned so that each entry-function block argument is bound to
the correct TileIR ``Value``.

``lower_kernel`` now sets ``builder.block.params`` from the GLOBAL
parameter buffer ``Value`` objects in ``program.params`` order (the
Values are the EXACT same objects referenced by the ops, since they come
from ``LoweringScope._buffers``).  This module simply reads those params
and builds the matching ``entry_args`` list.
"""

from __future__ import annotations

import re
from typing import Any

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
from tilelang.tileir.lowering.sem_to_ir import lower_kernel
from tilelang.tileir.lowering.mlir_emit import emit_module
from tilelang.tileir.passes.base import run_pipeline
from tilelang.tileir.passes.dataflow import dataflow_pass
from tilelang.tileir.passes.token_order import token_order_pass
from tilelang.tileir.passes.loop_carry import loop_carry_pass
from tilelang.tileir.passes.gemm_orientation import gemm_orientation_pass

__all__ = ["build_tileir_module", "lower_single_kernel_to_tileir"]

# Matches the bare "sm_XX" prefix of an arch string, stripping any
# Hopper/Blackwell variant suffix (e.g. "sm_90a" -> "sm_90", "sm_100a" ->
# "sm_100"; see mlir_emit.py's `self.arch: str | None  # e.g. "sm_90a"`).
_HINT_ARCH_PREFIX_RE = re.compile(r"^(sm_\d+)")


def _normalize_hint_arch_key(arch: str) -> str:
    """Normalize a compile-time arch string to the bare ``sm_XX`` form the
    ``tileir.hints`` per-arch dictionary keys use (``_HINT_ARCH_KEYS`` in
    ``lowering/__init__.py``)."""
    match = _HINT_ARCH_PREFIX_RE.match(arch)
    return match.group(1) if match else arch


def _validate_hints_cover_arch(hints: tuple, arch: str) -> None:
    """Raise ``TileIRLoweringError`` if *hints* has no entry for *arch* (nor a
    ``"default"`` fallback).

    ``hints`` is the frozen ``((arch_key, ((hint_key, value), ...)), ...)``
    form ``_lowering_options`` produces from ``T.Kernel(tileir_hints=...)``.
    The CUDA Tile IR assembler picks whichever per-arch sub-dict matches the
    ACTUAL compile arch (falling back to ``"default"`` if present); if
    neither is present, the hints dictionary the user wrote has NO EFFECT on
    this specific compile at all -- a silent no-op the user cannot detect.
    Validating here, in ``build_tileir_module`` (which is the one place both
    the resolved ``hints`` tuple and the resolved compile ``arch`` are always
    simultaneously in hand as plain parameters -- ``_lowering_options``
    itself never learns the compile arch, and ``lower_single_kernel_to_tileir``
    would need to duplicate this check for every OTHER direct caller of
    ``build_tileir_module``, e.g. the existing ``test_tileir_hints_*_mlir``
    tests), catches it at the earliest possible point instead of silently
    compiling a kernel whose hints were dead on arrival.
    """
    arch_key = _normalize_hint_arch_key(arch)
    covered = tuple(entry[0] for entry in hints)
    if arch_key in covered or "default" in covered:
        return
    from tilelang.tileir.errors import TileIRLoweringError

    raise TileIRLoweringError(
        f"TileIR backend: tileir.hints covers arch(es) {covered!r} but this compile's "
        f"target arch is {arch_key!r} (from {arch!r}); none of the hints dictionary's "
        "per-arch keys apply (no exact match and no 'default' fallback), so the hints "
        f"would silently have NO EFFECT on this compile. Add a {arch_key!r} entry or a "
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
    # 0. Validate the per-arch hints dictionary against the actual compile
    # arch, when both are known.
    if hints is not None and arch is not None:
        _validate_hints_cover_arch(hints, arch)

    # 1. TIR → SemanticIR
    program = tir_to_sem(prim_func)
    if len(program.kernels) != 1:
        raise ValueError(
            "build_tileir_module requires exactly one kernel; "
            f"tir_to_sem found {len(program.kernels)}. "
            "Use lower_primfunc_to_tileir for multi-kernel programs."
        )
    kernel = program.kernels[0]

    # 2. Derive kernel name
    if kernel_name is None:
        attrs = getattr(prim_func, "attrs", None)
        if attrs is not None and hasattr(attrs, "get"):
            gs = attrs.get("global_symbol", "")
            kernel_name = str(gs) if gs else None
        if not kernel_name:
            kernel_name = "kernel"

    # 3. SemanticIR → TileIR Block
    # lower_kernel populates builder.block.params with the GLOBAL param buffer
    # Values (exact same objects referenced by the ops) when program is passed.
    builder = IRBuilder()
    root = lower_kernel(kernel, builder, program=program, fast_math=fast_math)

    # 4. Run passes
    # Seed dataflow_pass with per-buffer param constraints so each
    # GLOBAL buffer param gets a distinct alias bit (instead of ALIAS_UNIVERSE
    # which would make every op depend on every other op in the token_order pass).
    # We set div_by=16 (CUDA buffer alignment guarantee) for all buffer params.
    # Also seed alloc_shared / alloc_fragment buffers with distinct alias bits
    # so SHARED/REGISTER ops are also distinguishable (not ALIAS_UNIVERSE).
    from tilelang.tileir.ir.types import MemSpace as _MemSpace

    param_constraints: dict[int, dict] = {}
    for pv in root.params:
        if hasattr(pv, "type") and getattr(pv.type, "space", None) == _MemSpace.GLOBAL:
            param_constraints[pv.id] = {"div_by": 16}
    # Seed alloc_buffers (SHARED / REGISTER) with distinct alias bits.
    # The "alias_distinct" flag tells dataflow_analysis to assign a fresh bit
    # (like a block param) instead of ALIAS_UNIVERSE, so that Copy(A→A_sh) and
    # Copy(B→B_sh) are seen as independent ops by the token_order pass.
    for av in getattr(root, "alloc_buffers", []):
        if hasattr(av, "type"):
            space = getattr(av.type, "space", None)
            if space in (_MemSpace.SHARED, _MemSpace.REGISTER):
                param_constraints[av.id] = {"div_by": 1, "alias_distinct": True}
    from tilelang.tileir.passes.base import PassContext as _PassContext

    pass_ctx = _PassContext()
    pass_ctx.results["param_constraints"] = param_constraints
    pass_ctx.results["target_arch"] = arch
    pass_ctx = run_pipeline(
        root,
        [gemm_orientation_pass, dataflow_pass, token_order_pass, loop_carry_pass],
        ctx=pass_ctx,
    )

    # 5. Build entry_args aligned with root.params
    # root.params is now the positionally-ordered list of all entry Values
    # (buffers AND scalars, in PrimFunc param order).
    # entry_args is built directly from root.params (the positionally-ordered
    # entry Values) using each Value's own TileType.
    entry_args = []
    for v in root.params:
        vname = v.name if v.name else "arg"
        # Use the Value's own TileType (already correctly constructed by LoweringScope).
        entry_args.append((vname, v.type))

    # 6. Emit MLIR module
    # Pass the token_plan from the pass context so _ensure_token
    # can compute precise RAW/WAW deps instead of the conservative per-buffer chain.
    # Forward num_cta / occupancy / num_worker_warps so the entry gets the right optimization hints.
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
    # Late imports to avoid circular-import at module load time (pipeline.py is
    # imported by lowering/__init__.py; lowering/__init__.py provides
    # assemble_tileir_module and the helper utilities).
    from tvm.target import Target  # type: ignore[import]
    from tilelang.tileir.checks import check_tileir_available
    from tilelang.tileir.launch import extract_launch_metadata, _argument_names
    from tilelang.tileir.lowering import assemble_tileir_module

    if toolchain is None:
        toolchain = check_tileir_available()

    if isinstance(target, str):
        target = Target(target)

    # Derive GPU arch from target for build_tileir_module
    arch: str | None = None
    target_arch_raw = getattr(target, "arch", None)
    if target_arch_raw is None:
        attrs = getattr(target, "attrs", None)
        if attrs is not None and "arch" in attrs:
            target_arch_raw = str(attrs["arch"])
    if target_arch_raw:
        arch = str(target_arch_raw)

    cfg = pass_configs or {}

    # Derive kernel name
    attrs = getattr(prim_func, "attrs", None)
    if attrs is not None and hasattr(attrs, "get"):
        gs = attrs.get("global_symbol", "")
        kernel_name = str(gs) if gs else "kernel"
    else:
        kernel_name = "kernel"

    # Read & VALIDATE all lowering options through _lowering_options()
    # (tilelang/tileir/lowering/__init__.py).
    # This runs validation: num_ctas must be a power-of-two in [1, 16],
    # occupancy in [1, 32], opt_level in [0, 3].  Invalid hints MUST raise
    # TileIRLoweringError (do NOT swallow it) so callers see the error instead of
    # silently producing a kernel with bad launch hints.
    from tilelang.tileir.lowering import _lowering_options

    options = _lowering_options(prim_func, cfg)
    num_cta = options.num_ctas
    occupancy_hint = options.occupancy
    num_worker_warps_hint = options.num_worker_warps
    hints = options.hints
    disable_tma = options.disable_tma
    fast_math = options.fast_math
    opt_level = options.opt_level

    # Build the structured MLIR module.
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

    # Extract launch metadata from the already-materialized prim_func
    launch_metadata = extract_launch_metadata(prim_func)
    argument_names = _argument_names(prim_func)

    return assemble_tileir_module(
        mlir_module,
        kernel_name=kernel_name,
        target=target,
        toolchain=toolchain,
        launch_metadata=launch_metadata,
        argument_names=argument_names,
        opt_level=opt_level,
    )
