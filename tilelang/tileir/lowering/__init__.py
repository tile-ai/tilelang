"""Structured TileLang frontend AST to CUDA Tile IR lowering boundary.

The TileIR backend receives the high-level TileLang TIR AST before CUDA-specific
TileLang lowering rewrites it into PTX intrinsics. It must lower that AST through
structured IR APIs. Textual MLIR string generation is deliberately not a
permitted path here: it would hide type, location, and verification mistakes that the
CUDA Tile dialect APIs should catch.
"""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.target import Target

from tilelang.transform import PassConfigKey

from ..artifact import TileIRLaunchMetadata, TileIRLoweringResult
from ..assembly import (
    assemble_tileir_module as _assemble_tileir_module,
    optimize_tileir_module as _optimize_tileir_module,
    run_tool as _run_tool,
    write_tileir_bytecode as _write_tileir_bytecode,
)
from ..checks import TileIRToolchain, check_tileir_available
from ..errors import (
    TileIRLoweringError,
    TileIRLoweringNotImplementedError,
)
from ..launch import (
    _split_grid_sync_primfunc,
    _split_host_orchestrated_primfunc,
    _temporary_buffer_metadata_from_semantics,
)
from .types import TileIRLoweringOptions as _TileIRLoweringOptions
from ..semantic import (
    SemanticProgram,
    TileLangSemanticError,
    extract_semantic_program,
    materialize_launch_nest,
)
from ..tir_analysis import _format_coverage_gap


def assemble_tileir_module(
    tileir_module: Any,
    *,
    kernel_name: str,
    target: Target,
    toolchain: TileIRToolchain | None = None,
    launch_metadata: TileIRLaunchMetadata | None = None,
    argument_names: tuple[str, ...] = (),
    opt_level: int = 3,
) -> TileIRLoweringResult:
    """Assemble a lowered CUDA Tile IR module into a kernel artifact.

    Binds the optimize/write/assemble toolchain callables at the lowering
    boundary so they are wired in one place and stay a single substitution
    point. ``opt_level`` is forwarded to both the CUDA Tile IR optimizer and the
    ``tileiras`` assembler.
    """

    return _assemble_tileir_module(
        tileir_module,
        kernel_name=kernel_name,
        target=target,
        toolchain=toolchain,
        launch_metadata=launch_metadata,
        argument_names=argument_names,
        opt_level=opt_level,
        optimize=lambda module: _optimize_tileir_module(module, opt_level=opt_level),
        write_bytecode=_write_tileir_bytecode,
        run=lambda cmd: _run_tool(cmd, stage="assembly"),
    )


def _extract_semantic_program_for_lowering(prim_func: tirx.PrimFunc, target: Target) -> SemanticProgram:
    try:
        return extract_semantic_program(prim_func)
    except TileLangSemanticError as exc:
        kernel_name = str(prim_func.attrs.get("global_symbol", "main")) if prim_func.attrs else "main"
        coverage_gap = _format_coverage_gap(prim_func)
        raise TileIRLoweringNotImplementedError(
            "TileIR backend rejected the PrimFunc at the TileLang Semantic IR boundary. "
            f"kernel={kernel_name}, target={target}. {coverage_gap}. "
            f"First unsupported semantic construct: {exc}. "
            "Add an explicit Semantic IR node and a structured CUDA Tile IR lowering; do not add a fallback path."
        ) from exc


def _pass_config_value(cfg: dict[str, Any], key: PassConfigKey, default: Any) -> Any:
    """Read a pass-config value tolerating both enum and raw-string keys."""
    return cfg.get(key, cfg.get(key.value, default))


def _kernel_launch_annotations(prim_func: tirx.PrimFunc) -> dict[str, Any]:
    """Collect TileLang kernel-launch block annotations (e.g. ``T.Kernel`` hints).

    ``T.Kernel`` stores launch-level metadata as annotations on the kernel-launch
    ``SBlock`` (the same mechanism as ``cluster_dims``), not on ``prim_func.attrs``.
    """
    annotations: dict[str, Any] = {}

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.annotations:
            annotations.update(dict(node.annotations))

    tirx.stmt_functor.post_order_visit(prim_func.body, visit)
    return annotations


def _as_opt_int(value: Any, *, field: str, minimum: int, maximum: int) -> int | None:
    """Coerce an annotation/pass-config value to an int in ``[minimum, maximum]``."""
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TileIRLoweringError(f"TileIR backend requires an integer {field}; got {value!r}.") from exc
    if result < minimum or result > maximum:
        raise TileIRLoweringError(f"TileIR backend requires {field} in [{minimum}, {maximum}]; got {result}.")
    return result


# Architecture keys accepted in a `tileir.hints` per-architecture dictionary.
# Validation is restricted to the targets supported by this backend.
_HINT_ARCH_KEYS: tuple[str, ...] = ("sm_90", "sm_100", "sm_103", "sm_110", "sm_120", "sm_121", "default")

# Per-hint-key validators for `tileir.hints`, matching the single-value
# validations in `_lowering_options`.
#
# Entry-scoped hint keys (on T.Kernel):
#   num_cta_in_cga           pow2 in [1, 16]   (== the tileir.num_ctas knob)
#   num_worker_warps_per_cta in {4, 8}         (== the tileir.num_worker_warps knob)
#   occupancy                in [1, 32]        (== the tileir.occupancy knob)
#
# Load/store-scoped hint keys belong on per-copy T.copy hints, not entries.
#   allow_tma                bool (LOAD/STORE ONLY — rejected on entry)
#   latency                  in [1, 10] (LOAD/STORE ONLY — rejected on entry)
_HINT_KEY_SPECS: dict[str, dict[str, Any]] = {
    "num_cta_in_cga": {"minimum": 1, "maximum": 16, "power_of_two": True},
    "num_worker_warps_per_cta": {"choices": (4, 8)},
    "occupancy": {"minimum": 1, "maximum": 32},
}

# Load/Store-scoped keys that must be rejected on entry (per CUDA Tile IR spec).
_LOAD_STORE_SCOPED_KEYS: frozenset[str] = frozenset(("allow_tma", "latency"))


def _validated_hints(raw_hints: Any) -> tuple[tuple[str, tuple[tuple[str, int | bool], ...]], ...]:
    """Validate a raw ``tileir.hints`` annotation into the frozen options form.

    ``raw_hints`` is the nested str->(str->int) mapping stored by
    ``T.Kernel(tileir_hints=...)`` (a TVM ``Map`` after the annotation
    round-trip; bools were encoded as ints at trace time). Returns the
    hashable ``((arch_key, ((hint_key, value), ...)), ...)`` representation
    with both arch keys and hint keys sorted alphabetically ("default" always
    last) to ensure stable rendering.

    Rejects load/store-scoped keys (allow_tma, latency) that have no effect on
    the kernel entry; these must use per-copy T.copy hints instead.
    """
    validated: list[tuple[str, tuple[tuple[str, int | bool], ...]]] = []
    for arch_key, arch_hints in raw_hints.items():
        arch_key = str(arch_key)
        if arch_key not in _HINT_ARCH_KEYS:
            raise TileIRLoweringError(
                f"TileIR backend does not recognize tileir.hints arch key {arch_key!r}; allowed arch keys: {', '.join(_HINT_ARCH_KEYS)}."
            )
        entries: list[tuple[str, int | bool]] = []
        for hint_key, hint_value in arch_hints.items():
            hint_key = str(hint_key)
            # Reject load/store-scoped keys on entry.
            if hint_key in _LOAD_STORE_SCOPED_KEYS:
                raise TileIRLoweringError(
                    f"TileIR backend hint {hint_key!r} is load/store-scoped in CUDA Tile IR and has "
                    f"no effect on the kernel entry; use the per-copy mechanism instead "
                    f"(T.copy(disable_tma=...) / T.copy(latency=...))."
                )
            spec = _HINT_KEY_SPECS.get(hint_key)
            if spec is None:
                raise TileIRLoweringError(
                    f"TileIR backend does not recognize tileir.hints hint key {hint_key!r} "
                    f"(arch {arch_key!r}); allowed hint keys: {', '.join(_HINT_KEY_SPECS)}."
                )
            field = f"tileir.hints[{arch_key}].{hint_key}"
            if "choices" in spec:
                choices = spec["choices"]
                value = _as_opt_int(hint_value, field=field, minimum=min(choices), maximum=max(choices))
                if value not in choices:
                    raise TileIRLoweringError(f"TileIR backend requires {field} to be one of {choices}; got {value}.")
            else:
                value = _as_opt_int(hint_value, field=field, minimum=spec["minimum"], maximum=spec["maximum"])
                if spec.get("power_of_two") and (value & (value - 1)) != 0:
                    raise TileIRLoweringError(
                        f"TileIR backend requires {field} to be a power of two in [{spec['minimum']}, {spec['maximum']}]; got {value}."
                    )
            entries.append((hint_key, value))
        # Sort hint keys alphabetically for canonical ordering.
        entries.sort(key=lambda x: x[0])
        validated.append((arch_key, tuple(entries)))
    # Sort arch keys alphabetically, with "default" always last.
    validated.sort(key=lambda x: (x[0] == "default", x[0]))
    return tuple(validated)


def _lowering_options(prim_func: tirx.PrimFunc, pass_configs: dict[str, Any] | None) -> _TileIRLoweringOptions:
    cfg = pass_configs or {}
    annotations = _kernel_launch_annotations(prim_func)

    raw_hints = annotations.get("tileir.hints")
    if raw_hints is not None:
        conflicting = [
            knob for knob in ("tileir.num_ctas", "tileir.occupancy", "tileir.num_worker_warps") if annotations.get(knob) is not None
        ]
        if conflicting:
            raise TileIRLoweringError(
                "TileIR backend does not accept the per-arch tileir.hints dictionary together with "
                f"single-value entry knobs ({', '.join(conflicting)}); move the single-value knobs into "
                "the per-arch dictionary (num_ctas -> num_cta_in_cga, num_worker_warps -> "
                "num_worker_warps_per_cta, occupancy -> occupancy)."
            )
    hints = _validated_hints(raw_hints) if raw_hints is not None else None

    num_ctas = _as_opt_int(annotations.get("tileir.num_ctas"), field="num_ctas", minimum=1, maximum=16)
    if num_ctas is not None and (num_ctas & (num_ctas - 1)) != 0:
        raise TileIRLoweringError(f"TileIR backend requires num_ctas to be a power of two in [1, 16]; got {num_ctas}.")
    occupancy = _as_opt_int(annotations.get("tileir.occupancy"), field="occupancy", minimum=1, maximum=32)
    num_worker_warps = _as_opt_int(annotations.get("tileir.num_worker_warps"), field="num_worker_warps", minimum=4, maximum=8)
    if num_worker_warps is not None and num_worker_warps not in (4, 8):
        raise TileIRLoweringError(f"TileIR backend requires num_worker_warps to be 4 or 8; got {num_worker_warps}.")

    opt_level = _as_opt_int(
        _pass_config_value(cfg, PassConfigKey.TL_TILEIR_OPT_LEVEL, 3),
        field="opt_level",
        minimum=0,
        maximum=3,
    )

    return _TileIRLoweringOptions(
        fast_math=bool(_pass_config_value(cfg, PassConfigKey.TL_ENABLE_FAST_MATH, False)),
        opt_level=3 if opt_level is None else opt_level,
        num_ctas=num_ctas,
        occupancy=occupancy,
        num_worker_warps=num_worker_warps,
        disable_tma=bool(_pass_config_value(cfg, PassConfigKey.TL_DISABLE_TMA_LOWER, False)),
        hints=hints,
    )


def lower_primfunc_to_tileir(
    prim_func: tirx.PrimFunc,
    target: Target,
    toolchain: TileIRToolchain | None = None,
    pass_configs: dict[str, Any] | None = None,
) -> TileIRLoweringResult:
    """Lower a TileLang PrimFunc to CUDA Tile IR.

    The lowering boundary is TileLang TIR AST -> TileLang Semantic IR -> CUDA
    Tile IR. The semantic pass must recognize every TileLang statement before
    the CUDA Tile IR builder sees it; unsupported semantics fail here instead
    of falling through to CUDA/PTX lowering or textual MLIR.

    Single-kernel and multi-kernel programs use ``build_tileir_module``:
    TIR → SemanticIR → TileIR Block → passes → MLIR.
    """

    if toolchain is None:
        toolchain = check_tileir_available()

    # Materialize the T.Kernel launch nest (kThreadBinding For loops -> thread_extent
    # AttrStmts) before the grid-sync split runs: _split_grid_sync_primfunc only
    # recognizes the thread_extent form, so on the raw frontend PrimFunc it is a
    # no-op and tl.sync_grid would survive into semantic extraction.
    prim_func = materialize_launch_nest(prim_func)
    prim_func = _split_grid_sync_primfunc(prim_func)
    kernel_name = str(prim_func.attrs.get("global_symbol", "main"))
    semantic_program = _extract_semantic_program_for_lowering(prim_func, target)
    launch_count = len(semantic_program.kernels)

    if launch_count == 1:
        from tilelang.tileir.pipeline import lower_single_kernel_to_tileir

        return lower_single_kernel_to_tileir(prim_func, target, toolchain=toolchain, pass_configs=pass_configs)

    elif launch_count < 1:
        raise TileIRLoweringError(f"TileIR backend expected at least one TileLang kernel launch in `{kernel_name}`.")

    temporary_buffers = _temporary_buffer_metadata_from_semantics(semantic_program.global_alloc_buffers)
    split_kernels = _split_host_orchestrated_primfunc(prim_func)
    # Route each split sub-kernel through the pipeline.
    # Each sub-kernel is a single-kernel PrimFunc produced by
    # _split_host_orchestrated_primfunc; lower_single_kernel_to_tileir handles exactly
    # that shape.
    from tilelang.tileir.pipeline import lower_single_kernel_to_tileir

    lowered_kernels = tuple(
        lower_single_kernel_to_tileir(
            kernel,
            target,
            toolchain=toolchain,
            pass_configs=pass_configs,
        )
        for kernel in split_kernels
    )
    tileir_source = "\n\n".join(kernel.tileir_source or "" for kernel in lowered_kernels)
    return TileIRLoweringResult(
        kernel_name=kernel_name,
        cubin=b"",
        tileir_source=tileir_source,
        temporary_buffers=temporary_buffers,
        kernels=lowered_kernels,
    )
