# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared helper functions for reduction schedule rules."""

from __future__ import annotations


from . import utils

from tvm import ir, tirx as tir
from tvm.target import Target

_MIN_ELEMS_PER_THREAD = 16


def _as_const_int(expr: tir.PrimExpr) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _as_const_float(expr: tir.PrimExpr) -> float | None:
    if isinstance(expr, tir.FloatImm):
        return float(expr.value)
    if isinstance(expr, tir.IntImm):
        return float(expr.value)
    return None


def _is_accumulator_term(store: tir.BufferStore, expr: tir.PrimExpr) -> bool:
    return ir.structural_equal(
        expr,
        tir.BufferLoad(store.buffer, store.indices),
        map_free_vars=True,
    )


def _analyze_reduction_update(block: tir.SBlock) -> tuple[str, tir.PrimExpr] | None:
    """Infer reduction type and source expression from block update."""
    buffer_store = block.body
    if not isinstance(buffer_store, tir.BufferStore):
        return None

    value = buffer_store.value
    if isinstance(value, tir.Add):
        if _is_accumulator_term(buffer_store, value.a):
            return "sum", value.b
        if _is_accumulator_term(buffer_store, value.b):
            return "sum", value.a
        return None
    if isinstance(value, tir.Max):
        if _is_accumulator_term(buffer_store, value.a):
            return "max", value.b
        if _is_accumulator_term(buffer_store, value.b):
            return "max", value.a
        return None
    if isinstance(value, tir.Min):
        if _is_accumulator_term(buffer_store, value.a):
            return "min", value.b
        if _is_accumulator_term(buffer_store, value.b):
            return "min", value.a
        return None

    value_type = type(value).__name__
    if value_type in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
        lhs = value.a
        rhs = value.b
        if _is_accumulator_term(buffer_store, lhs):
            src_expr = rhs
        elif _is_accumulator_term(buffer_store, rhs):
            src_expr = lhs
        else:
            return None
        reduce_type = {
            "BitwiseAnd": "bitand",
            "BitwiseOr": "bitor",
            "BitwiseXor": "bitxor",
        }[value_type]
        return reduce_type, src_expr

    return None


def _extract_single_input_buffer(rhs: tir.PrimExpr, write_buffer: tir.Buffer) -> tir.Buffer | None:
    buffers: list[tir.Buffer] = []

    def _collect(expr):
        if (
            isinstance(expr, tir.BufferLoad)
            and (not expr.buffer.same_as(write_buffer))
            and not any(expr.buffer.same_as(buf) for buf in buffers)
        ):
            buffers.append(expr.buffer)

    tir.stmt_functor.post_order_visit(rhs, _collect)
    if len(buffers) != 1:
        return None
    return buffers[0]


def _collect_input_buffers(rhs: tir.PrimExpr, write_buffer: tir.Buffer) -> list[tir.Buffer]:
    buffers: list[tir.Buffer] = []

    def _collect(expr):
        if (
            isinstance(expr, tir.BufferLoad)
            and (not expr.buffer.same_as(write_buffer))
            and not any(expr.buffer.same_as(buf) for buf in buffers)
        ):
            buffers.append(expr.buffer)

    tir.stmt_functor.post_order_visit(rhs, _collect)
    return buffers


def _unwrap_to_buffer_load(expr: tir.PrimExpr) -> tir.BufferLoad | None:
    """Strip element-wise wrappers (Cast, etc.) to find the underlying BufferLoad."""
    while True:
        if isinstance(expr, tir.BufferLoad):
            return expr
        if isinstance(expr, tir.Cast):
            expr = expr.value
            continue
        return None


def _classify_reduce_expr(
    rhs: tir.PrimExpr,
    target_buffer: tir.Buffer,
    reduce_type: str,
) -> str | None:
    """Determine the T.reduce type that is semantically equivalent to
    ``reduce_type(rhs)`` where *rhs* may contain inlined element-wise
    transforms (Cast, pow, abs, neg, …).

    Returns one of ``"sum"``, ``"sumsq"``, ``"abssum"``, ``"max"``,
    ``"absmax"``, ``"min"`` if a match is found, else ``None``.

    This generalises the old ``_is_direct_buffer_load`` /
    ``_is_square_of_buffer_load`` checks so that FuseTIR-inlined
    expressions like ``pow(cast(buf[i]), 2)`` are recognised as
    ``"sumsq"`` without special-casing each form.
    """
    # Identity: rhs is (possibly cast) buf[i]
    inner = _unwrap_to_buffer_load(rhs)
    if inner is not None and inner.buffer.same_as(target_buffer):
        return reduce_type  # sum→sum, max→max, etc.

    # Square patterns (map to "sumsq" when reduce_type is "sum"):
    #   a) buf[i] * buf[i]   (with optional casts)
    #   b) pow(buf[i], 2)    (with optional casts)
    if reduce_type == "sum":
        if isinstance(rhs, tir.Mul):
            la = _unwrap_to_buffer_load(rhs.a)
            lb = _unwrap_to_buffer_load(rhs.b)
            if (
                la is not None
                and lb is not None
                and la.buffer.same_as(target_buffer)
                and lb.buffer.same_as(target_buffer)
                and all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(la.indices, lb.indices))
            ):
                return "sumsq"
        if isinstance(rhs, tir.Call) and getattr(rhs.op, "name", "") == "tir.pow" and len(rhs.args) == 2:
            base = _unwrap_to_buffer_load(rhs.args[0])
            exp_val = _as_const_float(rhs.args[1])
            if base is not None and exp_val == 2.0 and base.buffer.same_as(target_buffer):
                return "sumsq"

    # Absolute value patterns (map to "abssum"/"absmax"):
    #   abs(buf[i]) or max(buf[i], -buf[i])
    # Not needed for current use case — left for future extension.

    return None


def _is_direct_buffer_load(expr: tir.PrimExpr, target_buffer: tir.Buffer) -> bool:
    """Check whether expr is exactly a direct load from target_buffer."""
    return isinstance(expr, tir.BufferLoad) and expr.buffer.same_as(target_buffer)


def _is_same_buffer_load(
    lhs: tir.PrimExpr,
    rhs: tir.PrimExpr,
    target_buffer: tir.Buffer | None = None,
) -> bool:
    if not isinstance(lhs, tir.BufferLoad) or not isinstance(rhs, tir.BufferLoad):
        return False
    if not lhs.buffer.same_as(rhs.buffer):
        return False
    if target_buffer is not None and not lhs.buffer.same_as(target_buffer):
        return False
    if len(lhs.indices) != len(rhs.indices):
        return False
    return all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(lhs.indices, rhs.indices))


def _is_square_of_buffer_load(expr: tir.PrimExpr, target_buffer: tir.Buffer) -> bool:
    if not isinstance(expr, tir.Mul):
        return False
    # Allow element-wise wrappers (Cast, etc.) around the BufferLoads so that
    # patterns like ``cast(x[i,j], fp32) * cast(x[i,j], fp32)`` — produced when
    # the layernorm pattern inlines the x_fp32 cast into sum_x_sq — still
    # count as a square of ``x``.  Without this, LayerNormLike would reject
    # the second-moment variant of LayerNorm that avoids the diff fragment.
    la = _unwrap_to_buffer_load(expr.a)
    lb = _unwrap_to_buffer_load(expr.b)
    if la is None or lb is None:
        return False
    if not la.buffer.same_as(lb.buffer):
        return False
    if not la.buffer.same_as(target_buffer):
        return False
    if len(la.indices) != len(lb.indices):
        return False
    return all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(la.indices, lb.indices))


def _block_writes_buffer(block: tir.SBlock, target_buffer: tir.Buffer) -> bool:
    return any(write_region.buffer.same_as(target_buffer) for write_region in block.writes)


def _find_buffer_index(regions, target_buffer: tir.Buffer) -> int | None:
    for idx, region in enumerate(regions):
        if region.buffer.same_as(target_buffer):
            return idx
    return None


def _infer_reduce_dim(read_buffer: tir.Buffer, write_buffer: tir.Buffer) -> int:
    src_shape = list(read_buffer.shape)
    dst_shape = list(write_buffer.shape)

    def _shape_equal(lhs, rhs):
        if len(lhs) != len(rhs):
            return False
        return all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(lhs, rhs))

    # Case 1: dst rank is src rank - 1
    if len(dst_shape) == len(src_shape) - 1:
        for dim in range(len(src_shape)):
            if _shape_equal(src_shape[:dim] + src_shape[dim + 1 :], dst_shape):
                return dim

    # Case 2: dst rank equals src rank and reduced axis is kept with extent 1
    if len(dst_shape) == len(src_shape):
        for dim in range(len(src_shape)):
            if not _shape_equal(src_shape[:dim], dst_shape[:dim]):
                continue
            if not _shape_equal(src_shape[dim + 1 :], dst_shape[dim + 1 :]):
                continue
            if _as_const_int(dst_shape[dim]) == 1:
                return dim

    return 0


def _default_init_value(reduce_type: str) -> float:
    if reduce_type in ("sum", "abssum", "bitor", "bitxor"):
        return 0.0
    if reduce_type in ("max", "absmax"):
        return float("-inf")
    if reduce_type == "min":
        return float("inf")
    if reduce_type == "bitand":
        return -1.0
    return 0.0


def _infer_init_value(block: tir.SBlock, reduce_type: str) -> float:
    if block.init is not None and isinstance(block.init, tir.BufferStore):
        value = _as_const_float(block.init.value)
        if value is not None:
            return value
    return _default_init_value(reduce_type)


def _choose_num_threads(target: Target, reduction_extent: tir.PrimExpr) -> int:
    """Pick cooperative thread count for one-block-per-output reduction.

    Ensures each thread handles at least ``_MIN_ELEMS_PER_THREAD`` elements
    so that local accumulation amortises the cost of the cross-thread tree
    reduction (warp shuffles + shared-memory sync).

    For very large reduction extents we let the thread count grow up to
    512 threads/CTA — empirically this halves softmax(N=32k) and similar
    wide-row reductions on H100 vs the previous 256-thread cap.  At
    N≤4k the heuristic still picks 256 because of the elements-per-thread
    minimum, so smaller reductions are unaffected.
    """

    max_threads = int(utils.max_threads_per_block(target))
    max_threads = min(max_threads, 512)
    if max_threads <= 0:
        return 1

    extent = _as_const_int(reduction_extent)
    if extent is None or extent <= 0:
        return max_threads

    # Cap so that each thread has at least _MIN_ELEMS_PER_THREAD elements.
    # 16 elements/thread keeps mid-range (N≈4k) reductions at 256 threads:
    # fewer cross-warp bar.sync rounds in AllReduce, while wide reductions
    # (N≥16k) still reach the 512-thread ceiling.
    effective_max = min(max_threads, max(1, extent // _MIN_ELEMS_PER_THREAD))
    threads = 1
    while (threads << 1) <= effective_max and (threads << 1) <= extent:
        threads <<= 1
    return max(threads, 1)


def _choose_reduction_step(target: Target, reduction_extent: tir.PrimExpr) -> int | None:
    """Choose reduction chunk size for very large K.

    Returns None when extent is dynamic, so the template falls back to
    non-chunked reduction.
    """

    extent = _as_const_int(reduction_extent)
    if extent is None:
        return None
    if extent <= 0:
        return 1

    max_threads = int(utils.max_threads_per_block(target))
    max_threads = min(max_threads, 256)
    # TileLang favors wider tile chunks than thread-level templates.
    # Keep chunk size capped to avoid oversized local fragments while
    # allowing softmax-like K=16384 to stay within a single chunk.
    step_limit = max(1, min(max_threads * 64, 16384))
    step = min(extent, step_limit)

    # Keep static divisibility to avoid tail-copy OOB in cache_read_at.
    while step > 1 and extent % step != 0:
        step >>= 1
    return max(step, 1)
