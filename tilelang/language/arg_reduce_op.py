"""Tile-level index reductions built from the ordinary reduction primitives."""

from __future__ import annotations

from tvm import arith, tirx
from tvm.tirx.script.builder.ir import buffer_store, evaluate

from tilelang.language.allocate import alloc_fragment
from tilelang.language.copy_op import copy
from tilelang.language.loop import Parallel
from tilelang.language.reduce_op import reduce
from tilelang.utils.language import is_fragment, is_shared


def _reduce_arg(buffer: tirx.Buffer, out: tirx.Buffer, dim: int, kind: str) -> None:
    op_name = f"reduce_arg{kind}"
    if not isinstance(buffer, tirx.Buffer) or not isinstance(out, tirx.Buffer):
        raise TypeError(f"{op_name} expects input and output Buffers")
    if not isinstance(dim, int):
        raise TypeError(f"{op_name} dim must be an integer, got {dim!r}")
    rank = len(buffer.shape)
    if not -rank <= dim < rank:
        raise ValueError(f"{op_name} dim {dim} is out of bounds for input rank {rank}")
    dim %= rank
    if out.dtype not in ("int32", "int64"):
        raise ValueError(f"{op_name} output dtype must be int32 or int64, got {out.dtype}")
    supported_dtypes = (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    )
    if buffer.dtype not in supported_dtypes:
        raise ValueError(f"{op_name} does not support input dtype {buffer.dtype}")
    if not (is_fragment(buffer) or is_shared(buffer)) or not (is_fragment(out) or is_shared(out)):
        raise ValueError(f"{op_name} expects fragment or shared buffers, got {buffer.scope()} and {out.scope()}")

    analyzer = arith.Analyzer()
    extent = analyzer.simplify(buffer.shape[dim])
    if not isinstance(extent, tirx.IntImm) or extent.value <= 0:
        raise ValueError(f"{op_name} currently requires a positive constant reduction extent, got {extent}")
    index_bits = 32 if out.dtype == "int32" else 64
    if extent.value > (1 << (index_bits - 1)) - 1:
        raise ValueError(f"{op_name} reduction extent {extent} cannot be represented by output dtype {out.dtype}")

    squeezed_shape = list(buffer.shape[:dim]) + list(buffer.shape[dim + 1 :])
    kept_shape = list(buffer.shape)
    kept_shape[dim] = tirx.IntImm("int32", 1)

    def matches(shape):
        return len(out.shape) == len(shape) and all(analyzer.can_prove_equal(a, b) for a, b in zip(out.shape, shape))

    keep_dim = matches(kept_shape)
    if not keep_dim and not matches(squeezed_shape):
        raise ValueError(f"{op_name} output shape must be {squeezed_shape} or {kept_shape}, got {out.shape}")

    # The ordinary butterfly reduction requires power-of-two lane groups.
    # Pad only the reduction axis and mask padded indices out of the arg result.
    is_float = buffer.dtype.startswith("float") or buffer.dtype == "bfloat16"
    padded_extent = 1 << (extent.value - 1).bit_length()
    needs_padding = padded_extent != extent.value
    src = buffer
    if needs_padding:
        padded_shape = list(buffer.shape)
        padded_shape[dim] = padded_extent
        src = alloc_fragment(padded_shape, buffer.dtype)
        if is_float:
            identity = tirx.const(float("-inf") if kind == "max" else float("inf"), buffer.dtype)
        else:
            identity = tirx.min_value(buffer.dtype) if kind == "max" else tirx.max_value(buffer.dtype)
        with Parallel(*padded_shape) as loop_vars:
            coords = [loop_vars] if rank == 1 else list(loop_vars)
            value = tirx.if_then_else(coords[dim] < extent, tirx.BufferLoad(buffer, coords), identity)
            buffer_store(src, value, coords)
    elif is_shared(buffer):
        src = alloc_fragment(buffer.shape, buffer.dtype)
        evaluate(copy(buffer, src))
    reduced_shape = squeezed_shape or [1]
    values = alloc_fragment(reduced_shape, buffer.dtype)
    candidates = alloc_fragment(src.shape, out.dtype)
    indices = alloc_fragment(reduced_shape, out.dtype)
    reduce(src, values, kind, dim, True)

    extent_index = tirx.IntImm(out.dtype, extent.value)
    with Parallel(*src.shape) as loop_vars:
        coords = [loop_vars] if rank == 1 else list(loop_vars)
        reduced_coords = coords[:dim] + coords[dim + 1 :] or [0]
        index = tirx.Cast(out.dtype, coords[dim])
        value = tirx.BufferLoad(src, coords)
        candidate = tirx.Select(value == tirx.BufferLoad(values, reduced_coords), index, extent_index)
        if is_float:
            # NaN indices occupy [-extent, -1], so they precede every numeric
            # candidate and still order ties by the original logical index.
            # This does not depend on the value reducer's NaN propagation mode.
            nan_value = tirx.Cast("float32", value) if buffer.dtype == "bfloat16" else value
            candidate = tirx.Select(tirx.isnan(nan_value), index - extent_index, candidate)
        if needs_padding:
            candidate = tirx.Select(index < extent_index, candidate, extent_index)
        buffer_store(candidates, candidate, coords)
    reduce(candidates, indices, "min", dim, True)

    with Parallel(*(list(out.shape) or [1])) as loop_vars:
        coords = ([loop_vars] if len(out.shape) <= 1 else list(loop_vars)) if out.shape else []
        reduced_coords = (coords[:dim] + coords[dim + 1 :] if keep_dim else coords) or [0]
        index = tirx.BufferLoad(indices, reduced_coords)
        result = tirx.Select(index < 0, index + extent_index, index) if is_float else index
        buffer_store(out, result, coords)


def reduce_argmax(buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1) -> None:
    """Write the index of the maximum value along ``dim`` to ``out``.

    ``buffer`` and ``out`` must be fragment or shared buffers. The output must
    have dtype ``int32`` or ``int64`` and the input shape with ``dim`` removed
    or replaced by one. Negative dimensions are accepted. The reduction extent
    must be a positive compile-time constant representable by the output dtype.
    Non-power-of-two reduction extents are padded internally; padding indices
    never participate in the result.

    Indices are zero-based within the reduction dimension. Ties select the
    first index, including ties between positive and negative zero. If any
    input is NaN, the first NaN index is returned. Input values are not modified.

    Expands to a value reduction followed by an index reduction in the same
    kernel, with register temporaries and the ordinary reduction backend's
    synchronization. It does not launch another kernel or allocate global memory.
    All participating threads must execute the operation collectively.
    """
    _reduce_arg(buffer, out, dim, "max")


def reduce_argmin(buffer: tirx.Buffer, out: tirx.Buffer, dim: int = -1) -> None:
    """Write the index of the minimum value along ``dim`` to ``out``.

    Uses the same shape, dtype, scope, collective-execution and first-index tie
    rules as :func:`reduce_argmax`. If any input is NaN, the first NaN index is
    returned. Expands to two reductions in the same kernel.
    """
    _reduce_arg(buffer, out, dim, "min")
