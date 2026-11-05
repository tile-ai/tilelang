"""Annotation helpers exposed on the TileLang language surface."""

from collections.abc import Callable

from tilelang.layout import Fragment, Layout, PartialFragment
from tilelang.utils.language import is_fragment, is_reducer
from tilelang.language._native import FloatImm, IntImm, tvm_tuple
from tilelang.language._native import attr
from tilelang.language._native import sblock_attr
from tilelang.language._native import _source_sblock_attr, tensor_storage_source
from tvm_tirx.tirx import is_tensor_var

__all__ = [
    "WSID",
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
    "annotate_min_blocks_per_sm",
    "annotate_ws_pipeline_depth",
    "annotate_ws_schedule",
    "ws_op",
]

# Annotation key carrying a statement's stable warp-specialization op id
# (kWSOpIdKey in src/transform/common/warp_specialize.h). Shorthand so kernels
# write ``annotations={T.WSID: "copy_A"}`` on tile ops and loops.
WSID = "tl.ws_op_id"


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    if order == "row":
        device_func = "rasterization2DRow"
    elif order == "column":
        device_func = "rasterization2DColumn"
    elif order == "mlx":
        device_func = "rasterization2DMLX"
    else:
        raise ValueError(f"Unsupported swizzle order: {order}")
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", tvm_tuple(device_func, panel_size))


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    values = []
    for buffer, layout in layout_map.items():
        if is_reducer(buffer):
            # The reducer's replicas are addends awaiting the finalize
            # collective, not equal copies: plain Fragment semantics would
            # silently misdescribe them.
            assert isinstance(layout, PartialFragment), (
                f"for reducer {buffer} (T.alloc_reducer), the layout must be a "
                f"PartialFragment describing the per-thread partials, but got {type(layout)}"
            )
        elif is_fragment(buffer):
            assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
        if isinstance(layout, Layout):
            values.append((buffer, layout))
        elif isinstance(layout, Callable):
            values.append((buffer, Layout(buffer.shape, layout)))
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return _annotate_tensor_values("layout_map", values)


def _annotate_tensor_values(key, entries):
    """Preserve same-call last assignment for an identical source data key."""
    tensors, sources, selected = {}, {}, {}
    for tensor, value in entries:
        source = tensor_storage_source(tensor)
        previous = selected.get(source)
        if previous is not None:
            del tensors[previous]
        selected[source] = tensor
        tensors[tensor] = value
        sources[source] = value
    return _source_sblock_attr({key: tensors}, {key: sources})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    return _annotate_tensor_values("safe_value_map", safe_value_map.items())


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    values = []
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert str(buffer.scope()) == "global", "persistent L2 can only be applied to global buffers"
        values.append((buffer, FloatImm("float32", float(hit_ratio))))
    return _annotate_tensor_values("l2_hit_ratio_map", values)


def annotate_min_blocks_per_sm(n: int):
    """Annotate the minimum number of thread blocks per SM (multiprocessor).

    When set, this value is passed as the second argument of
    ``__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`` in
    the generated CUDA kernel.  A larger value hints the compiler to limit
    register usage so that more blocks can reside on each SM simultaneously,
    which can improve occupancy at the cost of potentially more register
    spilling.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(...):
    ...     with T.Kernel(...):
                T.annotate_min_blocks_per_sm(2)
    ...         ...
    """
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    return attr(None, "tl.min_blocks_per_sm", n)


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    tensors = []
    for buf in buffers:
        if not is_tensor_var(buf):
            raise TypeError(f"annotate_restrict_buffers expects tensor arguments, got {type(buf)}")
        tensors.append(buf)
    # The consumer resolves explicit view aliases before lowering the ABI.
    # A freshly constructed tensor_data_ptr Call is not a stable identity key.
    return sblock_attr({"tl.non_restrict_params": tensors})


def annotate_ws_pipeline_depth(depth_map: dict):
    """Declare the depth of the pipelines the automatic scheduler hosts at
    the ENCLOSING scope for the given buffers, asserting each recurrence
    resets per version — otherwise the depth derives from the loop's
    ``num_stages``, and accumulators pin single-buffered.

    ``depth_map`` maps each buffer to its pipeline depth at this scope.
    """
    _map = {}
    for buffer, depth in depth_map.items():
        assert isinstance(depth, int) and depth >= 1, f"pipeline depth must be a positive int, got {depth!r}"
        if not is_tensor_var(buffer):
            raise TypeError(f"annotate_ws_pipeline_depth expects tensor keys, got {type(buffer)}")
        _map[buffer] = IntImm("int32", depth)
    return attr(_map, "tl.ws_pipeline_depth", 0)


def annotate_ws_schedule(schedule):
    """Attach a warp-specialization schedule to the kernel.

    ``schedule`` is a typed :class:`~tilelang.language.warp_specialize.WSSchedule`
    object describing how to transform the straight-line kernel into a
    warp-specialized one: warp roles, pipelines (full/empty barrier pairs
    protecting multi-versioned buffers), and per-role instruction sequences
    per loop scope. It is materialized by the ``MaterializeWSSchedule`` pass;
    see ``examples/aws/gemm.py`` for a complete example.
    """
    from tilelang.language.warp_specialize import WSSchedule

    assert isinstance(schedule, WSSchedule), f"annotate_ws_schedule expects a T.WSSchedule object, got {type(schedule)}"
    return sblock_attr({"tl.ws_schedule": schedule})


def ws_op(op_id: str):
    """Manually annotate the enclosed statement(s) with a stable
    warp-specialization op or scope id.

    Wraps the body in an ``AttrStmt`` with key :data:`WSID`. This is the
    annotation of last resort, for statement forms that cannot carry
    ``annotations=`` themselves. The enclosed statements — a scalar Bind,
    or several statements such as an inlined scheduler method — become
    ONE opaque op:

    >>> with T.ws_op("rescale_vote"):
    ...     should_rescale = T.any_sync(scale_shared[tid] < 1.0)
    >>> with T.ws_op("sched_next"):
    ...     sched.next_tile()

    With a scope id the wrapper encloses a ``while`` loop and opens that
    scope (serial scope loops carry the id in their own annotations):

    >>> with T.ws_op("loop_wave"):
    ...     while sched.valid():
    ...         ...

    Tile ops and loops pass ``annotations={T.WSID: ...}`` directly
    instead.
    """
    assert isinstance(op_id, str) and op_id, "ws_op id must be a non-empty string"
    return attr(None, WSID, op_id)
