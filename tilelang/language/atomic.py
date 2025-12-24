"""Atomic operations exposed on the TileLang language surface."""

from __future__ import annotations

import tilelang.language as T
from tvm import ir
from tvm.tir import PrimExpr, Buffer, BufferRegion, Var, op
from tilelang.utils.language import to_buffer_region, legalize_pairwise_extents

_MEMORY_ORDER_ID_MAP = {
    "relaxed": 0,
    "consume": 1,
    "acquire": 2,
    "release": 3,
    "acq_rel": 4,
    "seq_cst": 5,
}


def atomic_max(dst: Buffer, value: PrimExpr, memory_order: str | None = None, return_prev: bool = False) -> PrimExpr:
    """
    Perform an atomic maximum on the value stored at dst with an optional memory-order.

    If memory_order is None the runtime extern "AtomicMax" is called without an explicit memory-order id; otherwise the provided memory_order string is mapped to a numeric id using the module's memory-order map and passed to the extern.

    Parameters:
        dst (Buffer): Destination buffer/address to apply the atomic max.
        value (PrimExpr): Value to compare/store atomically.
        memory_order (Optional[str]): Optional memory-order name (e.g. "relaxed", "acquire", "seq_cst").
            If provided, it is translated to the corresponding numeric memory-order id before the call.
        return_prev (bool): If True, return the previous value; if False, return handle (default False).

    Returns:
        PrimExpr: A handle/expression representing the issued atomic maximum operation, or the previous value if return_prev is True.

    Examples:
        >>> # Basic atomic max operation
        >>> counter = T.Tensor([1], "float32", name="counter")
        >>> atomic_max(counter, 42.0)

        >>> # With memory ordering
        >>> atomic_max(counter, 100.0, memory_order="acquire")

        >>> # Get the previous value
        >>> prev_value = atomic_max(counter, 50.0, return_prev=True)
        >>> # prev_value now contains the value that was in counter before the max operation

        >>> # Use in parallel reduction to find global maximum
        >>> @T.prim_func
        >>> def find_max(data: T.Buffer, result: T.Buffer):
        >>>     for i in T.thread_binding(128, "threadIdx.x"):
        >>>         atomic_max(result, data[i])
    """
    func_name = "AtomicMaxRet" if return_prev else "AtomicMax"
    return_type = dst.dtype if return_prev else "handle"

    if memory_order is None:
        return T.call_extern(return_type, func_name, T.address_of(dst), value)
    else:
        return T.call_extern(
            return_type,
            func_name,
            T.address_of(dst),
            value,
            _MEMORY_ORDER_ID_MAP[memory_order],
        )


def atomic_min(dst: Buffer, value: PrimExpr, memory_order: str | None = None, return_prev: bool = False) -> PrimExpr:
    """
    Atomically update the value at dst to the minimum of its current value and value.

    If memory_order is provided, it selects the memory-order semantic used by the underlying extern call;
    allowed names are "relaxed", "consume", "acquire", "release", "acq_rel", and "seq_cst" (mapped internally
    to integer IDs). If memory_order is None, the extern is invoked without an explicit memory-order argument.

    Parameters:
        dst (Buffer): Destination buffer/address to apply the atomic min.
        value (PrimExpr): Value to compare/store atomically.
        memory_order (Optional[str]): Optional memory-order name controlling the atomic operation's ordering.
        return_prev (bool): If True, return the previous value; if False, return handle (default False).

    Returns:
        PrimExpr: A handle expression representing the atomic-min operation, or the previous value if return_prev is True.

    Examples:
        >>> # Basic atomic min operation
        >>> min_val = T.Tensor([1], "int32", name="min_val")
        >>> atomic_min(min_val, 10)

        >>> # Find minimum across threads
        >>> @T.prim_func
        >>> def find_min(data: T.Buffer, result: T.Buffer):
        >>>     for i in T.thread_binding(256, "threadIdx.x"):
        >>>         atomic_min(result, data[i])

        >>> # Track minimum with previous value
        >>> threshold = T.Tensor([1], "float32", name="threshold")
        >>> old_min = atomic_min(threshold, 3.14, return_prev=True)
        >>> # old_min contains the previous minimum value

        >>> # With relaxed memory ordering for performance
        >>> atomic_min(min_val, 5, memory_order="relaxed")
    """
    func_name = "AtomicMinRet" if return_prev else "AtomicMin"
    return_type = dst.dtype if return_prev else "handle"

    if memory_order is None:
        return T.call_extern(return_type, func_name, T.address_of(dst), value)
    else:
        return T.call_extern(
            return_type,
            func_name,
            T.address_of(dst),
            value,
            _MEMORY_ORDER_ID_MAP[memory_order],
        )


def atomic_add(dst: Buffer, value: PrimExpr, memory_order: str | None = None, return_prev: bool = False, use_tma: bool = False) -> PrimExpr:
    """
    Atomically add `value` into `dst`, returning a handle to the operation.

    Supports scalar/addressed extern atomic add when neither argument exposes extents, or tile-region-based atomic add for Buffer/BufferRegion/BufferLoad inputs. If both arguments are plain Buffers their shapes must be structurally equal. If at least one side exposes extents, extents are aligned (missing dimensions are treated as size 1); an assertion is raised if extents cannot be deduced. The optional `memory_order` (one of "relaxed","consume","acquire","release","acq_rel","seq_cst") is used only for the direct extern `AtomicAdd` path when no extents are available — otherwise the tile-region path ignores `memory_order`.

    Parameters:
        dst (Buffer): Destination buffer/address to apply the atomic add.
        value (PrimExpr): Value to add atomically.
        memory_order (Optional[str]): Optional memory-order name controlling the atomic operation's ordering.
        return_prev (bool): If True, return the previous value; if False, return handle (default False).
        use_tma (bool): If True, use TMA (cp.reduce) to perform the atomic add. This is available only for sm90+ (default False).

    Returns:
        PrimExpr: A handle representing the atomic addition operation, or the previous value if return_prev is True.

    Examples:
        >>> # Basic atomic addition
        >>> counter = T.Tensor([1], "int32", name="counter")
        >>> atomic_add(counter, 1)  # Increment counter by 1

        >>> # Parallel sum reduction
        >>> @T.prim_func
        >>> def parallel_sum(data: T.Buffer, result: T.Buffer):
        >>>     for i in T.thread_binding(1024, "threadIdx.x"):
        >>>         atomic_add(result, data[i])

        >>> # Get previous value for debugging
        >>> old_value = atomic_add(counter, 5, return_prev=True)
        >>> # old_value contains the value before adding 5

        >>> # Tensor-to-tensor atomic add (tile-region based)
        >>> src_tensor = T.Tensor([128, 64], "float32", name="src")
        >>> dst_tensor = T.Tensor([128, 64], "float32", name="dst")
        >>> atomic_add(dst_tensor, src_tensor)  # Add entire tensors atomically

        >>> # With memory ordering for scalar operations
        >>> atomic_add(counter, 10, memory_order="acquire")

        >>> # Accumulate gradients in training
        >>> gradients = T.Tensor([1000], "float32", name="gradients")
        >>> global_grad = T.Tensor([1000], "float32", name="global_grad")
        >>> atomic_add(global_grad, gradients)
    """

    def get_extent(data):
        """
        Return the inferred extent (shape) of a buffer-like object.

        If `data` is a Var bound to a let value, the let value is resolved before inspection.
        Parameters:
            data: A Var, Buffer, or BufferRegion to inspect.

        Returns:
            The shape/extents as a list-like of PrimExpr (Buffer.shape or list of region item extents), or None if the extent cannot be determined.
        """
        if isinstance(data, Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, Buffer):
            return data.shape
        elif isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(value)
    dst_extent = get_extent(dst)

    if dst_extent is None and src_extent is None:
        func_name = "AtomicAddRet" if return_prev else "AtomicAdd"
        return_type = dst.dtype if return_prev else "handle"

        # Pass destination by pointer to match device signature
        if memory_order is None:
            return T.call_extern(return_type, func_name, T.address_of(dst), value)
        else:
            return T.call_extern(
                return_type,
                func_name,
                T.address_of(dst),
                value,
                _MEMORY_ORDER_ID_MAP[memory_order],
            )

    if isinstance(dst, Buffer) and isinstance(value, Buffer):
        ir.assert_structural_equal(dst.shape, value.shape)

    assert src_extent or dst_extent, "Can't deduce atomicadd extents from args"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    value = to_buffer_region(value, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    # Note: tile-region-based atomic operations don't support return_prev yet
    # This would need to be implemented in the tile runtime
    if return_prev:
        raise NotImplementedError("return_prev is not supported for tile-region-based atomic operations")

    if memory_order is None:
        return T.call_intrin("handle", op.Op.get("tl.tileop.atomicadd"), value, dst, use_tma, 0)
    else:
        return T.call_intrin("handle", op.Op.get("tl.tileop.atomicadd"), value, dst, use_tma, _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_addx2(dst: Buffer, value: PrimExpr, return_prev: bool = False) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)
        return_prev (bool): If True, return the previous value; if False, return handle (default False)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation, or the previous value if return_prev is True

    Examples:
        >>> # Atomic addition with FP16 pairs
        >>> half_dst = T.Tensor([2], "float16", name="half_dst")
        >>> half_val = T.Tensor([2], "float16", name="half_val")
        >>> atomic_addx2(half_dst, half_val)

        >>> # BF16 vectorized atomic add (requires CUDA Arch > 750)
        >>> bf16_dst = T.Tensor([2], "bfloat16", name="bf16_dst")
        >>> bf16_val = T.Tensor([2], "bfloat16", name="bf16_val")
        >>> atomic_addx2(bf16_dst, bf16_val)

        >>> # Get previous paired values
        >>> prev_values = atomic_addx2(half_dst, half_val, return_prev=True)
        >>> # prev_values is a half2 containing the two previous FP16 values

        >>> # Efficient gradient accumulation for mixed precision training
        >>> @T.prim_func
        >>> def accumulate_fp16_gradients(grads: T.Buffer, global_grads: T.Buffer):
        >>>     for i in T.thread_binding(128, "threadIdx.x"):
        >>>         for j in range(0, grads.shape[1], 2):  # Process in pairs
        >>>             atomic_addx2(global_grads[i, j:j+2], grads[i, j:j+2])
    """
    func_name = "AtomicAddx2Ret" if return_prev else "AtomicAddx2"
    return_type = dst.dtype if return_prev else "handle"
    return T.call_extern(return_type, func_name, T.address_of(dst), T.address_of(value))


def atomic_addx4(dst: Buffer, value: PrimExpr, return_prev: bool = False) -> PrimExpr:
    """Perform an atomic addition operation with quad-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (quad-width)
        return_prev (bool): If True, return the previous value; if False, return handle (default False)

    Returns:
        PrimExpr: Handle to the quad-width atomic addition operation, or the previous value if return_prev is True

    Examples:
        >>> # Atomic addition with float4 (requires CUDA Arch >= 900)
        >>> float4_dst = T.Tensor([4], "float32", name="float4_dst")
        >>> float4_val = T.Tensor([4], "float32", name="float4_val")
        >>> atomic_addx4(float4_dst, float4_val)

        >>> # Get previous float4 values
        >>> prev_float4 = atomic_addx4(float4_dst, float4_val, return_prev=True)
        >>> # prev_float4 is a float4 containing the four previous float32 values

        >>> # High-throughput gradient accumulation for large models
        >>> @T.prim_func
        >>> def accumulate_float4_gradients(grads: T.Buffer, global_grads: T.Buffer):
        >>>     for i in T.thread_binding(256, "threadIdx.x"):
        >>>         for j in range(0, grads.shape[1], 4):  # Process 4 floats at once
        >>>             atomic_addx4(global_grads[i, j:j+4], grads[i, j:j+4])

        >>> # Efficient RGBA pixel blending
        >>> rgba_dst = T.Tensor([4], "float32", name="rgba_dst")  # R, G, B, A channels
        >>> rgba_add = T.Tensor([4], "float32", name="rgba_add")
        >>> atomic_addx4(rgba_dst, rgba_add)  # Atomic blend of all 4 channels
    """
    func_name = "AtomicAddx4Ret" if return_prev else "AtomicAddx4"
    return_type = "float4" if "float" in str(dst.dtype).lower() else "handle"
    return T.call_extern(return_type, func_name, T.address_of(dst), T.address_of(value))


def atomic_load(src: Buffer, memory_order: str = "seq_cst") -> PrimExpr:
    """
    Load a value from the given buffer using the specified atomic memory ordering.

    Performs an atomic load from `src` and returns a PrimExpr representing the loaded value.
    memory_order selects the ordering and must be one of: "relaxed", "consume", "acquire",
    "release", "acq_rel", or "seq_cst" (default).
    Raises KeyError if an unknown memory_order is provided.

    Note: atomic_load always returns the loaded value, so no return_prev parameter is needed.

    Examples:
        >>> # Basic atomic load
        >>> shared_var = T.Tensor([1], "int32", name="shared_var")
        >>> value = atomic_load(shared_var)

        >>> # Load with specific memory ordering
        >>> value = atomic_load(shared_var, memory_order="acquire")
        >>> # Ensures all subsequent memory operations happen after this load

        >>> # Relaxed load for performance-critical code
        >>> value = atomic_load(shared_var, memory_order="relaxed")

        >>> # Producer-consumer pattern
        >>> @T.prim_func
        >>> def consumer(flag: T.Buffer, data: T.Buffer, result: T.Buffer):
        >>>     # Wait until producer sets flag
        >>>     while atomic_load(flag, memory_order="acquire") == 0:
        >>>         pass  # Spin wait
        >>>     # Now safely read data
        >>>     result[0] = data[0]

        >>> # Load counter for statistics
        >>> counter = T.Tensor([1], "int64", name="counter")
        >>> current_count = atomic_load(counter, memory_order="relaxed")
    """
    return T.call_extern(src.dtype, "AtomicLoad", T.address_of(src), _MEMORY_ORDER_ID_MAP[memory_order])


def atomic_store(dst: Buffer, src: PrimExpr, memory_order: str = "seq_cst") -> PrimExpr:
    """
    Perform an atomic store of `src` into `dst` with the given memory ordering.

    Parameters:
        dst (Buffer): Destination buffer to store into.
        src (PrimExpr): Value to store.
        memory_order (str, optional): Memory ordering name; one of "relaxed", "consume",
            "acquire", "release", "acq_rel", or "seq_cst". Defaults to "seq_cst".
            The name is mapped to an internal numeric ID used by the underlying runtime.

    Returns:
        PrimExpr: A handle representing the issued atomic store operation.

    Raises:
        KeyError: If `memory_order` is not one of the supported names.

    Note: atomic_store doesn't return a previous value, so no return_prev parameter is needed.

    Examples:
        >>> # Basic atomic store
        >>> shared_var = T.Tensor([1], "int32", name="shared_var")
        >>> atomic_store(shared_var, 42)

        >>> # Store with release ordering to publish data
        >>> data = T.Tensor([1000], "float32", name="data")
        >>> ready_flag = T.Tensor([1], "int32", name="ready_flag")
        >>> # ... fill data ...
        >>> atomic_store(ready_flag, 1, memory_order="release")
        >>> # Ensures all previous writes are visible before flag is set

        >>> # Relaxed store for performance
        >>> atomic_store(shared_var, 100, memory_order="relaxed")

        >>> # Producer-consumer synchronization
        >>> @T.prim_func
        >>> def producer(data: T.Buffer, flag: T.Buffer):
        >>>     data[0] = 3.14159  # Write data first
        >>>     atomic_store(flag, 1, memory_order="release")
        >>>     # Consumer can now safely read data after seeing flag == 1

        >>> # Update configuration atomically
        >>> config = T.Tensor([1], "int32", name="config")
        >>> new_config = 0x12345678
        >>> atomic_store(config, new_config, memory_order="seq_cst")

        >>> # Thread-safe logging counter
        >>> log_counter = T.Tensor([1], "int64", name="log_counter")
        >>> atomic_store(log_counter, 0)  # Reset counter atomically
    """
    return T.call_extern("handle", "AtomicStore", T.address_of(dst), src, _MEMORY_ORDER_ID_MAP[memory_order])
