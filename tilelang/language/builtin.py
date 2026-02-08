"""Builtin operations exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BufferLikeTypeTuple, BarrierType, DType
from tilelang import tvm as tvm
from tilelang.language import ptx_arrive_barrier, evaluate
from tilelang.language.kernel import get_thread_bindings, get_block_extents
from tilelang.utils.target import check_hip_availability
from tvm import DataType, tir
from tvm.runtime import convert
from tvm.tir import PrimExpr, Var, Call, BufferLoad, BufferRegion
from tilelang.utils.language import retrieve_ptr

_IS_HIP_AVAILABLE = check_hip_availability()


def _normalize_index_arg(value: int | PrimExpr | None) -> PrimExpr | None:
    """
    Normalize warp sizing arguments so both Python ints and PrimExpr values
    are accepted uniformly.
    """
    if value is None:
        return None
    if isinstance(value, PrimExpr):
        return value
    if isinstance(value, int):
        return tir.IntImm("int32", value)
    raise TypeError(f"Expect warp sizing argument to be int or PrimExpr, but got {type(value)}.")


def _mbar_to_buffer_load(mbar: BarrierType) -> BufferLoad:
    """Convert a memory barrier to a buffer load.

    Args:
        mbar: BarrierType
            The memory barrier to convert

    Returns:
        tir.BufferLoad: A buffer load of the memory barrier
    """
    if isinstance(mbar, tir.BufferLoad):
        return mbar
    elif isinstance(mbar, tir.Buffer):
        assert len(mbar.shape) == 1, f"mbarrier must be a single element buffer, but got {mbar.shape}"
        return tir.BufferLoad(mbar, [0])
    else:
        raise TypeError(f"mbarrier must be an tir.BufferLoad or a tir.Buffer, but got {type(mbar)}")


def __ldg(load_or_buf: BufferLoad | tir.Buffer, index: PrimExpr | int | None = None) -> PrimExpr:
    """Explicitly load via CUDA read-only data cache.

    Prefer calling with a BufferLoad: `T.__ldg(x[i])` emits `__ldg(&x[i])` on CUDA.
    On non-CUDA backends, falls back to a regular load.

    Args:
        load_or_buf: A `BufferLoad` like `x[i]`, or a `Buffer`.
        index: Optional index when passing a `Buffer` directly.

    Returns:
        PrimExpr: The loaded value.
    """
    if isinstance(load_or_buf, BufferLoad):
        dtype = load_or_buf.dtype
        return tir.call_intrin(str(dtype), tir.op.Op.get("tl.__ldg"), load_or_buf)
    if isinstance(load_or_buf, tir.Buffer):
        if index is None:
            raise ValueError("T.__ldg(Buffer, index) requires an index when passing a Buffer.")
        idx = index
        if isinstance(index, (list, tuple)):
            if len(index) != 1:
                raise ValueError("T.__ldg currently supports 1D flattened indices.")
            idx = index[0]
        bl = BufferLoad(load_or_buf, [idx])
        return tir.call_intrin(str(load_or_buf.dtype), tir.op.Op.get("tl.__ldg"), bl)
    raise TypeError("T.__ldg expects a BufferLoad or a Buffer.")


def create_tma_descriptor(*args):
    """Create a Tensor Memory Access (TMA) descriptor.

    Args:
        *args: Variable arguments defining the TMA descriptor configuration

    Returns:
        tir.Call: A handle to the created TMA descriptor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.create_tma_descriptor"), *args)


# NOTE(wt): T.create_list_of_mbarrier and T.get_mbarrier is now only an intermediate intrinsic
# during transforms, and won't be exposed to frontend. For creating mbarriers, please use T.alloc_barrier instead.


def tma_load(*args):
    """Perform a Tensor Memory Access (TMA) load operation.

    Args:
        *args: Variable arguments specifying the TMA load parameters

    Returns:
        tir.Call: A handle to the TMA load operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.tma_load"), *args)


def fence_proxy_async(*args):
    """Create a fence for asynchronous proxy operations.

    Args:
        *args: Variable arguments for fence configuration

    Returns:
        tir.Call: A handle to the fence operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.fence_proxy_async"), *args)


def tma_store_arrive(*args):
    """Signal the arrival of a TMA store operation.

    Args:
        *args: Variable arguments for the store arrival operation

    Returns:
        tir.Call: A handle to the store arrive operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.tma_store_arrive"), *args)


def tma_store_wait(*args):
    """Wait for completion of TMA store operations.

    Args:
        *args: Variable arguments specifying which store operations to wait for

    Returns:
        tir.Call: A handle to the store wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.tma_store_wait"), *args)


def set_max_nreg(reg_count: int, is_inc: int):
    """Set the maximum number of registers to use.
    Detailed Documentation:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg

    Args:
        reg_count: int
            The number of registers to allocate
        is_inc: int
            Whether to increment or decrement the register count
            0 if decrement, 1 if increment

    Returns:
        tir.Call: A handle to the register setting operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.set_max_nreg"), reg_count, is_inc)


def inc_max_nreg(reg_count: int):
    """Increment the maximum number of registers to use."""
    return set_max_nreg(reg_count, 1)


def dec_max_nreg(reg_count: int):
    """Decrement the maximum number of registers to use."""
    return set_max_nreg(reg_count, 0)


def annotate_producer_reg_dealloc(reg_count: int = 24):
    """Annotate the producer reg dealloc."""
    return dec_max_nreg(reg_count)


def annotate_consumer_reg_alloc(reg_count: int = 240):
    """Annotate the consumer reg alloc."""
    return inc_max_nreg(reg_count)


def no_set_max_nreg():
    """Disable the maximum register limit setting."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.no_set_max_nreg"))


def disable_warp_group_reg_alloc():
    """Disable the warp group reg alloc."""
    return no_set_max_nreg()


def mbarrier_wait_parity(mbarrier: BarrierType, parity: int | Var):
    """Wait for memory barrier parity condition.

    Args:
            mbarrier: BarrierType
            The memory barrier to wait on
        parity: int | Var
            The parity value to wait for
    Examples:
        .. code-block:: python

            mbar = T.alloc_barrier(1)
            # Wait for parity 0 on a single mbarrier
            T.mbarrier_wait_parity(mbar, 0)

            mbars = T.alloc_barrier([128] * n)
            # Wait for parity value on one of the mbarriers
            T.mbarrier_wait_parity(mbars[ko], ko)

            # Common usage in pipelined kernels:
            for ko in range(num_stages):
                # Producer waits for consumer to finish previous iteration
                T.mbarrier_wait_parity(mbars[1], ko ^ 1)
                # Producer copies data
                T.copy(A_global, A_shared)
                # Producer signals data ready
                T.mbarrier_arrive(mbars[0])

                # Consumer waits for producer data
                T.mbarrier_wait_parity(mbars[0], ko)
                # Consumer computes
                T.gemm(A_shared, B_shared, C_local)
                # Consumer signals completion
                T.mbarrier_arrive(mbars[1])
    Returns:
        tir.Call: A handle to the barrier wait operation
    """
    mbarrier = _mbar_to_buffer_load(mbarrier)
    return tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_wait_parity"), mbarrier, parity)


def mbarrier_arrive(mbarrier: BarrierType):
    """Arrive at memory barrier.

    Args:
        mbarrier: BarrierType
            The memory barrier to arrive at
    """
    mbarrier = _mbar_to_buffer_load(mbarrier)
    return ptx_arrive_barrier(mbarrier)


def mbarrier_expect_tx(mbarrier: BarrierType, tx: int):
    """Set expected transaction count for memory barrier.

    Args:
        mbarrier: BarrierType
            The memory barrier to expect transaction count for
        tx: int
            The expected transaction count

    Returns:
        tir.Call: A handle to the barrier expectation operation
    """
    mbarrier = _mbar_to_buffer_load(mbarrier)
    return tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_expect_tx"), mbarrier, tx)


def warpgroup_arrive():
    """Signal warpgroup readiness for subsequent WGMMA operations.

    Returns:
        tir.Call: A handle to the warpgroup arrive operation.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.warpgroup_arrive"))


def warpgroup_commit_batch():
    """Commit the current warpgroup batch for WGMMA operations.

    Returns:
        tir.Call: A handle to the warpgroup commit batch operation.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.warpgroup_commit_batch"))


def warpgroup_wait(num_mma: int):
    """Wait for completion of the specified warpgroup batch.

    Args:
        num_mma: int
            Identifier of the warpgroup MMA batch to wait on.

    Returns:
        tir.Call: A handle to the warpgroup wait operation.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.warpgroup_wait"), num_mma)


def get_lane_idx(
    warp_size: int | PrimExpr | None = None,
) -> PrimExpr:
    """Return the logical lane index of the calling thread within a warp.

    Parameters
    ----------
    warp_size : Optional[int, PrimExpr]
        Logical warp (or wavefront) size. Defaults to 32 on NVIDIA and 64 on AMD.

    Example
    -------
    >>> lane = T.get_lane_idx()
    >>> custom_lane = T.get_lane_idx(64)  # override warp size explicitly

    Implementation Notes
    --------------------
    Lowers to the CUDA helper `tl::get_lane_idx(warp_size)` defined in
    `src/tl_templates/cuda/intrin.h`, which computes the lane index from the
    linear thread id using the provided `warp_size`.
    """
    warp_size_expr = _normalize_index_arg(warp_size)
    if warp_size_expr is None:
        return tir.call_intrin("int32", tir.op.Op.get("tl.get_lane_idx"))
    return tir.call_intrin("int32", tir.op.Op.get("tl.get_lane_idx"), warp_size_expr)


def get_warp_idx_sync(
    warp_size: int | PrimExpr | None = None,
) -> PrimExpr:
    """Return the canonical warp index, assuming the warp's threads are converged.

    Parameters
    ----------
    warp_size : Optional[int, PrimExpr]
        Logical warp size used for the index calculation.

    Example
    -------
    >>> warp = T.get_warp_idx_sync()
    >>> custom_warp = T.get_warp_idx_sync(64)

    Implementation Notes
    --------------------
    Emits `tl::get_warp_idx_sync(warp_size)` which divides the block-linear
    thread id by `warp_size`, matching the semantics of CUTLASS' canonical helpers.
    """
    warp_size_expr = _normalize_index_arg(warp_size)
    if warp_size_expr is None:
        return tir.call_intrin("int32", tir.op.Op.get("tl.get_warp_idx_sync"))
    return tir.call_intrin("int32", tir.op.Op.get("tl.get_warp_idx_sync"), warp_size_expr)


def get_warp_idx(
    warp_size: int | PrimExpr | None = None,
) -> PrimExpr:
    """Return the canonical warp index without synchronizing the warp.

    Parameters
    ----------
    warp_size : Optional[int, PrimExpr]
        Logical warp size used for the index calculation.

    Example
    -------
    >>> warp = T.get_warp_idx()
    >>> custom_warp = T.get_warp_idx(64)

    Implementation Notes
    --------------------
    Lowers to `tl::get_warp_idx(warp_size)` which divides the block-linear
    thread id by the provided `warp_size` without requiring warp convergence.
    """
    warp_size_expr = _normalize_index_arg(warp_size)
    if warp_size_expr is None:
        return tir.call_intrin("int32", tir.op.Op.get("tl.get_warp_idx"))
    return tir.call_intrin("int32", tir.op.Op.get("tl.get_warp_idx"), warp_size_expr)


def get_warp_group_idx(
    warp_size: int | PrimExpr | None = None,
    warps_per_group: int | PrimExpr | None = None,
) -> PrimExpr:
    """Return the canonical warp group index for the calling thread.

    Parameters
    ----------
    warp_size : Optional[int, PrimExpr]
        Logical warp size to use (defaults to 32 on NVIDIA / 64 on AMD).
    warps_per_group : Optional[int, PrimExpr]
        Number of warps per warp-group. Defaults to 4 on NVIDIA architectures.

    Example
    -------
    >>> group = T.get_warp_group_idx()
    >>> custom_group = T.get_warp_group_idx(32, 6)  # treat 6 warps as a group

    Implementation Notes
    --------------------
    Generates `tl::get_warp_group_idx(warp_size, warps_per_group)` which
    divides the block-linear thread id by `warp_size * warps_per_group`,
    matching the canonical ordering while allowing architecture-specific overrides.
    """
    warp_size_expr = _normalize_index_arg(warp_size)
    warps_per_group_expr = _normalize_index_arg(warps_per_group)
    args = []
    if warp_size_expr is not None:
        args.append(warp_size_expr)
    if warps_per_group_expr is not None:
        if warp_size_expr is None:
            raise ValueError("get_warp_group_idx expects `warp_size` when specifying `warps_per_group`.")
        args.append(warps_per_group_expr)
    return tir.call_intrin("int32", tir.op.Op.get("tl.get_warp_group_idx"), *args)


def shuffle_elect(thread_extent: int) -> PrimExpr:
    """Elect exactly one lane within a logical thread group.

    Parameters
    ----------
    thread_extent : int
        Size (in threads) of the group in which a single lane should be elected.
        Passing 0 elects a single lane in the entire thread block.

    Example
    -------
    >>> is_leader = T.shuffle_elect(64)
    >>> T.if_then_else(is_leader, do_leader_work(), T.evaluate(0))

    Implementation Notes
    --------------------
    Lowered to the CUDA helper `tl::tl_shuffle_elect<thread_extent>()` defined in
    `src/tl_templates/cuda/intrin.h`, which relies on
    `cutlass::canonical_warp_idx_sync()` and `cute::elect_one_sync()` (or
    `__shfl_sync`) to pick one lane per group.
    """
    return tir.call_intrin("bool", tir.op.Op.get("tl.tl_shuffle_elect"), thread_extent)


def warpgroup_fence_operand(
    buffer_or_ptr: BufferLikeType | PrimExpr,
    offset: int | PrimExpr = 0,
    num_regs: int | PrimExpr | None = None,
    dtype: DType | None = None,
):
    """Insert a warpgroup fence for the destination accumulator registers.

    This prevents NVCC from sinking uses of accumulator fragments past the corresponding
    WGMMA operations by issuing an empty inline assembly barrier on every register.

    Args:
        buffer_or_ptr: BufferLikeType | PrimExpr
            A buffer representing the accumulator fragment, a buffer load/region
            that identifies a starting element within the fragment, or a pointer expression
            (e.g., tvm_access_ptr/address_of/typed Var).
        offset: int | PrimExpr
            Element offset from the start of the accumulator fragment.
        num_regs: int | PrimExpr | None
            Number of 32-bit registers to fence. If None and a Buffer is provided, it will be
            derived from the buffer shape and dtype.
        dtype: DType | None
            Data type string of the accumulator elements. When passing a buffer or
            buffer-derived expression, dtype is inferred. It is required only when
            passing a raw pointer expression that cannot be inferred.

    Returns:
        tir.Call: A handle to the warpgroup fence operation.
    """
    if isinstance(buffer_or_ptr, BufferLoad):
        # Treat BufferLoad as a request to fence starting from the loaded element's address
        buf = buffer_or_ptr.buffer
        data_ptr = buf.data
        inferred_dtype = buf.dtype
        if dtype is not None and dtype != inferred_dtype:
            raise ValueError(f"dtype mismatch: provided {dtype}, buffer uses {inferred_dtype}.")
        dtype = inferred_dtype
        # Compute element offset from indices using strides if present, otherwise row-major
        if len(buf.strides) == len(buf.shape) and len(buf.strides) > 0:
            elem_off = 0
            for idx, stride in zip(buffer_or_ptr.indices, buf.strides):
                elem_off = elem_off + idx * stride
        else:
            elem_off = 0
            stride_acc = 1
            for idx, dim in zip(reversed(buffer_or_ptr.indices), reversed(buf.shape)):
                elem_off = elem_off + idx * stride_acc
                stride_acc = stride_acc * dim
        # Combine with user-provided offset
        offset = elem_off + convert(offset)
        if num_regs is None:
            raise ValueError("num_regs must be provided when passing a BufferLoad.")
        return evaluate(
            tir.call_intrin(
                "handle",
                tir.op.Op.get("tl.warpgroup_fence_operand"),
                dtype,
                data_ptr,
                convert(offset),
                convert(num_regs),
            )
        )

    if isinstance(buffer_or_ptr, tir.Buffer):
        data_ptr = buffer_or_ptr.data
        inferred_dtype = buffer_or_ptr.dtype
        if dtype is not None and dtype != inferred_dtype:
            raise ValueError(f"dtype mismatch: provided {dtype}, buffer uses {inferred_dtype}.")
        dtype = inferred_dtype
        if num_regs is None:
            total_elems = 1
            for dim in buffer_or_ptr.shape:
                if isinstance(dim, tir.IntImm):
                    total_elems *= int(dim)
                else:
                    raise ValueError("warpgroup_fence_operand requires num_regs when buffer shape is symbolic.")
            bits_per_elem = DataType(dtype).bits
            num_regs = (total_elems * bits_per_elem + 31) // 32
    elif isinstance(buffer_or_ptr, BufferRegion):
        buf = buffer_or_ptr.buffer
        data_ptr = buf.data
        inferred_dtype = buf.dtype
        if dtype is not None and dtype != inferred_dtype:
            raise ValueError(f"dtype mismatch: provided {dtype}, buffer uses {inferred_dtype}.")
        dtype = inferred_dtype
        # Compute element offset from region min using strides if present, otherwise row-major
        if len(buf.strides) == len(buf.shape) and len(buf.strides) > 0:
            elem_off = 0
            for r, stride in zip(buffer_or_ptr.region, buf.strides):
                elem_off = elem_off + r.min * stride
        else:
            elem_off = 0
            stride_acc = 1
            for r, dim in zip(reversed(buffer_or_ptr.region), reversed(buf.shape)):
                elem_off = elem_off + r.min * stride_acc
                stride_acc = stride_acc * dim
        # Combine with user-provided offset
        offset = elem_off + convert(offset)
        # Try derive num_regs from region extents if fully static; otherwise require user input
        if num_regs is None:
            total_elems = 1
            static = True
            for r in buffer_or_ptr.region:
                if isinstance(r.extent, tir.IntImm):
                    total_elems *= int(r.extent)
                else:
                    static = False
                    break
            if static:
                bits_per_elem = DataType(dtype).bits
                num_regs = (total_elems * bits_per_elem + 31) // 32
            else:
                raise ValueError("warpgroup_fence_operand requires num_regs when BufferRegion extent is symbolic.")
        return evaluate(
            tir.call_intrin(
                "handle",
                tir.op.Op.get("tl.warpgroup_fence_operand"),
                dtype,
                data_ptr,
                convert(offset),
                convert(num_regs),
            )
        )
    else:
        data_ptr = buffer_or_ptr
        # Try to infer dtype from common pointer expressions when not provided
        if dtype is None:
            inferred = None
            # Case 1: Pointer from Buffer.access_ptr -> tir.builtin.tvm_access_ptr
            if isinstance(data_ptr, Call) and data_ptr.op.same_as(tir.builtin.tvm_access_ptr()):
                # args[0] is a type annotation call; its dtype carries the element dtype
                inferred = str(data_ptr.args[0].dtype)
            # Case 2: Pointer from tir.address_of(BufferLoad(...))
            elif isinstance(data_ptr, Call) and data_ptr.op.same_as(tir.builtin.address_of()):
                # args[0] should be a BufferLoad; its dtype is the element dtype
                inferred = str(data_ptr.args[0].dtype)
            # Case 3: Typed pointer Var with PrimType element (typed TIR)
            elif hasattr(data_ptr, "type_annotation") and data_ptr.type_annotation is not None:
                try:
                    elem_ty = getattr(data_ptr.type_annotation, "element_type", None)
                    if elem_ty is not None and hasattr(elem_ty, "dtype"):
                        inferred = str(elem_ty.dtype)
                except Exception:
                    inferred = None
            if inferred is None:
                raise ValueError("dtype must be provided when passing a pointer expression and cannot be inferred.")
            dtype = inferred
        if num_regs is None:
            raise ValueError("num_regs must be provided when passing a pointer expression.")

    return evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.warpgroup_fence_operand"),
            dtype,
            data_ptr,
            convert(offset),
            convert(num_regs),
        )
    )


def wait_wgmma(id: int):
    """Wait for WGMMA (Warp Group Matrix Multiply-Accumulate) operations to complete.

    Args:
        id: int
            The id of the WGMMA operation to wait for

    Returns:
        tir.Call: A handle to the WGMMA wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.wait_wgmma"), id)


def barrier_wait(mbarrier: BarrierType, parity: int | Var):
    """Wait for a memory barrier to complete.

    Args:
        mbarrier: BarrierType
            The memory barrier to wait on
        parity: int | Var
            The parity value to wait for
    Returns:
        tir.Call: A handle to the barrier wait operation
    Current implementation is a sugar syntax for mbarrier_wait_parity, as we only support parity 0 and 1.
    """
    return mbarrier_wait_parity(mbarrier, parity)


def barrier_arrive(mbarrier: BarrierType):
    """Arrive at a memory barrier.

    Args:
        mbarrier: BarrierType
            The memory barrier to arrive at
    """
    return mbarrier_arrive(mbarrier)


def shfl_xor(value: int | PrimExpr | tir.Call, offset: int | PrimExpr | tir.Call):
    """Perform a shuffle operation with XOR offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
        offset: Optional[int, PrimExpr]
            The offset for the shuffle operation
    Returns:
        tir.Call: A handle to the shuffle operation
    """
    if _IS_HIP_AVAILABLE:
        return tir.call_extern(value.dtype, "__shfl_xor", value, offset)
    else:
        return tir.call_extern(value.dtype, "__shfl_xor_sync", 0xFFFFFFFF, value, offset)


def shfl_down(value: int | PrimExpr | tir.Call, offset: int | PrimExpr | tir.Call):
    """Perform a shuffle operation with down offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
    """
    if _IS_HIP_AVAILABLE:
        return tir.call_extern(value.dtype, "__shfl_down", value, offset)
    else:
        return tir.call_extern(value.dtype, "__shfl_down_sync", 0xFFFFFFFF, value, offset)


def shfl_up(value: int | PrimExpr | tir.Call, offset: int | PrimExpr | tir.Call):
    """Perform a shuffle operation with up offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
    """
    if _IS_HIP_AVAILABLE:
        return tir.call_extern(value.dtype, "__shfl_up", value, offset)
    else:
        return tir.call_extern(value.dtype, "__shfl_up_sync", 0xFFFFFFFF, value, offset)


def sync_threads(barrier_id: int = None, arrive_count: int = None):
    """Synchronize all threads in a block."""
    args = []
    if barrier_id is not None:
        args.append(barrier_id)
    if arrive_count is not None:
        args.append(arrive_count)
    return tir.call_intrin("int32", "tir.tvm_storage_sync", "shared", *args)


def sync_warp(mask: int = None):
    """Synchronize all threads in a warp."""
    if mask is not None:
        return tir.call_intrin("void", tir.op.Op.get("tl.sync_warp"), mask)
    return tir.call_intrin("void", tir.op.Op.get("tl.sync_warp"))


def shfl_sync(mask: int, value: int | PrimExpr, srcLane: int, width: int = None):
    """Receives data from a thread in the same warp."""
    if width is None:
        return tir.call_extern(value.dtype, "__shfl_sync", mask, value, srcLane)
    return tir.call_extern(value.dtype, "__shfl_sync", mask, value, srcLane, width)


def sync_global():
    """Synchronize all threads in the entire grid."""
    tx, ty, tz = get_thread_bindings()
    ex, ey, ez = get_block_extents()
    print(tx, ty, tz, ex, ey, ez)
    args = ["global", tx == 0 and ty == 0 and tz == 0, ex * ey * ez]
    return evaluate(tir.Call("handle", "tir.tvm_storage_sync", args))


def sync_grid():
    """Synchronize all threads in a grid."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.sync_grid"))


def initialize_wgmma_descriptor(
    descriptor: tir.Buffer,
    start_address: PrimExpr,
    layout_type_: int = 0,
    leading_byte_offset: int = 0,
    stride_byte_offset: int = 0,
) -> PrimExpr:
    """Initialize a WGMMA/UTCMMA shared-memory descriptor."""

    if not isinstance(descriptor, (BufferLoad, tir.Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, tir.Buffer) and (len(descriptor.shape) != 1 or descriptor.shape[0] != 1):
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(descriptor, [0])

    return evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.initialize_wgmma_descriptor"),
            descriptor,
            start_address,
            layout_type_,
            int(leading_byte_offset),
            int(stride_byte_offset),
        )
    )


def initialize_tcgen05_descriptor(
    descriptor: tir.Buffer,
    start_address: PrimExpr,
    leading_byte_offset: int,
    stride_byte_offset: int,
    base_offset: int = 0,
    leading_is_absolute: bool = False,
    swizzle_mode: int = 0,
) -> PrimExpr:
    """Initialize a TCGEN05 shared-memory descriptor."""

    if not isinstance(descriptor, (BufferLoad, tir.Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, tir.Buffer) and (len(descriptor.shape) != 1 or descriptor.shape[0] != 1):
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(descriptor, [0])

    return evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.initialize_tcgen05_descriptor"),
            descriptor,
            start_address,
            int(leading_byte_offset),
            int(stride_byte_offset),
            int(base_offset),
            tir.IntImm("int32", 1 if leading_is_absolute else 0),
            int(swizzle_mode),
        )
    )


def increase_descriptor_offset(descriptor: PrimExpr, offset: PrimExpr) -> PrimExpr:
    """
    Increase the offset of a memory descriptor.

    Parameters:
        descriptor (PrimExpr): The memory descriptor to modify.
        offset (PrimExpr): The offset value to increase.

    Returns:
        PrimExpr: A handle representing the modified descriptor.
    """
    if not isinstance(descriptor, (BufferLoad, tir.Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, tir.Buffer) and len(descriptor.shape) != 1 or descriptor.shape[0] != 1:
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(descriptor, [0])

    return evaluate(tir.call_intrin("handle", tir.op.Op.get("tl.increase_descriptor_offset"), descriptor, offset))


def loop_break():
    """Break out of the innermost loop."""
    return tir.call_intrin("handle", tir.op.Op.get("tl.loop_break"))


def cp_async_barrier_noinc(barrier: BarrierType):
    """Perform a ptx async copy barrier using cp.async.mbarrier.arrive.noinc."""
    barrier = _mbar_to_buffer_load(barrier)
    return tir.call_intrin("handle", tir.op.Op.get("tl.ptx_cp_async_barrier_noinc"), barrier)


def tcgen05_mma_arrive(mbar_ptr):
    """Signal UMMA (TCGEN05) barrier arrival for a shared-memory mbarrier pointer.

    Parameters
    ----------
    mbar_ptr : PrimExpr
        Pointer to the mbarrier object in shared memory (e.g., Barrier*).
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.tcgen05_mma_arrive"), mbar_ptr)


def ptx_mma_sm70(
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
):
    """TVM intrinsic for ptx tensor core mma instructions on SM70 (Volta).

    This intrinsic provides SM70-specific MMA operations that support m16n16k4 shape
    with FP16 inputs and FP16/FP32 accumulation.

    Parameters
    ----------

    shape : str
        The shape of mma fragment (e.g., "m16n16k4").

    A_layout : str
        The layout of multiplicand fragment A ("row" or "col").

    B_layout : str
        The layout of multiplicand fragment B ("row" or "col").

    A_dtype : str
        The data type of multiplicand fragment A (typically "fp16").

    B_dtype : str
        The data type of multiplicand fragment B (typically "fp16").

    C_dtype : str
        The data type of accumulator fragment C ("fp16" or "fp32").

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment B.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    Returns
    -------
    call : PrimExpr
        The call expression.

    Examples
    --------
    >>> T.ptx_mma_sm70(
    ...     "float16",
    ...     "m16n16k4",
    ...     "row",
    ...     "col",
    ...     "fp16",
    ...     "fp16",
    ...     "fp16",
    ...     A_local.data,
    ...     0,
    ...     B_local.data,
    ...     0,
    ...     C_local.data,
    ...     0,
    ... )
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ptx_mma_sm70"),
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
    )


def ldg32(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr:
    """Load 32 bits (4 bytes) from global memory using explicit PTX instructions.

    Usage: `T.ldg32(x[i])` or `T.ldg32(x[i:i+2])` emits `tl::ldg32(ptr)`.

    Args:
        src: A `Buffer`, `BufferRegion`, or `BufferLoad`.
        pred: Optional predicate condition. If False, the load is skipped.

    Returns:
        PrimExpr: The loaded 32-bit value.

    Example:
        >>> val = T.ldg32(x[i])
        >>> val = T.ldg32(x[i:i+2])  # load 2 x fp16
        >>> val = T.ldg32(x[i], pred=i < N)  # predicated load
    """
    if not isinstance(src, BufferLikeTypeTuple):
        raise TypeError(f"T.ldg32 expects Buffer, BufferRegion, or BufferLoad. Got {type(src)}: {src}")
    ptr = retrieve_ptr(src, access_type="r")
    if pred is None:
        return tir.call_intrin("uint32", tir.op.Op.get("tl.ldg32"), ptr)
    else:
        return tir.call_intrin("uint32", tir.op.Op.get("tl.ldg32"), ptr, pred)


def ldg64(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr:
    """Load 64 bits (8 bytes) from global memory using explicit PTX instructions.

    Usage: `T.ldg64(x[i])` or `T.ldg64(x[i:i+4])` emits `tl::ldg64(ptr)`.

    Args:
        src: A `Buffer`, `BufferRegion`, or `BufferLoad`.
        pred: Optional predicate condition. If False, the load is skipped.

    Returns:
        PrimExpr: The loaded 64-bit value.

    Example:
        >>> val = T.ldg64(x[i])
        >>> val = T.ldg64(x[i:i+4])  # load 4 x fp16
        >>> val = T.ldg64(x[i], pred=i < N)  # predicated load
    """
    if not isinstance(src, BufferLikeTypeTuple):
        raise TypeError(f"T.ldg64 expects Buffer, BufferRegion, or BufferLoad. Got {type(src)}: {src}")
    ptr = retrieve_ptr(src, access_type="r")
    if pred is None:
        return tir.call_intrin("uint32x2", tir.op.Op.get("tl.ldg64"), ptr)
    else:
        return tir.call_intrin("uint32x2", tir.op.Op.get("tl.ldg64"), ptr, pred)


def ldg128(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr:
    """Load 128 bits (16 bytes) from global memory using explicit PTX instructions.

    Usage: `T.ldg128(x[i])` or `T.ldg128(x[i:i+8])` emits `tl::ldg128(ptr)`.

    Args:
        src: A `Buffer`, `BufferRegion`, or `BufferLoad`.
        pred: Optional predicate condition. If False, the load is skipped.

    Returns:
        PrimExpr: The loaded 128-bit value.

    Example:
        >>> val = T.ldg128(x[i])
        >>> val = T.ldg128(x[i:i+8])  # load 8 x fp16
        >>> val = T.ldg128(x[i], pred=i < N)  # predicated load
    """
    if not isinstance(src, BufferLikeTypeTuple):
        raise TypeError(f"T.ldg128 expects Buffer, BufferRegion, or BufferLoad. Got {type(src)}: {src}")
    ptr = retrieve_ptr(src, access_type="r")
    if pred is None:
        return tir.call_intrin("uint32x4", tir.op.Op.get("tl.ldg128"), ptr)
    else:
        return tir.call_intrin("uint32x4", tir.op.Op.get("tl.ldg128"), ptr, pred)


def ldg256(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr:
    """Load 256 bits (32 bytes) from global memory using explicit PTX instructions.

    Usage: `T.ldg256(x[i])` or `T.ldg256(x[i:i+16])` emits `tl::ldg256(ptr)`.

    Args:
        src: A `Buffer`, `BufferRegion`, or `BufferLoad`.
        pred: Optional predicate condition. If False, the load is skipped.

    Returns:
        PrimExpr: The loaded 256-bit value.

    Example:
        >>> val = T.ldg256(x[i])
        >>> val = T.ldg256(x[i:i+16])  # load 16 x fp16
        >>> val = T.ldg256(x[i], pred=i < N)  # predicated load
    """
    if not isinstance(src, BufferLikeTypeTuple):
        raise TypeError(f"T.ldg256 expects Buffer, BufferRegion, or BufferLoad. Got {type(src)}: {src}")
    ptr = retrieve_ptr(src, access_type="r")
    if pred is None:
        return tir.call_intrin("uint32x8", tir.op.Op.get("tl.ldg256"), ptr)
    else:
        return tir.call_intrin("uint32x8", tir.op.Op.get("tl.ldg256"), ptr, pred)


def stg32(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None:
    """Store 32 bits (4 bytes) to global memory using explicit PTX instructions.

    Usage: `T.stg32(y[i], value)` emits `tl::stg32(ptr, value)`.

    Args:
        dst: A `Buffer`, `BufferRegion`, or `BufferLoad` indicating the destination.
        value: The 32-bit value to store.
        pred: Optional predicate condition. If False, the store is skipped.

    Example:
        >>> T.stg32(y[i], val)
        >>> T.stg32(y[i], val, pred=i < N)  # predicated store
    """
    if not isinstance(dst, BufferLikeTypeTuple):
        raise TypeError(f"T.stg32 expects Buffer, BufferRegion, or BufferLoad. Got {type(dst)}: {dst}")
    ptr = retrieve_ptr(dst, access_type="w")
    if pred is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg32"), ptr, value)
    else:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg32"), ptr, value, pred)


def stg64(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None:
    """Store 64 bits (8 bytes) to global memory using explicit PTX instructions.

    Usage: `T.stg64(y[i:i+2], value)` emits `tl::stg64(ptr, value)`.

    Args:
        dst: A `Buffer`, `BufferRegion`, or `BufferLoad` indicating the destination.
        value: The 64-bit value to store (e.g., uint2).
        pred: Optional predicate condition. If False, the store is skipped.

    Example:
        >>> T.stg64(y[i:i+2], val)
        >>> T.stg64(y[i:i+2], val, pred=i < N)  # predicated store
    """
    if not isinstance(dst, BufferLikeTypeTuple):
        raise TypeError(f"T.stg64 expects Buffer, BufferRegion, or BufferLoad. Got {type(dst)}: {dst}")
    ptr = retrieve_ptr(dst, access_type="w")
    if pred is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg64"), ptr, value)
    else:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg64"), ptr, value, pred)


def stg128(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None:
    """Store 128 bits (16 bytes) to global memory using explicit PTX instructions.

    Usage: `T.stg128(y[i:i+4], value)` emits `tl::stg128(ptr, value)`.

    Args:
        dst: A `Buffer`, `BufferRegion`, or `BufferLoad` indicating the destination.
        value: The 128-bit value to store (e.g., uint4).
        pred: Optional predicate condition. If False, the store is skipped.

    Example:
        >>> T.stg128(y[i:i+4], val)
        >>> T.stg128(y[i:i+4], val, pred=i < N)  # predicated store
    """
    if not isinstance(dst, BufferLikeTypeTuple):
        raise TypeError(f"T.stg128 expects Buffer, BufferRegion, or BufferLoad. Got {type(dst)}: {dst}")
    ptr = retrieve_ptr(dst, access_type="w")
    if pred is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg128"), ptr, value)
    else:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg128"), ptr, value, pred)


def stg256(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None:
    """Store 256 bits (32 bytes) to global memory using explicit PTX instructions.

    Usage: `T.stg256(y[i:i+8], value)` emits `tl::stg256(ptr, value)`.

    Args:
        dst: A `Buffer`, `BufferRegion`, or `BufferLoad` indicating the destination.
        value: The 256-bit value to store (e.g., ulonglong4).
        pred: Optional predicate condition. If False, the store is skipped.

    Example:
        >>> T.stg256(y[i:i+8], val)
        >>> T.stg256(y[i:i+8], val, pred=i < N)  # predicated store
    """
    if not isinstance(dst, BufferLikeTypeTuple):
        raise TypeError(f"T.stg256 expects Buffer, BufferRegion, or BufferLoad. Got {type(dst)}: {dst}")
    ptr = retrieve_ptr(dst, access_type="w")
    if pred is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg256"), ptr, value)
    else:
        return tir.call_intrin("handle", tir.op.Op.get("tl.stg256"), ptr, value, pred)
