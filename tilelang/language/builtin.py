"""The language interface for tl programs."""

from tilelang import tvm as tvm
from tilelang.language import ptx_arrive_barrier, evaluate
from tilelang.language.kernel import get_thread_bindings, get_block_extents
from tilelang.utils.target import check_hip_availability
from tvm import tir
from typing import Union, Any, Optional
from tvm.tir import PrimExpr, Var, Call, Buffer, BufferLoad

_IS_HIP_AVAILABLE = check_hip_availability()


def _normalize_index_arg(value: Optional[Union[int, PrimExpr]]) -> Optional[PrimExpr]:
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


def create_list_of_mbarrier(*args: Any) -> Call:
    """
    Create a list of memory barrier handles.

    Parameters
    ----------
    *args : list or Any
        Either a single list of arguments, or multiple arguments directly.

    Returns
    -------
    tvm.tir.Call
        Handle to the created list of memory barriers.

    Raises
    ------
    TypeError
        If the input is not a list or variadic arguments.

    Examples
    --------
    >>> create_list_of_mbarrier([128, 128])
    >>> create_list_of_mbarrier(128, 128)
    """
    if len(args) == 1 and isinstance(args[0], list):
        return tir.call_intrin("handle", tir.op.Op.get("tl.create_list_of_mbarrier"), *args[0])
    elif len(args) >= 1:
        return tir.call_intrin("handle", tir.op.Op.get("tl.create_list_of_mbarrier"), *args)
    else:
        raise TypeError("create_list_of_mbarrier expects a list or one or more arguments.")


def get_mbarrier(*args):
    """Retrieve a memory barrier operation.

    Args:
        *args: Variable arguments to specify which memory barrier to retrieve

    Returns:
        tir.Call: A handle to the requested memory barrier
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), *args)


def create_tma_descriptor(*args):
    """Create a Tensor Memory Access (TMA) descriptor.

    Args:
        *args: Variable arguments defining the TMA descriptor configuration

    Returns:
        tir.Call: A handle to the created TMA descriptor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.create_tma_descriptor"), *args)


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
    """Increment the maximum number of registers to use.
    """
    return set_max_nreg(reg_count, 1)


def dec_max_nreg(reg_count: int):
    """Decrement the maximum number of registers to use.
    """
    return set_max_nreg(reg_count, 0)


def annotate_producer_reg_dealloc(reg_count: int = 24):
    """Annotate the producer reg dealloc.
    """
    return dec_max_nreg(reg_count)


def annotate_consumer_reg_alloc(reg_count: int = 240):
    """Annotate the consumer reg alloc.
    """
    return inc_max_nreg(reg_count)


def no_set_max_nreg():
    """Disable the maximum register limit setting.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.no_set_max_nreg"))


def disable_warp_group_reg_alloc():
    """Disable the warp group reg alloc.
    """
    return no_set_max_nreg()


def mbarrier_wait_parity(mbarrier: Union[int, PrimExpr, tir.Call], parity: Union[int, Var]):
    """Wait for memory barrier parity condition.

    Args:
        mbarrier: Optional[int, PrimExpr]
            The memory barrier to wait on
        parity: Optional[int, Var]
            The parity value to wait for
    Examples:
        .. code-block:: python

            # Wait for parity 0 on barrier 0
            T.mbarrier_wait_parity(0, 0)

            # Wait for parity value in variable ko on barrier 1
            T.mbarrier_wait_parity(1, ko)

            # Wait using barrier handle
            barrier = T.get_mbarrier(0)
            T.mbarrier_wait_parity(barrier, 1)

            # Common usage in pipelined kernels:
            for ko in range(num_stages):
                # Producer waits for consumer to finish previous iteration
                T.mbarrier_wait_parity(1, ko ^ 1)
                # Producer copies data
                T.copy(A_global, A_shared)
                # Producer signals data ready
                T.mbarrier_arrive(0)

                # Consumer waits for producer data
                T.mbarrier_wait_parity(0, ko)
                # Consumer computes
                T.gemm(A_shared, B_shared, C_local)
                # Consumer signals completion
                T.mbarrier_arrive(1)
    Returns:
        tir.Call: A handle to the barrier wait operation
    """
    if isinstance(mbarrier, (tir.Call, tir.BufferLoad)):
        mbarrier = mbarrier
    elif isinstance(mbarrier, (tir.PrimExpr, int)):
        mbarrier = get_mbarrier(mbarrier)
    elif isinstance(mbarrier, tir.Buffer):
        mbarrier = tir.BufferLoad(mbarrier, [0])
    else:
        raise TypeError(f"mbarrier must be an integer or a tir.Call, but got {type(mbarrier)}")
    return tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_wait_parity"), mbarrier, parity)


def mbarrier_arrive(mbarrier: Union[int, PrimExpr, tir.Call]):
    """Arrive at memory barrier.

    Args:
        mbarrier: Optional[int, PrimExpr]
            The memory barrier to arrive at
    """
    if isinstance(mbarrier, (tir.Call, tir.BufferLoad)):
        mbarrier = mbarrier
    elif isinstance(mbarrier, (tir.PrimExpr, int)):
        mbarrier = get_mbarrier(mbarrier)
    elif isinstance(mbarrier, tir.Buffer):
        mbarrier = tir.BufferLoad(mbarrier, [0])
    else:
        raise TypeError(f"mbarrier must be an integer or a tir.Call, but got {type(mbarrier)}")
    return ptx_arrive_barrier(mbarrier)


def mbarrier_expect_tx(*args):
    """Set expected transaction count for memory barrier.

    Args:
        *args: Variable arguments specifying the expected transaction count

    Returns:
        tir.Call: A handle to the barrier expectation operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_expect_tx"), *args)


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


def get_lane_idx(warp_size: Optional[Union[int, PrimExpr]] = None,) -> PrimExpr:
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


def get_warp_idx_sync(warp_size: Optional[Union[int, PrimExpr]] = None,) -> PrimExpr:
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


def get_warp_idx(warp_size: Optional[Union[int, PrimExpr]] = None,) -> PrimExpr:
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
    warp_size: Optional[Union[int, PrimExpr]] = None,
    warps_per_group: Optional[Union[int, PrimExpr]] = None,
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
            raise ValueError("get_warp_group_idx expects `warp_size` when specifying "
                             "`warps_per_group`.")
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


def wait_wgmma(id: int):
    """Wait for WGMMA (Warp Group Matrix Multiply-Accumulate) operations to complete.

    Args:
        id: int
            The id of the WGMMA operation to wait for

    Returns:
        tir.Call: A handle to the WGMMA wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.wait_wgmma"), id)


def barrier_wait(barrier_id: Union[int, PrimExpr, tir.Call], parity: Union[int, Var, None] = None):
    """Wait for a memory barrier to complete.

    Args:
        barrier_id: Optional[int, PrimExpr]
            The memory barrier to wait on
        parity: Optional[int, Var]
            The parity value to wait for
    Returns:
        tir.Call: A handle to the barrier wait operation
    Current implementation is a sugar syntax for mbarrier_wait_parity, as we only support parity 0 and 1.
    """
    return mbarrier_wait_parity(barrier_id, parity)


def barrier_arrive(barrier_id: Union[int, PrimExpr, tir.Call]):
    """Arrive at a memory barrier.

    Args:
        barrier_id: Optional[int, PrimExpr]
            The memory barrier to arrive at
    """
    return mbarrier_arrive(barrier_id)


def shfl_xor(value: Union[int, PrimExpr, tir.Call], offset: Union[int, PrimExpr, tir.Call]):
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
        return tir.call_extern(value.dtype, "__shfl_xor_sync", 0xffffffff, value, offset)


def shfl_down(value: Union[int, PrimExpr, tir.Call], offset: Union[int, PrimExpr, tir.Call]):
    """Perform a shuffle operation with down offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
    """
    if _IS_HIP_AVAILABLE:
        return tir.call_extern(value.dtype, "__shfl_down", value, offset)
    else:
        return tir.call_extern(value.dtype, "__shfl_down_sync", 0xffffffff, value, offset)


def shfl_up(value: Union[int, PrimExpr, tir.Call], offset: Union[int, PrimExpr, tir.Call]):
    """Perform a shuffle operation with up offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
    """
    if _IS_HIP_AVAILABLE:
        return tir.call_extern(value.dtype, "__shfl_up", value, offset)
    else:
        return tir.call_extern(value.dtype, "__shfl_up_sync", 0xffffffff, value, offset)


def sync_threads(barrier_id: int = None, arrive_count: int = None):
    """Synchronize all threads in a block.
    """
    args = []
    if barrier_id is not None:
        args.append(barrier_id)
    if arrive_count is not None:
        args.append(arrive_count)
    return tir.call_intrin("int32", "tir.tvm_storage_sync", "shared", *args)


def sync_global():
    """Synchronize all threads in the entire grid.
    """
    tx, ty, tz = get_thread_bindings()
    ex, ey, ez = get_block_extents()
    print(tx, ty, tz, ex, ey, ez)
    args = ["global", tx == 0 and ty == 0 and tz == 0, ex * ey * ez]
    return evaluate(tir.Call("handle", "tir.tvm_storage_sync", args))


def sync_grid():
    """Synchronize all threads in a grid.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.sync_grid"))


def initialize_descriptor(descriptor: Buffer,
                          start_address: PrimExpr,
                          layout_type_: int = 0,
                          leading_byte_offset: int = 0,
                          stride_byte_offset: int = 0) -> PrimExpr:
    """
    Initialize a memory descriptor with the given parameters.

    Parameters:
        descriptor (Buffer): The memory descriptor to initialize.
        start_address (PrimExpr): The starting address of the memory region.
        layout_type_ (int, optional): Layout type identifier. Defaults to 0.
        leading_byte_offset (int, optional): Leading byte offset. Defaults to 0.
        stride_byte_offset (int, optional): Stride byte offset. Defaults to 0.

    Returns:
        PrimExpr: A handle representing the initialized descriptor.
    """

    if not isinstance(descriptor, (BufferLoad, Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, Buffer) and len(descriptor.shape) != 1 or descriptor.shape[0] != 1:
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(
        descriptor, [0])

    return evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.initialize_descriptor"), descriptor,
                        start_address, layout_type_, int(leading_byte_offset),
                        int(stride_byte_offset)))


def increase_descriptor_offset(descriptor: PrimExpr, offset: PrimExpr) -> PrimExpr:
    """
    Increase the offset of a memory descriptor.

    Parameters:
        descriptor (PrimExpr): The memory descriptor to modify.
        offset (PrimExpr): The offset value to increase.

    Returns:
        PrimExpr: A handle representing the modified descriptor.
    """
    if not isinstance(descriptor, (BufferLoad, Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, Buffer) and len(descriptor.shape) != 1 or descriptor.shape[0] != 1:
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(
        descriptor, [0])

    return evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.increase_descriptor_offset"), descriptor,
                        offset))


def loop_break():
    """Break out of the innermost loop.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.loop_break"))


def cp_async_barrier_noinc(barrier_id: Union[int, PrimExpr, tir.Call]):
    """Perform a ptx async copy barrier using cp.async.mbarrier.arrive.noinc.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ptx_cp_async_barrier_noinc"), barrier_id)
