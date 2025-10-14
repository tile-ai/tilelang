"""The language interface for tl programs."""

from tilelang import tvm as tvm
from tilelang.language import ptx_arrive_barrier, evaluate
from tilelang.language.kernel import get_thread_bindings, get_block_extents
from tilelang.utils.target import check_hip_availability
from tvm import DataType, tir
from tvm.runtime import convert
from typing import Union, Any
from tvm.tir import PrimExpr, Var, Call, Buffer, BufferLoad

_IS_HIP_AVAILABLE = check_hip_availability()


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


def warpgroup_fence_operand(buffer_or_ptr: Union[Buffer, PrimExpr],
                            offset: Union[int, PrimExpr] = 0,
                            num_regs: Union[int, PrimExpr, None] = None,
                            dtype: Union[str, None] = None):
    """Insert a warpgroup fence for the destination accumulator registers.

    This prevents NVCC from sinking uses of accumulator fragments past the corresponding
    WGMMA operations by issuing an empty inline assembly barrier on every register.

    Args:
        buffer_or_ptr: Union[Buffer, PrimExpr]
            Either a buffer representing the accumulator fragment or a pointer expression.
        offset: Union[int, PrimExpr]
            Element offset from the start of the accumulator fragment.
        num_regs: Union[int, PrimExpr, None]
            Number of 32-bit registers to fence. If None and a Buffer is provided, it will be
            derived from the buffer shape and dtype.
        dtype: Optional[str]
            Data type string of the accumulator elements. Required when passing a pointer.

    Returns:
        tir.Call: A handle to the warpgroup fence operation.
    """
    if isinstance(buffer_or_ptr, BufferLoad):
        raise TypeError("Expected a buffer handle or pointer expression, got BufferLoad.")

    if isinstance(buffer_or_ptr, Buffer):
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
                    raise ValueError(
                        "warpgroup_fence_operand requires num_regs when buffer shape is symbolic.")
            bits_per_elem = DataType(dtype).bits
            num_regs = (total_elems * bits_per_elem + 31) // 32
    else:
        data_ptr = buffer_or_ptr
        if dtype is None:
            raise ValueError("dtype must be provided when passing a pointer expression.")
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
        ))


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


def initialize_wgmma_descriptor(
    descriptor: Buffer,
    start_address: PrimExpr,
    layout_type_: int = 0,
    leading_byte_offset: int = 0,
    stride_byte_offset: int = 0,
) -> PrimExpr:
    """Initialize a WGMMA/UTCMMA shared-memory descriptor."""

    if not isinstance(descriptor, (BufferLoad, Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, Buffer) and (len(descriptor.shape) != 1 or descriptor.shape[0] != 1):
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(
        descriptor, [0])

    return evaluate(
        tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.initialize_wgmma_descriptor"),
            descriptor,
            start_address,
            layout_type_,
            int(leading_byte_offset),
            int(stride_byte_offset),
        ))


def initialize_tcgen05_descriptor(
    descriptor: Buffer,
    start_address: PrimExpr,
    leading_byte_offset: int,
    stride_byte_offset: int,
    base_offset: int = 0,
    leading_is_absolute: bool = False,
    swizzle_mode: int = 0,
) -> PrimExpr:
    """Initialize a TCGEN05 shared-memory descriptor."""

    if not isinstance(descriptor, (BufferLoad, Buffer)):
        raise TypeError("Descriptor must be a tvm.tir.Buffer or tvm.tir.BufferLoad.")

    if isinstance(descriptor, Buffer) and (len(descriptor.shape) != 1 or descriptor.shape[0] != 1):
        raise ValueError("Descriptor must be a 1D buffer of size 1.")

    descriptor = descriptor if isinstance(descriptor, BufferLoad) else tir.BufferLoad(
        descriptor, [0])

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
        ))


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
