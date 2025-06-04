# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tilelang import tvm as tvm
from tilelang.language import ptx_arrive_barrier, evaluate
from tilelang.language.kernel import get_thread_bindings, get_block_extents
from tvm import tir
from typing import Union, Any
from tvm.tir import PrimExpr, Var, Call


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


def no_set_max_nreg():
    """Disable the maximum register limit setting.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.no_set_max_nreg"))


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
    if isinstance(mbarrier, tir.Call):
        mbarrier = mbarrier
    elif isinstance(mbarrier, (tir.PrimExpr, int)):
        mbarrier = get_mbarrier(mbarrier)
    else:
        raise TypeError("mbarrier must be an integer or a tir.Call")
    return tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_wait_parity"), mbarrier, parity)


def mbarrier_arrive(mbarrier: Union[int, PrimExpr, tir.Call]):
    """Arrive at memory barrier.

    Args:
        mbarrier: Optional[int, PrimExpr]
            The memory barrier to arrive at
    """
    if isinstance(mbarrier, tir.Call):
        mbarrier = mbarrier
    elif isinstance(mbarrier, (tir.PrimExpr, int)):
        mbarrier = get_mbarrier(mbarrier)
    else:
        raise TypeError("mbarrier must be an integer or a tir.Call")
    return ptx_arrive_barrier(mbarrier)


def mbarrier_expect_tx(*args):
    """Set expected transaction count for memory barrier.

    Args:
        *args: Variable arguments specifying the expected transaction count

    Returns:
        tir.Call: A handle to the barrier expectation operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_expect_tx"), *args)


def wait_wgmma(*args):
    """Wait for WGMMA (Warp Group Matrix Multiply-Accumulate) operations to complete.

    Args:
        *args: Variable arguments specifying which operations to wait for

    Returns:
        tir.Call: A handle to the WGMMA wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.wait_wgmma"), *args)


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
    return tir.call_extern(value.dtype, "__shfl_xor_sync", 0xffffffff, value, offset)


def shfl_down(value: Union[int, PrimExpr, tir.Call], offset: Union[int, PrimExpr, tir.Call]):
    """Perform a shuffle operation with down offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
    """
    return tir.call_extern(value.dtype, "__shfl_down_sync", 0xffffffff, value, offset)


def shfl_up(value: Union[int, PrimExpr, tir.Call], offset: Union[int, PrimExpr, tir.Call]):
    """Perform a shuffle operation with up offset.

    Args:
        value: Optional[int, PrimExpr]
            The value to shuffle
    """
    return tir.call_extern(value.dtype, "__shfl_up_sync", 0xffffffff, value, offset)


def sync_threads():
    """Synchronize all threads in a warp.
    """
    return tir.op.tvm_storage_sync("shared")


def sync_thread_partial(barrier_id: Union[int, PrimExpr, tir.Call]):
    """Synchronize threads within a warp.

    Args:
        barrier_id: Optional[int, PrimExpr]
            The memory barrier to synchronize

    Returns:
        tir.Call: A handle to the synchronization operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.sync_thread_partial"), barrier_id)


def sync_global():
    """Synchronize all threads in a block.
    """
    tx, ty, tz = get_thread_bindings()
    ex, ey, ez = get_block_extents()
    print(tx, ty, tz, ex, ey, ez)
    args = ["global", tx == 0 and ty == 0 and tz == 0, ex * ey * ez]
    return evaluate(tir.Call("handle", "tir.tvm_storage_sync", args))
