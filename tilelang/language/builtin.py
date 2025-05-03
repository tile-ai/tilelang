# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tilelang import tvm as tvm
from tilelang.language import ptx_arrive_barrier
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


def set_max_nreg(*args):
    """Set the maximum number of registers to use.

    Args:
        *args: Variable arguments specifying register allocation limits

    Returns:
        tir.Call: A handle to the register setting operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.set_max_nreg"), *args)


def no_set_max_nreg(*args):
    """Disable the maximum register limit setting.

    Args:
        *args: Variable arguments for the operation

    Returns:
        tir.Call: A handle to the register limit disable operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.no_set_max_nreg"), *args)


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
