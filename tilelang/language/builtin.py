# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir


def CreateListofMBarrierOp(*args):
    """Create a list of memory barrier operations.

    Args:
        *args: Variable arguments passed to the memory barrier creation operation

    Returns:
        tir.Call: A handle to the created list of memory barriers
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.CreateListofMBarrierOp"), *args)


def GetMBarrierOp(*args):
    """Retrieve a memory barrier operation.

    Args:
        *args: Variable arguments to specify which memory barrier to retrieve

    Returns:
        tir.Call: A handle to the requested memory barrier
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetMBarrierOp"), *args)


def CreateTMADescriptorOp(*args):
    """Create a Tensor Memory Access (TMA) descriptor.

    Args:
        *args: Variable arguments defining the TMA descriptor configuration

    Returns:
        tir.Call: A handle to the created TMA descriptor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.CreateTMADescriptorOp"), *args)


def TMALoadOp(*args):
    """Perform a Tensor Memory Access (TMA) load operation.

    Args:
        *args: Variable arguments specifying the TMA load parameters

    Returns:
        tir.Call: A handle to the TMA load operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.TMALoadOp"), *args)


def FenceProxyAsyncOp(*args):
    """Create a fence for asynchronous proxy operations.

    Args:
        *args: Variable arguments for fence configuration

    Returns:
        tir.Call: A handle to the fence operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.FenceProxyAsyncOp"), *args)


def TMAStoreArrive(*args):
    """Signal the arrival of a TMA store operation.

    Args:
        *args: Variable arguments for the store arrival operation

    Returns:
        tir.Call: A handle to the store arrive operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.TMAStoreArrive"), *args)


def TMAStoreWait(*args):
    """Wait for completion of TMA store operations.

    Args:
        *args: Variable arguments specifying which store operations to wait for

    Returns:
        tir.Call: A handle to the store wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.TMAStoreWait"), *args)


def SetMaxNReg(*args):
    """Set the maximum number of registers to use.

    Args:
        *args: Variable arguments specifying register allocation limits

    Returns:
        tir.Call: A handle to the register setting operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.SetMaxNReg"), *args)


def NoSetMaxNReg(*args):
    """Disable the maximum register limit setting.

    Args:
        *args: Variable arguments for the operation

    Returns:
        tir.Call: A handle to the register limit disable operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.NoSetMaxNReg"), *args)


def MBarrierWaitParity(*args):
    """Wait for memory barrier parity condition.

    Args:
        *args: Variable arguments specifying the parity wait condition

    Returns:
        tir.Call: A handle to the barrier wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.MBarrierWaitParity"), *args)


def MBarrierExpectTX(*args):
    """Set expected transaction count for memory barrier.

    Args:
        *args: Variable arguments specifying the expected transaction count

    Returns:
        tir.Call: A handle to the barrier expectation operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.MBarrierExpectTX"), *args)


def WaitWgmma(*args):
    """Wait for WGMMA (Warp Group Matrix Multiply-Accumulate) operations to complete.

    Args:
        *args: Variable arguments specifying which operations to wait for

    Returns:
        tir.Call: A handle to the WGMMA wait operation
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.WaitWgmma"), *args)
