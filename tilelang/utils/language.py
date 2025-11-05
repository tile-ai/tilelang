from __future__ import annotations
from tvm.tir import Buffer
from functools import reduce
from tvm import IRModule
from tvm.tir import PrimFunc
from tvm import ir, tir

# Scope Checkers for TVM Buffers
# These utility functions check the memory scope of a given TVM buffer.


def is_global(buffer: Buffer) -> bool:
    """
    Check if the buffer is in the global memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in global memory, False otherwise.
    """
    return buffer.scope() == "global"


def is_shared(buffer: Buffer, allow_dynamic: bool = True) -> bool:
    """
    Check if the buffer is in the shared memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in shared memory, False otherwise.
    """
    conditions = [False]
    conditions.append(buffer.scope() == "shared")
    if allow_dynamic:
        conditions.append(is_shared_dynamic(buffer))
    return any(conditions)


def is_shared_dynamic(buffer: Buffer) -> bool:
    """
    Check if the buffer is in the dynamic shared memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in dynamic shared memory, False otherwise.
    """
    return buffer.scope() == "shared.dyn"


def is_tensor_memory(buffer: Buffer) -> bool:
    """
    Check if the buffer is in tensor memory scope (e.g., shared.tmem).

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in tensor memory, False otherwise.
    """
    return buffer.scope().startswith("shared.tmem")


def is_local(buffer: Buffer) -> bool:
    """
    Check if the buffer is in the local memory scope.

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is in local memory, False otherwise.
    """
    return buffer.scope() == "local"


def is_fragment(buffer: Buffer) -> bool:
    """
    Check if the buffer is a fragment (e.g., for matrix multiplication operations).

    Args:
        buffer (Buffer): The TVM buffer to check.

    Returns:
        bool: True if the buffer is a fragment, False otherwise.
    """
    return buffer.scope().startswith("local.fragment")


def get_buffer_elems(buffer: Buffer) -> int:
    """
    Get the number of elements in the buffer.
    """
    return reduce(lambda x, y: x * y, buffer.shape)


def array_reduce(array: list[int]) -> int:
    """
    Reduce an array of integers to a single integer.

    Args:
        array (List[int]): The array of integers to reduce.

    Returns:
        int: The reduced integer.
    """
    return reduce(lambda x, y: x * y, array)


def retrieve_func_from_module(ir_module: IRModule) -> PrimFunc:
    """
    Retrieve the single PrimFunc from an IRModule.

    Args:
        ir_module (IRModule): The TVM IRModule to extract the function from.
            The module should contain exactly one global function.

    Returns:
        PrimFunc: The single function contained in the module.

    Raises:
        ValueError: If ir_module is not an IRModule.
        AssertionError: If the module contains more than one global function.
    """
    if not isinstance(ir_module, IRModule):
        raise ValueError("Not supported type: ", type(ir_module))
    assert len(ir_module.get_global_vars()) == 1, (
        "The optimized module should only have one global variable for default schedule.")
    func = list(ir_module.functions.values())[0]
    return func


def get_buffer_region_from_load(buffer_load: tir.BufferLoad) -> tir.BufferRegion | None:
    """
    Get the buffer region from a buffer load.

    May encounter buffer load like C[0:128, 0:32], ref to pull request
    for buffer wise op: https://github.com/apache/tvm/pull/14693
    convert load to region
    """
    buffer, indices = buffer_load.buffer, buffer_load.indices
    regions = []
    found_ramp: bool = False
    for indice in indices:
        if isinstance(indice, tir.Ramp):
            regions.append(ir.Range.from_min_extent(indice.base, indice.lanes))
            found_ramp = True
        elif isinstance(indice, tir.PrimExpr):
            regions.append(ir.Range.from_min_extent(indice, 1))
        else:
            raise ValueError("Unsupported type: ", type(indice))
    if found_ramp:
        return tir.BufferRegion(buffer, regions)
    else:
        return None
