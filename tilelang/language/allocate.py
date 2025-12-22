"""Memory allocation utilities for Tile-AI programs.

This module provides a set of functions for allocating different types of memory buffers
in Tile-AI programs. It wraps TVM's buffer allocation functionality with convenient
interfaces for different memory scopes.

Available allocation functions:
    - alloc_shared: Allocates shared memory buffers for inter-thread communication
    - alloc_local: Allocates local memory buffers for thread-private storage
    - alloc_fragment: Allocates fragment memory buffers for specialized operations
    - alloc_var: Allocates single-element variable buffers

Each function takes shape and dtype parameters and returns a TVM buffer object
with the appropriate memory scope.
"""

from __future__ import annotations
from typing import TypeVar, overload, Literal, Callable

# Python 3.9 compatibility for advanced typing features (PEP 646)
try:
    from typing import TypeVarTuple, Unpack  # type: ignore[attr-defined]
except Exception:
    from typing_extensions import TypeVarTuple, Unpack  # type: ignore
from tilelang import tvm as tvm
from tvm.script import tir as T
from tvm.tir import PrimExpr
from tvm.script.parser.tir import block_attr
from tvm.tir.buffer import Buffer
from tvm.tir.expr import FloatImm, IntImm
from .v2 import dtypes as _dtypes
from .v2.dtypes import dtype as tl_dtype
from .v2.builder import OutTensor
from .v2.annot import Tensor, SharedBuffer, LocalBuffer, FragmentBuffer

_Shapes = TypeVarTuple("_Shapes")
_DType = TypeVar("_DType")


def alloc_shared(shape: tuple[Unpack[_Shapes]], dtype: _DType, scope="shared.dyn") -> SharedBuffer[Callable[[Unpack[_Shapes]]], _DType]:
    """Allocate a shared memory buffer for inter-thread communication.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "shared.dyn"

    Returns:
        T.Buffer: A TVM buffer object allocated in shared memory
    """
    if dtype == "bool":
        # lei: This is a hack to handle bool type.
        # Because tilelang's merge smem pass cannot merge bool type currently.
        scope = "shared"
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_local(shape: tuple[Unpack[_Shapes]], dtype: _DType, scope="local") -> LocalBuffer[Callable[[Unpack[_Shapes]]], _DType]:
    """Allocate a local memory buffer for thread-private storage.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local"

    Returns:
        T.Buffer: A TVM buffer object allocated in local memory
    """
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_fragment(
    shape: tuple[Unpack[_Shapes]], dtype: _DType, scope="local.fragment"
) -> FragmentBuffer[Callable[[Unpack[_Shapes]]], _DType]:
    """Allocate a fragment memory buffer for specialized operations.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local.fragment"

    Returns:
        T.Buffer: A TVM buffer object allocated in fragment memory
    """
    return T.alloc_buffer(shape, dtype, scope=scope)


@overload
def alloc_var(dtype: str, init: PrimExpr | int | float, scope: str = "local.var") -> Buffer: ...


@overload
def alloc_var(dtype: str, scope: str = "local.var", *, init: PrimExpr | int | float | None = None) -> Buffer: ...


def alloc_var(dtype, *args, scope="local.var", init: PrimExpr | None = None):
    """Allocate a single-element variable buffer.

    Args:
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        *args: Optional positional arguments. A single positional string is treated
            as the scope for backward compatibility. A single non-string positional
            argument (or keyword ``init``) specifies the initializer. When two
            positional arguments are provided, they are interpreted as
            ``(init, scope)``.
        scope (str, optional): The memory scope. Defaults to "local.var".
            Use as keyword argument for clarity when also providing an initializer.
        init (PrimExpr, optional): The optional initializer value. When provided,
            the generated code will initialize the variable with this value instead
            of defaulting to zero.
    Examples:
        a = T.alloc_var('int32', 1) # var with init 1
        a = T.alloc_var('int32', 'local.var') # var with local.var scope
        a = T.alloc_var('int32', 1, 'local.var') # var with init 1 and local.var scope
        a = T.alloc_var('int32', 'local.var', init=1) # var with init 1 and local.var scope
        a = T.alloc_var('int32', init=1) # var with init 1 and local.var scope
    Returns:
        T.Buffer: A TVM buffer object allocated as a single-element variable
    """
    parsed_scope = scope
    parsed_init = init

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, str) and parsed_init is None and scope == "local.var":
            parsed_scope = arg
        else:
            if parsed_init is not None:
                raise TypeError("Initializer specified multiple times in alloc_var.")
            parsed_init = arg
    elif len(args) == 2:
        if parsed_init is not None:
            raise TypeError("Initializer specified multiple times in alloc_var.")
        parsed_init, parsed_scope_arg = args
        if not isinstance(parsed_scope_arg, str):
            raise TypeError("Scope must be provided as a string in alloc_var.")
        parsed_scope = parsed_scope_arg
    elif len(args) > 2:
        raise TypeError(f"alloc_var expected at most 3 positional arguments but got {len(args) + 1}.")

    if not isinstance(parsed_scope, str):
        raise TypeError("Scope must be a string in alloc_var.")

    buffer = T.alloc_buffer([1], dtype, scope=parsed_scope)
    if parsed_init is not None:
        if isinstance(parsed_init, (int, float, IntImm, FloatImm)):
            block_attr({"tl.local_var_init": {buffer.data: tl_dtype(dtype)(parsed_init)}})
        else:
            T.buffer_store(buffer, parsed_init, 0)
    return buffer


def alloc_barrier(arrive_count: int):
    """Allocate a barrier buffer.

    Args:
        arrive_count (int): The number of threads that need to arrive at the barrier

    Returns:
        T.Buffer: A TVM buffer object allocated as a barrier
    """
    return T.alloc_buffer([arrive_count], _dtypes.uint64, scope="shared.barrier")


def alloc_tmem(shape, dtype):
    """
    Allocate a Tensor Memory (TMEM) buffer for use with 5th generation Tensor Core operations (e.g., TCGEN5.MMA).

    TMEM is a dedicated on-chip memory introduced in Hopper GPUs, designed to reduce register pressure and enable asynchronous, single-threaded MMA operations. It is organized as a 2D array of 512 columns by 128 rows (lanes), with each cell being 32 bits. Allocation is performed in units of columns, and every lane of a column is allocated together.

    Key properties and requirements:
        - The number of columns allocated must be a power of 2 and at least 32.
        - TMEM allocations are dynamic and must be explicitly deallocated.
        - Both allocation and deallocation must be performed by the same warp.
        - The base address of the TMEM allocation is stored in shared memory and used as the offset for TCGEN5.MMA accumulator tensors.
        - Only TCGEN5.MMA and specific TMEM load/store instructions can access TMEM; all pre-processing must occur before data is loaded into TMEM, and all post-processing after data is retrieved.
        - The number of columns allocated should not increase between any two allocations in the execution order within the CTA.

    Args:
        num_cols (int): Number of columns to allocate in TMEM. Must be a power of 2 and >= 32 but less than or equal to 512.

    Returns:
        T.Buffer: A TVM buffer object allocated in TMEM scope, suitable for use as an accumulator or operand in TCGEN5.MMA operations.

    Note:
        - TMEM is only available on supported architectures (e.g., Hopper and later).
        - The buffer returned should be used according to TMEM access restrictions and deallocated appropriately.
    """

    assert len(shape) == 2, "shape must be a 2D tensor for TMEM allocation"
    return T.alloc_buffer(shape, dtype, scope="shared.tmem")


def alloc_reducer(shape, dtype, op="sum", replication=None):
    """
    Allocate a reducer buffer.

    Modifications needs to conform with `op`,
    such as `op="sum"` requires `reducer[...] += ...` and
    `op="max"` requires `reducer[...] = T.max(reducer[...], ...)`.

    Only after T.fill with proper initializer the reduction may begin;
    only after T.finalize_reducer the partial results will be available.

    For `op="sum"`, filled value must be 0; for min and max, the filled initializer will become max or min clamper correspondingly.
    You may want to use `T.max_value` for min and `T.min_value` for max.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        op (str): The reduce operation corresponded with the reducer
        replication (str | None): Replication strategy, can be "all" or "none". Defaults to not specified, and the compiler will do whatever it want.

    Returns:
        T.Buffer: A TVM buffer object allocated in thread-private storage, available to reduce values in T.Parallel loops.
    """

    assert op in ["sum", "max", "min"]
    # TODO: support automatic layout
    if replication is None:
        replication = "none"
    assert replication in ["all", "none"]

    reducer = T.alloc_buffer(shape, dtype, scope="local.fragment")
    block_attr({"reducer_info": {reducer.data: {"rep": replication, "op": op}}})

    return reducer


DescKind = Literal["wgmma", "tcgen05_smem", "tcgen05_instr"]


def alloc_descriptor(
    kind: DescKind = "wgmma",
    dtype: str = _dtypes.uint64,
):
    """Allocate a descriptor buffer for WGMMA and TCGEN5.MMA.

    Args:
        kind: The descriptor kind, one of "wgmma", "tcgen05" ("utcmma" as alias).

    Returns:
        T.Buffer: A TVM buffer object allocated as a descriptor
    """

    scope = "local.descriptor." + kind
    # Buffer naming via `name` is not supported by this TVM builder signature;
    # keep parameter for forward-compat, but do not pass it.
    return T.alloc_buffer([1], dtype, scope=scope)


def alloc_wgmma_desc(dtype: str = _dtypes.uint64):
    return alloc_descriptor("wgmma", dtype=dtype)


def alloc_tcgen05_smem_desc(dtype: str = _dtypes.uint64):
    return alloc_descriptor("tcgen05_smem", dtype=dtype)


def alloc_tcgen05_instruction_desc(dtype: str = _dtypes.uint32):
    return alloc_descriptor("tcgen05_instr", dtype=dtype)


# Alias: short name consistent with imports
def alloc_tcgen05_instr_desc(dtype: str = _dtypes.uint32):
    return alloc_tcgen05_instruction_desc(dtype)


@overload
def empty(shape: tuple[Unpack[_Shapes]], dtype: str = _dtypes.float32) -> Tensor[Callable[[Unpack[_Shapes]]], _DType]: ...


def empty(*shape: Unpack[_Shapes], dtype: str = _dtypes.float32) -> Tensor[Callable[[Unpack[_Shapes]]], _DType]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return OutTensor(shape[0], dtype)
    elif len(shape) == 2 and isinstance(shape[0], (tuple, list)) and isinstance(shape[1], str):
        return OutTensor(shape[0], shape[1])
    elif all([isinstance(x, (int, PrimExpr)) for x in shape]):
        return OutTensor(shape, dtype)
    else:
        raise RuntimeError(f"Invalid shape {shape}")
