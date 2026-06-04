"""Memory allocation utilities for Tile-AI programs.

This module provides a set of functions for allocating different types of memory buffers
in Tile-AI programs. It wraps TVM's buffer allocation functionality with convenient
interfaces for different memory scopes.

Available allocation functions:
    - alloc_shared: Allocates shared memory buffers for inter-thread communication
    - alloc_local: Allocates local memory buffers for thread-private storage
    - alloc_fragment: Allocates fragment memory buffers for specialized operations
    - alloc_var: Allocates single-element variable buffers
    - alloc_global: Allocates global memory buffers as workspace

Each function takes shape and dtype parameters and returns a TVM buffer object
with the appropriate memory scope.
"""

from __future__ import annotations
from typing import overload, Literal
from tilelang._typing import DType, ShapeType
from tilelang import tvm as tvm
from tvm.script import tirx as T
from tvm.tirx import PrimExpr
from tvm.tirx.script.builder.ir import sblock_attr
from tvm.tirx.buffer import Buffer
from tvm.tirx.expr import FloatImm, IntImm

from . import dtypes as _dtypes
from .dtypes import dtype as tl_dtype
from .eager.builder import OutTensor
from .proxy import Tensor, ptr as _ptr_sentinel


def alloc_shared(shape: ShapeType, dtype: DType, scope="shared.dyn") -> Buffer:
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
    return T.sblock_alloc_buffer(shape, dtype, scope=scope)


def alloc_local(shape: ShapeType, dtype: DType, scope="local") -> Buffer:
    """Allocate a local memory buffer for thread-private storage.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local"

    Returns:
        T.Buffer: A TVM buffer object allocated in local memory
    """
    return T.sblock_alloc_buffer(shape, dtype, scope=scope)


def alloc_fragment(shape: ShapeType, dtype: DType, scope="local.fragment") -> Buffer:
    """Allocate a fragment memory buffer for specialized operations.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local.fragment"

    Returns:
        T.Buffer: A TVM buffer object allocated in fragment memory
    """
    return T.sblock_alloc_buffer(shape, dtype, scope=scope)


@overload
def alloc_var(dtype: DType, init: PrimExpr | int | float, scope: str = "local.var") -> Buffer: ...


@overload
def alloc_var(dtype: DType, scope: str = "local.var", *, init: PrimExpr | int | float | None = None) -> Buffer: ...


def alloc_var(dtype: DType, *args, scope: str = "local.var", init: PrimExpr | int | float | None = None) -> Buffer:
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

    if dtype is _ptr_sentinel:
        dtype = _dtypes.int64

    buffer = T.sblock_alloc_buffer([1], dtype, scope=parsed_scope)
    if parsed_init is not None:
        # Always use T.buffer_store for reliable initialisation across all
        # backends.  The sblock_attr("tl.local_var_init") path feeds into the
        # flatten_buffer transform which does not reliably emit initialiser
        # code on some backends (e.g. HIP codegen silently drops the
        # annotation for integer/float literals, leaving the scalar
        # uninitialised).  T.buffer_store emits an explicit BufferStore TIR
        # node that every backend lowers to an assignment statement.
        if isinstance(parsed_init, (int, float, IntImm, FloatImm)):
            parsed_init = tl_dtype(dtype)(parsed_init)
        T.buffer_store(buffer, parsed_init, 0)
    return buffer


def alloc_global(shape: ShapeType, dtype: DType, scope="global") -> Buffer:
    """Allocate a global memory buffer as a global workspace.

    NOTE(chaofan): Memory allocated in this way doesn't go through torch allocator. Instead,
    it's allocated directly by the corresponding backend APIs, like cudaMalloc. We
    recommend allocating workspace in Torch side and pass it to the kernel via arguments,
    which is managed under the hood by the framework. This API is mainly for testing
    purposes and some specific purposes.

    NOTE(chaofan): This API may not be available in all backends (e.g. CuteDSL).

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "global"

    Returns:
        T.Buffer: A TVM buffer object allocated in global memory
    """

    return T.sblock_alloc_buffer(shape, dtype, scope=scope)


def alloc_barrier(arrive_count: int | list[int]) -> Buffer:
    """Allocate a barrier buffer.

    Args:
        arrive_count (int | list[int]): The number of threads that need to arrive at each barrier

    Returns:
        T.Buffer: A TVM buffer object allocated as a barrier

    Examples
    --------
    >>> mbar = alloc_barrier(128)  # allocate a barrier with arrive count 128
    >>> mbars = alloc_barrier([128] * n)  # allocate n barriers with the same arrive count 128
    """
    # Normalize to list
    if isinstance(arrive_count, int):
        arrive_count = [arrive_count]
    else:
        arrive_count = list(arrive_count)
    buffer = T.sblock_alloc_buffer((len(arrive_count),), _dtypes.uint64, scope="shared.barrier")
    # Convert to TIR IntImm expressions for C++ pass to consume as Map<Var, Array<PrimExpr>>
    # Use buffer.data as key to support multiple barrier buffer allocations
    arrive_count_exprs = [IntImm("int32", c) for c in arrive_count]
    sblock_attr({"barrier_init": {buffer.data: arrive_count_exprs}})

    return buffer


def alloc_cluster_barrier(arrive_count: int | list[int]) -> Buffer:
    """Allocate a cluster barrier buffer.

    Args:
        arrive_count (int | list[int]): The number of threads that need to arrive at each barrier

    Returns:
        T.Buffer: A TVM buffer object allocated as a cluster barrier
    """
    # Normalize to list
    if isinstance(arrive_count, int):
        arrive_count = [arrive_count]
    else:
        arrive_count = list(arrive_count)
    buffer = T.sblock_alloc_buffer((len(arrive_count),), _dtypes.uint64, scope="shared.cluster_barrier")
    # Convert to TIR IntImm expressions for C++ pass to consume as Map<Var, Array<PrimExpr>>
    # Use buffer.data as key to support multiple barrier buffer allocations
    arrive_count_exprs = [IntImm("int32", c) for c in arrive_count]
    sblock_attr({"barrier_init": {buffer.data: arrive_count_exprs}})

    return buffer


_tmem_alloc_counter = [0]


def alloc_tmem(
    shape: ShapeType,
    dtype: DType,
    *,
    alias: Buffer | None = None,
    col_offset: int = 0,
) -> Buffer:
    """
    Allocate a Tensor Memory (TMEM) buffer for use with 5th generation Tensor Core operations.

    TMEM is a dedicated on-chip memory introduced in Blackwell GPUs (128 lanes × 512
    cols of 32-bit cells, 64KB total).

    Args:
        shape: 2-D logical shape (M, N).
        dtype: Element dtype.
        alias: When supplied, the new buffer SHARES physical TMEM columns with
            ``alias`` — no extra ``tcgen05.alloc`` is issued. Useful when two
            logical TMEM views have non-overlapping lifetimes and can share
            physical columns. The caller owns the lifetime contract;
            tilelang does not verify it.
        col_offset: TMEM column offset from the alias parent's base address.
            Only meaningful with ``alias=...``. Must be a non-negative int.
    """

    assert len(shape) == 2, "shape must be a 2D tensor for TMEM allocation"
    if alias is None and col_offset != 0:
        raise ValueError("col_offset is only valid with alias=...")
    if alias is not None and col_offset < 0:
        raise ValueError(f"col_offset must be >= 0, got {col_offset}")
    buf = T.sblock_alloc_buffer(shape, dtype, scope="shared.tmem")
    if alias is not None:
        # Stash the alias info on the enclosing block. The lower_shared_tmem
        # C++ pass picks this up: it skips the tcgen05.alloc for `buf` and
        # instead emits `buf_addr[0] = alias_addr[0] + col_offset` after the
        # parent's init.
        #
        # The key/value use BUFFER objects (not their data Vars). Vars get
        # replaced by upstream passes (their identity isn't preserved across
        # pass boundaries), but Buffer instances are mutated in-place by the
        # script parser — Namer::Name does `buffer->name = name` rather than
        # creating a new Buffer node — so Buffer identity is stable.
        sblock_attr({
            "tmem_alias_buffers": {
                buf: [alias, IntImm("int32", int(col_offset))]
            }
        })
    return buf


ReducerOp = Literal["sum", "max", "min"]


def alloc_reducer(shape: ShapeType, dtype: DType, op: ReducerOp = "sum", replication=None) -> Buffer:
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

    reducer = T.sblock_alloc_buffer(shape, dtype, scope="local.fragment")
    sblock_attr({"reducer_info": {reducer.data: {"rep": replication, "op": op}}})

    return reducer


DescKind = Literal["wgmma", "tcgen05_smem", "tcgen05_instr"]


def alloc_descriptor(
    kind: DescKind = "wgmma",
    dtype: DType = _dtypes.uint64,
) -> Buffer:
    """Allocate a descriptor buffer for WGMMA and TCGEN5.MMA.

    Args:
        kind: The descriptor kind, one of "wgmma", "tcgen05" ("utcmma" as alias).

    Returns:
        T.Buffer: A TVM buffer object allocated as a descriptor
    """

    scope = "local.descriptor." + kind
    # Buffer naming via `name` is not supported by this TVM builder signature;
    # keep parameter for forward-compat, but do not pass it.
    return T.sblock_alloc_buffer([1], dtype, scope=scope)


def alloc_wgmma_desc(dtype: DType = _dtypes.uint64) -> Buffer:
    return alloc_descriptor("wgmma", dtype=dtype)


def alloc_tcgen05_smem_desc(dtype: DType = _dtypes.uint64) -> Buffer:
    return alloc_descriptor("tcgen05_smem", dtype=dtype)


def alloc_tcgen05_instruction_desc(dtype: DType = _dtypes.uint32) -> Buffer:
    return alloc_descriptor("tcgen05_instr", dtype=dtype)


# Alias: short name consistent with imports
def alloc_tcgen05_instr_desc(dtype: DType = _dtypes.uint32) -> Buffer:
    return alloc_tcgen05_instruction_desc(dtype)


@overload
def empty(shape, dtype: DType = _dtypes.float32) -> Tensor: ...


def empty(*shape, dtype: DType = _dtypes.float32) -> Tensor:
    """Declare the output tensor used in eager-style JIT.

    Tensors allocated in this way should be returned as the output of the function.

    Args:
        shape (tuple): The shape of the tensor to allocate
        dtype (str): The data type of the tensor (e.g., 'float32', 'int32')

    Returns:
        Tensor: The declared OutTensor object.
    """

    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return OutTensor(shape[0], dtype)
    elif len(shape) == 2 and isinstance(shape[0], (tuple, list)) and isinstance(shape[1], str):
        return OutTensor(shape[0], shape[1])
    elif all([isinstance(x, (int, PrimExpr)) for x in shape]):
        return OutTensor(shape, dtype)
    else:
        raise TypeError(f"Invalid shape {shape}")
