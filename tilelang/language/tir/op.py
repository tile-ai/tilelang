from __future__ import annotations
from typing import Any
import tvm
from tvm.ir import PrimExpr
from tvm.ir.base import Span
from tvm.runtime import const
from tvm.tir.expr import IntImm, PrimExprWithOp
import tvm.tir.op as _tvm_op


def call_packed(*args, span=None):
    """Build expression by call an external packed function.

    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is Buffer, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    return _tvm_op.call_packed(*args, span=span)


def call_cpacked(*args, span=None):
    """Build expression by call an external packed function.

    Same as call_packed, except that the first argument is the function name
    (as in call_extern), and the last argument is the resource handle.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    return _tvm_op.call_cpacked(*args, span=span)


def call_packed_lowered(*args, span=None):
    """Lowered version of call packed.
    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.
    When the argument is Buffer, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    return _tvm_op.call_packed_lowered(*args, span=span)


def call_cpacked_lowered(*args, span=None):
    """Lowered version of call c-packed.
    Same as call_packed, except that the first argument is the function name
    (as in call_extern), and the last argument is the resource handle.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    return _tvm_op.call_cpacked_lowered(*args, span=span)


def call_intrin(dtype, func_name, *args, span=None):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.call_intrin(dtype, func_name, *args, span=span)


def call_pure_extern(dtype, func_name, *args, span=None):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.call_pure_extern(dtype, func_name, *args, span=span)


def call_extern(dtype, func_name, *args, span=None):
    """Build expression by calling a extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.call_extern(dtype, func_name, *args, span=span)


def call_llvm_intrin(dtype, name, *args, span=None):
    """Build expression by calling a llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.call_llvm_intrin(dtype, name, *args, span=span)


def call_llvm_pure_intrin(dtype, name, *args, span=None):
    """Build expression by calling a pure llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.call_llvm_pure_intrin(dtype, name, *args, span=span)


def tvm_check_return(expected, return_unexpected, nested_call):
    """Return new on stack dtype[num]
    Parameters
    ----------
    expected : int
        The expected return code.
    return_unexpected : int
        The unexpected return code.
    nested_call : PrimExpr
        The call expression to check return.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_check_return(expected, return_unexpected, nested_call)


def tvm_stack_alloca(dtype_str, num):
    """Return new on stack dtype[num]

    Parameters
    ----------
    dtype_str : str
        The data type of array.

    num : int
        The size of array.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_stack_alloca(dtype_str, num)


def tvm_stack_make_shape(*args):
    """Allocate a shape tuple on stack, return the handle

    Parameters
    ----------
    args : int
        The tuple shape.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_stack_make_shape(*args)


def tvm_stack_make_array(data, shape, strides, ndim, arr_dtype, elem_offset):
    """Allocate a NDArray(DLTensor) on stack, return the handle

    Parameters
    ----------
    data : Expr
        The data of array.

    shape : Expr
        The shape of array.

    strides : Expr
        The strides of array.

    ndim : Expr
        The dimensions of array.

    arr_dtype : Expr
        The data type of array.

    elem_offse : Expr
        The element offset of array.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_stack_make_array(data, shape, strides, ndim, arr_dtype, elem_offset)


def assume(cond=None):
    """Provide a true statement that can be used for simplifications

    Parameters
    ----------
    cond : Expr
       The constraint condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.assume(cond)


def undef():
    """Returns an initialized but arbitrary value

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.undef()


def call_tir(global_var: tvm.ir.GlobalVar, *args):
    """Performs a call into another PrimFunc in the same IRModule

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.call_tir(global_var, *args)


def start_profile_intrinsic(id):
    """Start profile intrinsic.
    Parameters
    ----------
    id : int
        The intrinsic id.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.start_profile_intrinsic(id)


def end_profile_intrinsic(id):
    """End profile intrinsic.
    Parameters
    ----------
    id : int
        The intrinsic id.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.end_profile_intrinsic(id)


def tvm_tuple(*value):
    """Create a tuple structure in value field of AttrStmt

    Parameters
    ----------
    value : Expr
        The value in tuple.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_tuple(*value)


def tvm_struct_get(arr, index, field, dtype):
    """Get struct field value in array

    Parameters
    ----------
    dtype : str
        The date type of the result.

    arr : StructType*
        The array of struct.

    index : int
        The index of struct.

    field : int
        The field of struct.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_struct_get(arr, index, field, dtype)


def tvm_struct_set(arr, index, field, value):
    """Set value in struct field in array

    Parameters
    ----------
    arr : StructType*
        The array of struct.

    index : int
        The index of struct.

    field : int
        The field of struct.

    value : Expr
        The value to be set in field.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_struct_set(arr, index, field, value)


def address_of(buffer_load, span=None):
    """Returns the address of an element in the buffer

    Parameters
    ----------
    buffer_load: BufferLoad
        The buffer load.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.address_of(buffer_load, span=span)


def lookup_param(param_name, span=None):
    """Returns the param by name

    Parameters
    ----------
    param_name : str
        The name of param.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.lookup_param(param_name, span=span)


def tvm_thread_allreduce(*freduce_args):
    """Perform allreduce inside threadblock.

    Parameters
    ----------
    freduce_args : Expr
        The args.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_thread_allreduce(*freduce_args)


def tvm_thread_invariant(cond):
    """Mark condition as thread invariant.

    Parameters
    ----------
    cond : Expr
        The condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_thread_invariant(cond)


def tvm_storage_sync(storage_scope):
    """Perform synchronization in specified scope.

    Parameters
    ----------
    storage_scope : str
        The storage scope to perform synchronization.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_storage_sync(storage_scope)


def tvm_warp_shuffle(mask, value, warp_id, width, warp_size):
    """Exchange value between threads inside a warp.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    warp_id : PrimExpr
        The source lane index to fetch value.
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_warp_shuffle(mask, value, warp_id, width, warp_size)


def tvm_warp_shuffle_up(mask, value, offset, width, warp_size):
    """Copy value from a lane with lower (by offset) index relative to caller.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    offset : PrimExpr
        The difference between source lane index and destination lane index:
        `offset = dst_lane_idx - src_lane_idx`
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_warp_shuffle_up(mask, value, offset, width, warp_size)


def tvm_warp_shuffle_down(mask, value, offset, width, warp_size):
    """Copy value from a lane with higher (by offset) index relative to caller.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    offset : PrimExpr
        The difference between source lane index and destination lane index:
        `offset = src_lane_idx - dst_lane_idx`
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_warp_shuffle_down(mask, value, offset, width, warp_size)


def tvm_warp_activemask():
    """Return a 32-bit mask indicates currently active threads in a calling warp.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_warp_activemask()


def type_annotation(dtype):
    """Create a type annotation expression

    Parameters
    ----------
    dtype : Expr
        The data type.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.type_annotation(dtype)


def tvm_access_ptr(ptype, data, offset, extent, rw_mask):
    """Get head access address with memory access pattern info

    Parameters
    ----------
    ptype : Expr
        The data type of pointer.

    data : DType*
        The data of pointer.

    offset : int
        The offset of pointer.

    extent : int
        The extent of pointer.

    rw_mask : int
        The read write mask.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_access_ptr(ptype, data, offset, extent, rw_mask)


def tvm_throw_last_error():
    """Throw TVMGetLastError()

    Returns
    -------
    ret : PrimExpr
        The return expression
    """
    return _tvm_op.tvm_throw_last_error()


def tvm_load_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout):
    """TVM intrinsic for tensor core load operators

    Parameters
    ----------
    fragment : Var
        The wmma fragment.

    m : UIntImm
        The shape of wmma fragment.

    n : UIntImm
        The shape of wmma fragment.

    k : UIntImm
        The shape of wmma fragment.

    index : Expr
        The fragment index.

    buffer_ptr : Expr
        The fragment buffer pointer.

    stride : Expr
        The fragment stride.

    layout : Literal["row_major", "column_major"]
        The fragment layout.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_load_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout)


def tvm_mma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c):
    """TVM intrinsic for tensor core mma_sync operators

    Parameters
    ----------
    fragment_d : Var
        The wmma fragment_d.

    index_d : Expr
        The fragment_d index.

    fragment_a : Var
        The wmma fragment_a.

    index_a : Expr
        The fragment_a index.

    fragment_b : Var
        The wmma fragment_b.

    index_b : Expr
        The fragment_b index.

    fragment_c : Var
        The wmma fragment_c.

    index_c : Expr
        The fragment_c index.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_mma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c)


def tvm_bmma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c):
    """TVM intrinsic for tensor core bmma_sync operators

    Parameters
    ----------
    fragment_d : Var
        The bwmma fragment_d.

    index_d : Expr
        The fragment_d index.

    fragment_a : Var
        The bwmma fragment_a.

    index_a : Expr
        The fragment_a index.

    fragment_b : Var
        The bwmma fragment_b.

    index_b : Expr
        The fragment_b index.

    fragment_c : Var
        The bwmma fragment_c.

    index_c : Expr
        The fragment_c index.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_bmma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c)


def tvm_fill_fragment(fragment, m, n, k, index, value):
    """TVM intrinsic for tensor core fill_fragment operators

    Parameters
    ----------
    fragment : Var
        The wmma fragment

    m : UIntImm
        The shape of wmma fragment.

    n : UIntImm
        The shape of wmma fragment.

    k : UIntImm
        The shape of wmma fragment.

    index : Expr
        The fragment index.

    value : Expr
        The value to be filled in fragment.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_fill_fragment(fragment, m, n, k, index, value)


def tvm_store_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout):
    """TVM intrinsic for tensor core store operators

    Parameters
    ----------
    fragment : Var
        The wmma fragment.

    m : UIntImm
        The shape of wmma fragment.

    n : UIntImm
        The shape of wmma fragment.

    k : UIntImm
        The shape of wmma fragment.

    index : Expr
        The fragment index.

    buffer_ptr : Expr
        The fragment buffer pointer.

    stride : Expr
        The fragment stride.

    layout : Literal["row_major", "column_major"]
        The fragment layout.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.tvm_store_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout)


def ptx_mma(
    dtype,
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
    saturate,
    operator=None,
):
    """TVM intrinsic for ptx tensor core mma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment A.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    saturate : bool
        The optional saturation at the output.

    operator : Optional[Literal["xor", "and"]]
        The 1-bit operator.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_mma(
        dtype,
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
        saturate,
        operator,
    )


def ptx_mma_sp(
    dtype,
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
    metadata,
    meta_index,
    sparse_selector,
    saturate,
):
    """TVM intrinsic for sparse tensor core ptx instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

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

    metadata : Expr
        The metadata of operand.

    meta_index : Expr
        The metadata index of operand.

    sparse_selector : Expr
        The sparse selector indicating the thread that stores the metadata.

    saturate : bool
        The optional saturation at the output.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_mma_sp(
        dtype,
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
        metadata,
        meta_index,
        sparse_selector,
        saturate,
    )


def ptx_wgmma_ss(
    dtype,
    wgmma_prefix,
    a_is_k_major,
    b_is_k_major,
    a_dtype_abbrv,
    b_dtype_abbrv,
    accum_dtype_abbrv,
    A_desc,
    A_offset,
    B_desc,
    B_offset,
    C_data,
    C_offset,
    scale_out,
    scale_in_a,
    scale_in_b,
):
    """TVM intrinsic for ptx tensor core wmma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-wmma
    """
    return call_intrin(
        dtype,
        _tvm_op.Op.get("tl.ptx_wgmma_ss"),
        wgmma_prefix,
        a_is_k_major,
        b_is_k_major,
        a_dtype_abbrv,
        b_dtype_abbrv,
        accum_dtype_abbrv,
        A_desc,
        A_offset,
        B_desc,
        B_offset,
        C_data,
        C_offset,
        scale_out,
        scale_in_a,
        scale_in_b,
    )


def ptx_wgmma_rs(
    dtype,
    wgmma_prefix,
    b_is_k_major,
    a_dtype_abbrv,
    b_dtype_abbrv,
    accum_dtype_abbrv,
    A_buf,
    A_offset,
    B_desc,
    B_offset,
    C_data,
    C_offset,
    scale_out,
    scale_in_a,
    scale_in_b,
):
    return call_intrin(
        dtype,
        _tvm_op.Op.get("tl.ptx_wgmma_rs"),
        wgmma_prefix,
        b_is_k_major,
        a_dtype_abbrv,
        b_dtype_abbrv,
        accum_dtype_abbrv,
        A_buf,
        A_offset,
        B_desc,
        B_offset,
        C_data,
        C_offset,
        scale_out,
        scale_in_a,
        scale_in_b,
    )


def ptx_tcgen05_mma_ss(
    kind_dtype,
    desc_a,
    A_offset,
    desc_b,
    B_offset,
    C_ptr,
    C_offset,
    desc_val,
    scale_out,
    mask0,
    mask1,
    mask2,
    mask3,
    enable_ws=False,
    ws=None,
    warp_specialized=None,
    variant=None,
):
    """TVM intrinsic for tcgen05.mma shared-memory × shared-memory instructions.

    Expects 13 or 14 positional arguments:
    (kind_dtype, desc_a, A_offset, desc_b, B_offset, C_ptr, C_offset,
     desc_val, scale_out, mask0, mask1, mask2, mask3[, enable_ws]).
    Aliases: you can also pass `ws` or `warp_specialized` (booleans) instead of `enable_ws`.
    Alternatively, use `variant="ws"` (or "default").
    - kind_dtype: instruction kind selector (e.g., T.float16 for kind::f16,
      "tf32" for kind::tf32, "int8" for kind::i8, "float8_e4m3" for kind::f8f6f4).
    """
    # Aliases precedence: if either `ws` or `warp_specialized` is provided, they override enable_ws
    if ws is not None:
        enable_ws = bool(ws)
    if warp_specialized is not None:
        enable_ws = bool(warp_specialized)
    if variant is not None:
        if isinstance(variant, str):
            v = variant.lower()
            if v in ("ws", "warp_specialized", "warp-specialized"):
                enable_ws = True
            elif v in ("default", "std", "ss"):
                enable_ws = False
            else:
                raise ValueError(f"ptx_tcgen05_mma_ss: unknown variant: {variant}")
        else:
            # Treat non-string as truthy flag
            enable_ws = bool(variant)

    return call_intrin(
        "handle",
        _tvm_op.Op.get("tl.ptx_tcgen05_mma_ss"),
        kind_dtype,
        desc_a,
        A_offset,
        desc_b,
        B_offset,
        C_ptr,
        C_offset,
        desc_val,
        scale_out,
        mask0,
        mask1,
        mask2,
        mask3,
        enable_ws,
    )


def ptx_tcgen05_mma_ts(
    kind_dtype,
    A_ptr,
    A_offset,
    desc_b,
    B_offset,
    C_ptr,
    C_offset,
    desc_val,
    scale_out,
    mask0,
    mask1,
    mask2,
    mask3,
):
    """TVM intrinsic for tcgen05.mma tensor-memory × shared-memory instructions.

    Expects 13 positional arguments:
    (kind_dtype, A_ptr, A_offset, desc_b, B_offset, C_ptr, C_offset,
     desc_val, scale_out, mask0, mask1, mask2, mask3).
    - kind_dtype: instruction kind selector (e.g., T.float16 for kind::f16,
      "tf32" for kind::tf32, "int8" for kind::i8, "float8_e4m3" for kind::f8f6f4).
    """
    return call_intrin(
        "handle",
        _tvm_op.Op.get("tl.ptx_tcgen05_mma_ts"),
        kind_dtype,
        A_ptr,
        A_offset,
        desc_b,
        B_offset,
        C_ptr,
        C_offset,
        desc_val,
        scale_out,
        mask0,
        mask1,
        mask2,
        mask3,
    )


def mma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """TVM intrinsic for storing the result of PTX MMA into a destination pointer

    Parameters
    ----------
    dtype : str
        The data type of the result.

    m : IntImm
        The shape of mma fragment.

    n : IntImm
        The shape of mma fragment.

    dst_ptr : Var
        The destination pointer variable.

    src_ptr : Var
        The source pointer variable.

    src_offset : Expr
        The source offset.

    dst_stride : Var
        The destination stride.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.mma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride)


def mma_fill(dtype, local_size, local_ptr, offset):
    """TVM intrinsic for zero-initalizing an MMA accumulation register

    Parameters
    ----------
    dtype : str
        The data type of the result.

    local_size : IntImm
        The number of elements.

    local_ptr : Var
        The destination pointer variable.

    offset : Expr
        The destination offset.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.mma_fill(dtype, local_size, local_ptr, offset)


def ptx_ldmatrix(dtype, trans, num, type, local_ptr, local_offset, smem_ptr, smem_offset):
    """TVM intrinsic for ptx load matrix from shared memory
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Parameters
    ----------
    dtype : str
       The data type of the result.

    trans : bool
        The matrix is loaded in column-major format.

    num : IntImm
        The number of matrices.

    type : Literal[".b16"]
        The data type of the matrices.

    local_ptr : Var
        The local pointer variable.

    local_offset : Expr
        The offset of local pointer.

    smem_ptr : Var
        The shared memory pointer variable.

    smem_offset : Expr
        The offset of shared memort pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_ldmatrix(dtype, trans, num, type, local_ptr, local_offset, smem_ptr, smem_offset)


def ptx_cp_async(dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Parameters
    ----------
    dtype : str
       The data type of the result.

    shared_ptr : Var
        The shared memory pointer variable.

    shared_offset : Expr
        The offset of shared memory pointer.

    global_ptr : Var
        The global memory pointer variable.

    global_offset : Expr
        The offset of global memory pointer.

    bytes : int
        The data size to copy.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_cp_async(dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes)


def ptx_cp_async_bulk(dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes, barrier_id):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async.bulk
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

    Parameters
    ----------
    dtype : str
       The data type of the result.

    shared_ptr : Var
        The shared memory pointer variable.

    shared_offset : Expr
        The offset of shared memory pointer.

    global_ptr : Var
        The global memory pointer variable.

    global_offset : Expr
        The offset of global memory pointer.

    bytes : int
        The data size to copy.

    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_cp_async_bulk(dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes, barrier_id)


def ptx_commit_group():
    """TVM intrinsic for ptx async copy commit
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_commit_group()


def ptx_wait_group(num):
    """TVM intrinsic for ptx async copy wait
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group

    Parameters
    ----------
    num : int
        The number of the most recent uncommitted pending cp.async groups to wait.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_wait_group(num)


def tvm_mfma(
    dtype,
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
    """TVM intrinsic for amd matrix core mfma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment A.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_mfma"),
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


def tvm_mfma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """TVM intrinsic for storing the result of PTX MMA into a destination pointer

    Parameters
    ----------
    dtype : str
        The data type of the result.

    m : IntImm
        The shape of mma fragment.

    n : IntImm
        The shape of mma fragment.

    dst_ptr : Var
        The destination pointer variable.

    src_ptr : Var
        The source pointer variable.

    src_offset : Expr
        The source offset.

    dst_stride : Var
        The destination stride.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_mfma_store"),
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
    )


def tvm_rdna_wmma(
    dtype,
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
    """TVM intrinsic for amd matrix core mfma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment A.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_rdna_wmma"),
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


def tvm_rdna_wmma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """TVM intrinsic for storing the result of PTX MMA into a destination pointer

    Parameters
    ----------
    dtype : str
        The data type of the result.

    m : IntImm
        The shape of mma fragment.

    n : IntImm
        The shape of mma fragment.

    dst_ptr : Var
        The destination pointer variable.

    src_ptr : Var
        The source pointer variable.

    src_offset : Expr
        The source offset.

    dst_stride : Var
        The destination stride.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_rdna_wmma_store"),
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
    )


def ptx_cp_async_barrier(barrier_id):
    """TVM intrinsic for ptx async copy barrier using cp.async.mbarrier.arrive
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_cp_async_barrier(barrier_id)


def ptx_init_barrier_thread_count(barrier_id, thread_count):
    """TVM intrinsic for ptx barrier initialization of thread count using mbarrier.init
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    thread_count : int
        Number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_init_barrier_thread_count(barrier_id, thread_count)


def ptx_arrive_barrier(barrier_id):
    """TVM intrinsic for ptx barrier arrival using mbarrier.arrive
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_arrive_barrier(barrier_id)


def ptx_arrive_barrier_expect_tx(barrier_id, byte_count):
    """TVM intrinsic for ptx barrier arrival with expect tx using mbarrier.arrive.expect_tx
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    byte_count : int
        Increases the tx count of the mbarrier object to track completion of
        additional async transactions.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_arrive_barrier_expect_tx(barrier_id, byte_count)


def ptx_wait_barrier(barrier_id):
    """TVM intrinsic for ptx barrier wait using mbarrier.try_wait
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_wait_barrier(barrier_id)


def create_barriers(barrier_count):
    """TVM intrinsic to create N barriers

    Parameters
    ----------
    barrier_count : int
        The number of barriers to create.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.create_barriers(barrier_count)


def vectorlow(dtype, vec):
    """Get the low level half of the vector

    Parameters
    ----------
    dtype : str
       The data type of the result.

    vec : list
       The input vector.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.vectorlow(dtype, vec)


def vectorhigh(dtype, vec):
    """Get the high level half of the vector

    Parameters
    ----------
    dtype : str
       The data type of the result.

    vec : list
       The input vector.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.vectorhigh(dtype, vec)


def vectorcombine(dtype, vec1, vec2):
    """Concat two vectors

    Parameters
    ----------
    vec1 : list
       The input vector.

    vec2 : list
       The input vector.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.vectorcombine(dtype, vec1, vec2)


def ret(val):
    """Create a tir return expression

    Parameters
    ----------
    val : Expr
        The returned tir expression, whose data type is int, float or void pointer.

    Returns
    -------
    ret : PrimExpr
        The return expression
    """
    return _tvm_op.ret(val)


def any(*args, span=None):
    """Create a new expression of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    return _tvm_op.any(*args, span=span)


def all(*args, span=None):
    """Create a new expression of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    return _tvm_op.all(*args, span=span)


def trace(args, trace_action="tvm.default_trace_action"):
    """Trace tensor data at the runtime.

    The trace function allows to trace specific tensor at the
    runtime. The tracing value should come as last argument.
    The trace action should be specified, by default
    tvm.default_trace_action is used.

    Parameters
    ----------
    args : list of Expr or Buffers.
        Positional arguments.

    trace_action : str.
        The name of the trace action.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    tvm.tir.call_packed : Creates packed function.
    """
    return _tvm_op.trace(args, trace_action)


def min_value(dtype, span=None):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The minimum value of dtype.
    """
    return _tvm_op.min_value(dtype, span)


def max_value(dtype: str, span: Span | None = None) -> Any:
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The maximum value of dtype.
    """
    return _tvm_op.max_value(dtype, span)


def infinity(dtype: str, span: Span | None = None) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The infinity value of dtype.
    """
    return call_intrin(dtype, _tvm_op.Op.get("tl.infinity"), dtype, span=span)


def reinterpret(dtype, value, span: Span | None = None) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    value : PrimExpr
        The input value.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The reinterpret cast value of dtype.
    """
    return _tvm_op.reinterpret(dtype, value, span)


def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.exp(x)


def exp2(x):
    """Calculate 2**x

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.exp2(x)


def exp10(x):
    """Calculate 10**x

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.exp10(x)


def erf(x):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.erf(x)


def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.tanh(x)


def sigmoid(x):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.sigmoid(x)


def log(x):
    """Take log of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.log(x)


def log2(x):
    """Take log2 of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.log2(x)


def log10(x):
    """Take log10 of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.log10(x)


def log1p(x):
    """Take log(x + 1) with respect to input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.log1p(x)


def tan(x):
    """Take tan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.tan(x)


def cos(x):
    """Take cos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.cos(x)


def cosh(x):
    """Take cosh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.cosh(x)


def acos(x):
    """Take acos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.acos(x)


def acosh(x):
    """Take acos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.acosh(x)


def sin(x):
    """Take sin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.sin(x)


def sinh(x):
    """Take sinh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.sinh(x)


def asin(x):
    """Take asin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.asin(x)


def asinh(x):
    """Take asinh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.asinh(x)


def atan(x):
    """Take atan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.atan(x)


def atanh(x):
    """Take atanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.atanh(x)


def atan2(x1, x2):
    """Take arctan2(x1, x2).

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.atan2(x1, x2)


def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.sqrt(x)


def rsqrt(x):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.rsqrt(x)


def clz(x):
    """Count leading zero bits of an integer x.

    Parameters
    ----------
    x : PrimExpr
        Input 32 or 64 bit integer.
        The result is undefined if the input is 0.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.clz(x)


def floor(x: PrimExprWithOp, span=None):
    """Take floor of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.floor(x, span)


def ceil(x, span=None):
    """Take ceil of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.ceil(x, span)


def trunc(x, span=None):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.trunc(x, span)


def abs(x, span=None):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.abs(x, span)


def bitwise_and(x, y, span=None):
    """Take bitwise and of two values

    Parameters
    ----------
    x : PrimExpr
        Left operand

    y : PrimExpr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _tvm_op.bitwise_and(x, y, span)


def bitwise_not(x, span=None):
    """Take bitwise not of input value

    Parameters
    ----------
    x : PrimExpr
        Input operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _tvm_op.bitwise_not(x, span)


def bitwise_or(x, y, span=None):
    """Take bitwise or of two values

    Parameters
    ----------
    x : PrimExpr
        Left operand

    y : PrimExpr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _tvm_op.bitwise_or(x, y, span)


def bitwise_xor(x, y, span=None):
    """Take bitwise xor of two values

    Parameters
    ----------
    x : PrimExpr
        Left operand

    y : PrimExpr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _tvm_op.bitwise_xor(x, y, span)


def round(x, span=None):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.round(x, span)


def nearbyint(x, span=None):
    """Round elements of the array to the nearest integer.
    This intrinsic uses llvm.nearbyint instead of llvm.round
    which is faster but will results different from te.round.
    Notably nearbyint rounds according to the rounding mode,
    whereas te.round (llvm.round) ignores that.
    For differences between the two see:
    https://en.cppreference.com/w/cpp/numeric/math/round
    https://en.cppreference.com/w/cpp/numeric/math/nearbyint

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.nearbyint(x, span)


def nextafter(x1, x2):
    """Return the next floating-point value after x1 towards x2.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.nextafter(x1, x2)


def hypot(x1, x2):
    """Equivalent to sqrt(x1**2 + x2**2), element-wise.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.hypot(x1, x2)


def copysign(x1, x2):
    """Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.copysign(x1, x2)


def ldexp(x1, x2):
    """Returns x1 * (2 ** x2).

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.ldexp(x1, x2)


def likely(cond, span=None):
    """Mark condition as likely.

    Parameters
    ----------

    cond : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The marked expression.
    """
    return _tvm_op.likely(cond, span)


def isnan(x, span=None):
    """Check if input value is Nan.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.isnan(x, span)


def isnullptr(x, span=None):
    """Check if input value is nullptr.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.isnullptr(x, span)


def isfinite(x, span=None):
    """Check if input value is finite.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.isfinite(x, span)


def isinf(x, span=None):
    """Check if input value is infinite.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.isinf(x, span)


def pow_of_int(x: PrimExpr, y: int) -> PrimExpr:
    """Fast power operation than pow(float, float).

    Args:
        x (PrimExpr): Base value
        y (int): Exponent value
    """
    return call_intrin(
        x.dtype,
        tvm.tir.op.Op.get("tl.pow_of_int"),
        x,
        y,
    )


def power(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    if isinstance(y, (int, IntImm)):
        return pow_of_int(x, y)
    return _tvm_op.power(x, y, span)


def pow(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    if isinstance(y, (int, IntImm)):
        return pow_of_int(x, y)
    return _tvm_op.pow(x, y, span)


def popcount(x):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.popcount(x)


def q_multiply_shift(x, y, q, s):
    """Execute a multiplication between two Q-numbers x and y
    followed by a right shift s. The mathematical expression is:

       out = round(x*y*2^-s)

    More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
    The rounding rule is to the nearest value, rounding half up
    (i.e., round(x.1) = x and round (x.5) = x+1)

    Parameters
    ----------
    x : PrimExpr
        First Q-number
    y : PrimExpr
        Second Q-number
    q : PrimExpr
        Number of fractional bits in x and y. Needs to be > 0
    s : PrimExpr
        Integer shift

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _tvm_op.q_multiply_shift(x, y, q, s)


def q_multiply_shift_per_axis(
    x: PrimExpr,
    y: PrimExpr,
    ls: PrimExpr,
    rs: PrimExpr,
    q: IntImm,
    is_lshift_required: IntImm,
    is_rshift_required: IntImm,
):
    """Execute a multiplication between two Q-numbers x and y

    Parameters
    ----------
    x : PrimExpr
        First Q-number.
    y : PrimExpr
        Second Q-number.
    ls : PrimExpr
         Integer left shift.
    rs : PrimExpr
         Integer right shift.
    q : IntImm
        Number of fractional bits in x and y. Needs to be > 0.
    is_lshift_required : IntImm
                         Whether we need to do left shift or not.
    is_rshift_required : IntImm
                         Whether we need to do right shift or not.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _tvm_op.q_multiply_shift_per_axis(x, y, ls, rs, q, is_lshift_required, is_rshift_required)


def shift_left(x, y, span=None):
    """Return the result of x left shifted by y bits.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _tvm_op.shift_left(x, y, span)


def shift_right(x, y, span=None):
    """Return the result of x right shifted by y bits.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _tvm_op.shift_right(x, y, span)


def fmod(x, y):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.
    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _tvm_op.fmod(x, y)


def if_then_else(cond, t, f, span=None):
    """Conditional selection expression.

    Parameters
    ----------
    cond : PrimExpr
        The condition

    t : PrimExpr
        The result expression if cond is true.

    f : PrimExpr
        The result expression if cond is false.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    result : Node
        The result of conditional expression.

    Note
    ----
    Unlike Select, if_then_else will not execute
    the branch that does not satisfy the condition.
    You can use it to guard against out of bound access.
    Unlike Select, if_then_else cannot be vectorized
    if some lanes in the vector have different conditions.
    """
    return _tvm_op.if_then_else(cond, t, f, span)


def div(a, b, span=None):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b, span).
    """
    return _tvm_op.div(a, b, span)


def indexdiv(a, b, span=None):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _tvm_op.indexdiv(a, b, span)


def indexmod(a, b, span=None):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _tvm_op.indexmod(a, b, span)


def truncdiv(a, b, span=None):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _tvm_op.truncdiv(a, b, span)


def truncmod(a, b, span=None):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _tvm_op.truncmod(a, b, span)


def floordiv(a, b, span=None):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _tvm_op.floordiv(a, b, span)


def floormod(a, b, span=None):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _tvm_op.floormod(a, b, span)


def ceildiv(lhs, rhs, span=None):
    """Generic ceildiv operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of ceildiv operation.
    """
    return _tvm_op.ceildiv(lhs, rhs, span)


def comm_reducer(fcombine, fidentity, name="reduce"):
    """Create a commutative reducer for reduction.

    Parameters
    ----------
    fcombine : function(Expr -> Expr -> Expr)
        A binary function which takes two Expr as input to return a Expr.

    fidentity : function(str -> Expr)
        A function which takes a type string as input to return a const Expr.

    Returns
    -------
    reducer : function
        A function which creates a reduce expression over axis.
        There are two ways to use it:

        1. accept (expr, axis, where) to produce an Reduce Expr on
           specified axis;
        2. simply use it with multiple Exprs.

    Example
    -------
    .. code-block:: python

        n = te.var("n")
        m = te.var("m")
        mysum = te.comm_reducer(lambda x, y: x+y,
            lambda t: tvm.tir.const(0, dtype=t), name="mysum")
        A = te.placeholder((n, m), name="A")
        k = te.reduce_axis((0, m), name="k")
        B = te.compute((n,), lambda i: mysum(A[i, k], axis=k), name="B")
    """
    return _tvm_op.comm_reducer(fcombine, fidentity, name)


def TVMBackendAllocWorkspace(device_type, device_id, nbytes, dtype_code_hint, dtype_bits_hint):
    """Backend function to allocate temporal workspace

    Parameters
    ----------
    device_type : int
        The device type which the space will be allocated.

    device_id : int
        The device id which the space will be allocated.

    nbytes : int
        The size of the space requested.

    dtype_code_hint : int
        The type code of the array elements. Only used in certain backends such as OpenGL.

    dtype_bits_hint : int
        The type bits of the array elements. Only used in certain backends such as OpenGL.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.TVMBackendAllocWorkspace(device_type, device_id, nbytes, dtype_code_hint, dtype_bits_hint)


def TVMBackendFreeWorkspace(device_type, device_id, ptr):
    """Backend function to free temporal workspace.

    Parameters
    ----------
    device_type : int
        The device type which the space will be allocated.

    device_id : int
        The device id which the space will be allocated.

    ptr : Var
        The result allocated space pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.TVMBackendFreeWorkspace(device_type, device_id, ptr)


def anylist_getitem(list_handle, index):
    """Returns an item from any list.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.anylist_getitem(list_handle, index)


def anylist_resetitem(list_handle, index):
    """Reset an item from any list.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.anylist_resetitem(list_handle, index)


def anylist_setitem_call_packed(list_handle, index, func_name, *args):
    """Set anylist item by result of packed call.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    func_name: str
        The name of the function to be called.
    args:
        Extra arguments
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.anylist_setitem_call_packed(list_handle, index, func_name, *args)


def anylist_setitem_call_cpacked(list_handle, index, func_name, *args):
    """Set anylist item by result of packed call.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    func_name: str
        The name of the function to be called.
    args:
        Extra arguments
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.anylist_setitem_call_cpacked(list_handle, index, func_name, *args)


def vscale():
    """Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
    (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
    Returns
    -------
    call : PrimExpr
        Call to the vscale intrinsic
    """
    return _tvm_op.vscale()


# pylint: disable=unnecessary-lambda
sum = comm_reducer(lambda x, y: x + y, lambda t: const(0, dtype=t), name="sum")
min = comm_reducer(lambda x, y: _tvm_op._OpMin(x, y, None), max_value, name="min")
max = comm_reducer(lambda x, y: _tvm_op._OpMax(x, y, None), min_value, name="max")
