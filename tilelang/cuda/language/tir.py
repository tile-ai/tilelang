"""CUDA-specific low-level TIR language operators."""

from __future__ import annotations

import tvm
import tvm.tirx.op as _tvm_op
from tvm.tirx.expr import IntImm

from tilelang.language.tir.exports import CUDA_ONLY_TIR_EXPORTS, SHARED_LEGACY_TIR_EXPORTS
from tilelang.language.tir.ir import (
    _dtype_forward,
    _op_wrapper,
    ptx_arrive_barrier as ptx_arrive_barrier,
    ptx_arrive_barrier_expect_tx as ptx_arrive_barrier_expect_tx,
    ptx_commit_group as ptx_commit_group,
    ptx_cp_async as ptx_cp_async,
    ptx_cp_async_barrier as ptx_cp_async_barrier,
    ptx_init_barrier_thread_count as ptx_init_barrier_thread_count,
    ptx_wait_group as ptx_wait_group,
)
from tilelang.language.tir.op import call_intrin as _call_intrin


# Unchanged upstream CUDA operators are rebound directly in the CUDA dialect.
mma_store = _dtype_forward(_tvm_op.mma_store)
ptx_cp_async_bulk = _dtype_forward(_tvm_op.ptx_cp_async_bulk)
ptx_mma = _dtype_forward(_tvm_op.ptx_mma)
ptx_wait_barrier = _op_wrapper(_tvm_op.ptx_wait_barrier)


@_dtype_forward
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


@_dtype_forward
def ptx_mma_block_scale(
    accum_dtype,
    shape,
    A_layout,
    B_layout,
    kind,
    scale_vec_size,
    A_dtype,
    B_dtype,
    scale_type,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
    scale_a,
    scale_b,
    scale_a_byte_id=0,
    scale_a_thread_id=0,
    scale_b_byte_id=0,
    scale_b_thread_id=0,
):
    """TVM intrinsic for SM120a warp-level NVF4 block-scaled MMA."""

    def _selector_value(value):
        return IntImm("int32", value) if isinstance(value, int) else value

    return _call_intrin(
        accum_dtype,
        _tvm_op.Op.get("tl.ptx_mma_block_scale"),
        tvm.tirx.StringImm(str(accum_dtype)),
        tvm.tirx.StringImm(str(shape)),
        tvm.tirx.StringImm(str(A_layout)),
        tvm.tirx.StringImm(str(B_layout)),
        tvm.tirx.StringImm(str(kind)),
        IntImm("int32", scale_vec_size),
        tvm.tirx.StringImm(str(A_dtype)),
        tvm.tirx.StringImm(str(B_dtype)),
        tvm.tirx.StringImm(str(scale_type)),
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
        scale_a,
        scale_b,
        _selector_value(scale_a_byte_id),
        _selector_value(scale_a_thread_id),
        _selector_value(scale_b_byte_id),
        _selector_value(scale_b_thread_id),
    )


@_dtype_forward
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
    return _call_intrin(
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


@_dtype_forward
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
    return _call_intrin(
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


@_dtype_forward
def ptx_wgmma_sp_ss(
    dtype,
    wgmma_prefix,
    a_is_k_major,
    b_is_k_major,
    a_dtype_abbrv,
    b_dtype_abbrv,
    accum_dtype_abbrv,
    A_desc,
    A_offset,
    E_data,
    E_offset,
    sparse_selector,
    B_desc,
    B_offset,
    C_data,
    C_offset,
    scale_out,
    scale_in_a,
    scale_in_b,
):
    return _call_intrin(
        dtype,
        _tvm_op.Op.get("tl.ptx_wgmma_sp_ss"),
        wgmma_prefix,
        a_is_k_major,
        b_is_k_major,
        a_dtype_abbrv,
        b_dtype_abbrv,
        accum_dtype_abbrv,
        A_desc,
        A_offset,
        E_data,
        E_offset,
        sparse_selector,
        B_desc,
        B_offset,
        C_data,
        C_offset,
        scale_out,
        scale_in_a,
        scale_in_b,
    )


@_dtype_forward
def ptx_wgmma_sp_rs(
    dtype,
    wgmma_prefix,
    b_is_k_major,
    a_dtype_abbrv,
    b_dtype_abbrv,
    accum_dtype_abbrv,
    A_buf,
    A_offset,
    E_buf,
    E_offset,
    sparse_selector,
    B_desc,
    B_offset,
    C_data,
    C_offset,
    scale_out,
    scale_in_a,
    scale_in_b,
):
    return _call_intrin(
        dtype,
        _tvm_op.Op.get("tl.ptx_wgmma_sp_rs"),
        wgmma_prefix,
        b_is_k_major,
        a_dtype_abbrv,
        b_dtype_abbrv,
        accum_dtype_abbrv,
        A_buf,
        A_offset,
        E_buf,
        E_offset,
        sparse_selector,
        B_desc,
        B_offset,
        C_data,
        C_offset,
        scale_out,
        scale_in_a,
        scale_in_b,
    )


@_dtype_forward
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
    enable_2cta=False,
    ws=None,
    warp_specialized=None,
    variant=None,
):
    """TVM intrinsic for tcgen05.mma shared-memory x shared-memory instructions.

    Expects 14 or 15 positional arguments:
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

    return _call_intrin(
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
        enable_2cta,
    )


@_dtype_forward
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
    enable_2cta=False,
):
    """TVM intrinsic for tcgen05.mma tensor-memory x shared-memory instructions.

    Expects 13 positional arguments:
    (kind_dtype, A_ptr, A_offset, desc_b, B_offset, C_ptr, C_offset,
     desc_val, scale_out, mask0, mask1, mask2, mask3).
    - kind_dtype: instruction kind selector (e.g., T.float16 for kind::f16,
      "tf32" for kind::tf32, "int8" for kind::i8, "float8_e4m3" for kind::f8f6f4).
    """
    return _call_intrin(
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
        enable_2cta,
    )


@_dtype_forward
def ptx_tcgen05_mma_blockscaled_ss(
    kind_dtype,
    desc_a,
    A_offset,
    desc_b,
    B_offset,
    C_ptr,
    C_offset,
    desc_val,
    scale_out,
    sfa_ptr,
    sfa_offset,
    sfb_ptr,
    sfb_offset,
    reserved0=0,
    reserved1=0,
    enable_2cta=False,
):
    """TVM intrinsic for tcgen05.mma block-scaled (mxf8f6f4.block_scale) instructions.

    Block-scaled TCGEN05 is explicit-async and carries an explicit ``enable_2cta``
    flag, analogous to the regular SS/TS TCGEN05 intrinsics. There is no
    fallback path if 2CTA is requested.

    Positional args:
    kind_dtype, desc_a, A_offset, desc_b, B_offset, C_ptr, C_offset,
    desc_val, scale_out, sfa_ptr, sfa_offset, sfb_ptr, sfb_offset,
    reserved0, reserved1, enable_2cta.
    """

    return _call_intrin(
        "handle",
        _tvm_op.Op.get("tl.ptx_tcgen05_mma_blockscaled_ss"),
        kind_dtype,
        desc_a,
        A_offset,
        desc_b,
        B_offset,
        C_ptr,
        C_offset,
        desc_val,
        scale_out,
        sfa_ptr,
        sfa_offset,
        sfb_ptr,
        sfb_offset,
        reserved0,
        reserved1,
        enable_2cta,
    )


@_dtype_forward
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


@_dtype_forward
def ptx_ldmatrix(trans, num, src_access_ptr, dst_access_ptr):
    """TileLang intrinsic for ptx load matrix from shared memory

    Uses `tl.ptx_ldmatrix` which expects access pointers created via
    `T.access_ptr` (i.e. `tl.access_ptr` wrapping a `BufferLoad`).

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Parameters
    ----------
    trans : bool
        The matrix is loaded in column-major format.

    num : IntImm
        The number of matrices (2 or 4).

    src_access_ptr : PrimExpr
        A `tl.access_ptr` pointing to the source (shared memory) buffer.

    dst_access_ptr : PrimExpr
        A `tl.access_ptr` pointing to the destination (local/register) buffer.

    Returns
    -------
    call : PrimExpr
        The call expression (handle-typed).
    """
    return tvm.tirx.call_intrin(
        "handle",
        tvm.tirx.op.Op.get("tl.ptx_ldmatrix"),
        trans,
        num,
        src_access_ptr,
        dst_access_ptr,
    )


@_op_wrapper
def ptx_fence_barrier_init():
    """TVM intrinsic for ptx fence barrier initialization.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _call_intrin("handle", _tvm_op.Op.get("tl.ptx_fence_barrier_init"))


__all__ = tuple(sorted(CUDA_ONLY_TIR_EXPORTS | SHARED_LEGACY_TIR_EXPORTS))
