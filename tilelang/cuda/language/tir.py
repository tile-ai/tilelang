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
mma_fill = _dtype_forward(_tvm_op.mma_fill)
mma_store = _dtype_forward(_tvm_op.mma_store)
ptx_cp_async_bulk = _dtype_forward(_tvm_op.ptx_cp_async_bulk)
ptx_mma = _dtype_forward(_tvm_op.ptx_mma)
ptx_mma_sp = _dtype_forward(_tvm_op.ptx_mma_sp)
ptx_wait_barrier = _op_wrapper(_tvm_op.ptx_wait_barrier)


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
    """Build an SM120a warp-level NVF4 block-scaled MMA call."""

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
    """Build a TCGEN05 shared-memory x shared-memory MMA call."""
    if ws is not None:
        enable_ws = bool(ws)
    if warp_specialized is not None:
        enable_ws = bool(warp_specialized)
    if variant is not None:
        if isinstance(variant, str):
            variant = variant.lower()
            if variant in ("ws", "warp_specialized", "warp-specialized"):
                enable_ws = True
            elif variant in ("default", "std", "ss"):
                enable_ws = False
            else:
                raise ValueError(f"ptx_tcgen05_mma_ss: unknown variant: {variant}")
        else:
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
def ptx_ldmatrix(trans, num, src_access_ptr, dst_access_ptr):
    """Build a TileLang PTX ldmatrix call from source/destination access pointers."""
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
    """Build a PTX fence for initialized mbarriers."""
    return _call_intrin("handle", _tvm_op.Op.get("tl.ptx_fence_barrier_init"))


__all__ = tuple(sorted(CUDA_ONLY_TIR_EXPORTS | SHARED_LEGACY_TIR_EXPORTS))
