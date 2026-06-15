"""Reusable TileLang TIR helpers for the SM120 SageAttention3 example."""

from __future__ import annotations

import tilelang.language as T


@T.macro
def pack_cast2_u32(target_dtype, storage_dtype, v0, v1):
    """Pack two scalar values by vector-casting them to a packed dtype."""
    return T.Cast(
        T.uint32,
        T.reinterpret(
            T.Cast(T.dtype(target_dtype).with_lanes(2), T.Shuffle([v0, v1], [0, 1])),
            storage_dtype,
        ),
    )


@T.macro
def mma_m16n32k64_blockscale_f32(
    a_regs,
    a_offset,
    b_regs,
    b_offset,
    acc,
    c_offset,
    scale_a,
    scale_b,
    scale_id_a,
    scale_id_b_base,
) -> None:
    """Emit four m16n8k64 FP4 MMAs in contiguous n8-atom register order."""
    scale_a_reg = T.alloc_var("uint32", init=scale_a, role_scoped=True)
    scale_b_reg = T.alloc_var("uint32", init=scale_b, role_scoped=True)
    T.ptx_mma_blockscaled(
        "float32",
        "m16n8k64",
        "row",
        "col",
        "e2m1",
        "e2m1",
        "float32",
        "float8_e4m3",
        4,
        a_regs,
        a_offset,
        b_regs,
        b_offset,
        acc.data,
        c_offset,
        scale_a_reg,
        scale_b_reg,
        scale_id_a,
        scale_id_b_base,
    )
    T.ptx_mma_blockscaled(
        "float32",
        "m16n8k64",
        "row",
        "col",
        "e2m1",
        "e2m1",
        "float32",
        "float8_e4m3",
        4,
        a_regs,
        a_offset,
        b_regs,
        b_offset + 4,
        acc.data,
        c_offset + 4,
        scale_a_reg,
        scale_b_reg,
        scale_id_a,
        scale_id_b_base + 1,
    )
    T.ptx_mma_blockscaled(
        "float32",
        "m16n8k64",
        "row",
        "col",
        "e2m1",
        "e2m1",
        "float32",
        "float8_e4m3",
        4,
        a_regs,
        a_offset,
        b_regs,
        b_offset + 8,
        acc.data,
        c_offset + 8,
        scale_a_reg,
        scale_b_reg,
        scale_id_a,
        scale_id_b_base + 2,
    )
    T.ptx_mma_blockscaled(
        "float32",
        "m16n8k64",
        "row",
        "col",
        "e2m1",
        "e2m1",
        "float32",
        "float8_e4m3",
        4,
        a_regs,
        a_offset,
        b_regs,
        b_offset + 12,
        acc.data,
        c_offset + 12,
        scale_a_reg,
        scale_b_reg,
        scale_id_a,
        scale_id_b_base + 3,
    )
