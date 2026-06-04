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
    for n8 in T.unroll(4):
        # The upstream v0.1.13 intrinsic takes native register-array offsets.
        # The original Sage3 helper expressed A/B offsets in packed FP4 slots,
        # hence the factor-of-two conversion for uint32 register buffers.
        T.ptx_mma_block_scale(
            "float32",
            "m16n8k64",
            "row",
            "col",
            "mxf4nvf4",
            4,
            "e2m1",
            "e2m1",
            "ue4m3",
            a_regs,
            a_offset // 2,
            b_regs,
            (b_offset + n8 * 4) // 2,
            acc.data,
            c_offset + n8 * 4,
            T.access_ptr(scale_a_reg, "r"),
            T.access_ptr(scale_b_reg, "r"),
            0,
            scale_id_a,
            0,
            scale_id_b_base + n8,
        )
