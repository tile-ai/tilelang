"""
Block-scaled warp MMA for the CuTeDSL backend (SM120 family: SM120, SM121).
Based on tl_templates/cuda/instruction/mma_block_scale.h

Supported configurations (row.col, f32 accumulator):
- m16n8k64 kind::mxf4nvf4 scale_vec::4X, e2m1 x e2m1, ue4m3 scales (NVFP4)
- m16n8k64 kind::mxf4nvf4 scale_vec::2X, e2m1 x e2m1, ue8m0 scales (MXFP4)
- m16n8k32 kind::mxf8f6f4 scale_vec::1X, {e2m1,e3m2,e2m3,e4m3,e5m2} operands,
  ue8m0 scales (MXFP8/MXFP6/MXFP4, in any combination)

Each scale operand is a packed uint32 scale word. The byte selector picks one
scale byte (scale_vec::1X) or a pair of scale bytes (scale_vec::2X) of the
word, and the thread selector picks the lane that holds it.
"""

__all__ = [
    "ptx_mma_block_scale",
]

import functools

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer
import cutlass.cute as cute

# (kind, scale_vec_size, scale_type) -> PTX shape
_BLOCK_SCALE_SHAPES = {
    ("mxf4nvf4", 4, "ue4m3"): "m16n8k64",
    ("mxf4nvf4", 2, "ue8m0"): "m16n8k64",
    ("mxf8f6f4", 1, "ue8m0"): "m16n8k32",
}
_MXF8F6F4_TYPES = ("e2m1", "e3m2", "e2m3", "e4m3", "e5m2")

# 4 D outputs, then A (4), B (2), C (4), scale A data, its byte and thread
# selectors, scale B data, its byte and thread selectors.
_CONSTRAINTS = ",".join(["=f"] * 4 + ["r"] * 6 + ["f"] * 4 + ["r", "h", "h", "r", "h", "h"])


def _load_words(ptr, offset, count, loc=None, ip=None):
    """Read ``count`` 32-bit registers starting ``offset`` elements past ``ptr``.

    The base pointer is recast before offsetting: a sub-byte (packed E2M1)
    pointer plus a runtime offset has no known alignment to recast from.
    """
    elems_per_word = 32 // ptr.dtype.width
    words = cute.make_tensor(cute.recast_ptr(ptr, dtype=cute.Int32) + offset // elems_per_word, (count,))
    return [cute.Int32(words[i]).ir_value(loc=loc, ip=ip) for i in range(count)]


@functools.cache
def _make_ptx_mma_block_scale(shape, kind, scale_vec_size, a_type, b_type, scale_type):
    """Factory to create a @dsl_user_op for one ``mma.sync ... .block_scale`` form."""
    ptx_asm = (
        f"mma.sync.aligned.{shape}.row.col.kind::{kind}.block_scale.scale_vec::{scale_vec_size}X"
        f".f32.{a_type}.{b_type}.f32.{scale_type}"
        " {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13},"
        " {$14}, {$15, $16}, {$17}, {$18, $19};"
    )

    @dsl_user_op
    def mma_op(
        a_ptr: Pointer,
        a_offset,
        b_ptr: Pointer,
        b_offset,
        c_ptr: Pointer,
        c_offset,
        scale_a_ptr: Pointer,
        scale_b_ptr: Pointer,
        scale_a_byte_id,
        scale_a_thread_id,
        scale_b_byte_id,
        scale_b_thread_id,
        *,
        loc=None,
        ip=None,
    ) -> None:
        a_vals = _load_words(a_ptr, a_offset, 4, loc, ip)
        b_vals = _load_words(b_ptr, b_offset, 2, loc, ip)
        c_base = c_ptr + c_offset
        c_tensor = cute.make_tensor(c_base, (4,))
        c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

        def scale_operands(scale_ptr, byte_id, thread_id):
            return _load_words(scale_ptr, 0, 1, loc, ip) + [
                cute.Uint16(byte_id).ir_value(loc=loc, ip=ip),
                cute.Uint16(thread_id).ir_value(loc=loc, ip=ip),
            ]

        scale_vals = scale_operands(scale_a_ptr, scale_a_byte_id, scale_a_thread_id) + scale_operands(
            scale_b_ptr, scale_b_byte_id, scale_b_thread_id
        )

        f32 = T.f32()
        result = llvm.inline_asm(
            llvm.StructType.get_literal([f32] * 4),
            a_vals + b_vals + c_vals + scale_vals,
            ptx_asm,
            _CONSTRAINTS,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

        d_tensor = cute.make_tensor(c_base, (4,))
        for i in range(4):
            d_tensor[i] = cute.Float32(llvm.extractvalue(f32, result, [i], loc=loc, ip=ip))

    return mma_op


def ptx_mma_block_scale(
    accum_dtype: str,
    shape: str,
    a_layout: str,
    b_layout: str,
    kind: str,
    scale_vec_size: int,
    a_dtype: str,
    b_dtype: str,
    scale_type: str,
    a_ptr,
    a_offset,
    b_ptr,
    b_offset,
    c_ptr,
    c_offset,
    scale_a_ptr,
    scale_b_ptr,
    scale_a_byte_id=0,
    scale_a_thread_id=0,
    scale_b_byte_id=0,
    scale_b_thread_id=0,
):
    """Generic SM120-family block-scaled MMA dispatcher.

    Mirrors ``tl::sm120_mma_sync_blockscaled``: validates the configuration
    and issues the matching ``mma.sync ... .block_scale`` instruction.
    """
    key = (kind, int(scale_vec_size), scale_type)
    if kind == "mxf4nvf4":
        operands_ok = a_dtype == "e2m1" and b_dtype == "e2m1"
    else:
        operands_ok = a_dtype in _MXF8F6F4_TYPES and b_dtype in _MXF8F6F4_TYPES
    supported = (
        accum_dtype in ("float32", "fp32", "f32")
        and (a_layout, b_layout) == ("row", "col")
        and _BLOCK_SCALE_SHAPES.get(key) == shape
        and operands_ok
    )
    if not supported:
        raise ValueError(
            "Unsupported block-scaled MMA configuration: "
            f"accum_dtype={accum_dtype}, shape={shape}, layout={a_layout}.{b_layout}, kind={kind}, "
            f"scale_vec_size={scale_vec_size}, a_dtype={a_dtype}, b_dtype={b_dtype}, scale_type={scale_type}. "
            "Currently supported: f32 m16n8k64 row.col kind::mxf4nvf4 scale_vec::4X e2m1.e2m1 ue4m3 "
            "or scale_vec::2X e2m1.e2m1 ue8m0, and f32 m16n8k32 row.col kind::mxf8f6f4 scale_vec::1X "
            "{e2m1,e3m2,e2m3,e4m3,e5m2} ue8m0."
        )
    mma_op = _make_ptx_mma_block_scale(shape, kind, int(scale_vec_size), a_dtype, b_dtype, scale_type)
    return mma_op(
        a_ptr,
        a_offset,
        b_ptr,
        b_offset,
        c_ptr,
        c_offset,
        scale_a_ptr,
        scale_b_ptr,
        scale_a_byte_id,
        scale_a_thread_id,
        scale_b_byte_id,
        scale_b_thread_id,
    )
