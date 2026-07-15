"""SemanticIR -> TileIR lowering: weight-decode tile-op handlers.

Provides the ``@tile_op_impl`` handlers for the packed-weight decode/dp4a extern
ops (decode_i4u_to_f16, decode_i2u_to_i8s, DP4A).  Imports the shared
foundation; the ``_extract_access_ptr_buffer_name`` helper from ``tile_ops`` is
imported function-locally to keep the import graph acyclic.
"""

from __future__ import annotations

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import _UnsupportedTileIRNode
from tilelang.tileir.semantic import SemanticStmt

from ._base import LoweringScope, tile_op_impl

# tile_ops does not import decoders, so this module-level import is acyclic.
from .tile_ops import _extract_access_ptr_buffer_name


@tile_op_impl("decode_i4u_to_f16")
def _lower_decode_i4(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower decode_i4u_to_f16 via the DecodeI4 IR op.

    Extracts src/dst buffer values from the TIR call args (access_ptr structure)
    and creates a ``DecodeI4`` op that handles both GLOBAL ptr and REGISTER tile
    buffers in emit_mlir.
    """
    from tilelang.tileir.ir.ops import DecodeI4

    args = stmt.call_args
    if len(args) < 3:
        raise _UnsupportedTileIRNode("decode_i4u_to_f16: expected TIR call with 3 args (name, src_ptr, dst_ptr).")
    src_name = _extract_access_ptr_buffer_name(args[1])
    dst_name = _extract_access_ptr_buffer_name(args[2])
    if src_name is None or dst_name is None:
        raise _UnsupportedTileIRNode(
            f"decode_i4u_to_f16: could not extract buffer names from access_ptr args (got {args[1]!r}, {args[2]!r})."
        )
    try:
        src_val = scope.lookup_buffer(src_name)
        dst_val = scope.lookup_buffer(dst_name)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"decode_i4u_to_f16: buffer lookup failed: {exc}") from exc
    builder.create(DecodeI4(src=src_val, dst=dst_val))


@tile_op_impl("decode_i2u_to_i8s")
def _lower_decode_i2(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower decode_i2u_to_i8s via the DecodeI2 IR op.

    Extracts src/dst buffer values from the TIR call args (access_ptr structure)
    and creates a ``DecodeI2`` op that handles both GLOBAL ptr and REGISTER tile
    buffers in emit_mlir.
    """
    from tilelang.tileir.ir.ops import DecodeI2

    args = stmt.call_args
    if len(args) < 3:
        raise _UnsupportedTileIRNode("decode_i2u_to_i8s: expected TIR call with 3 args (name, src_ptr, dst_ptr).")
    src_name = _extract_access_ptr_buffer_name(args[1])
    dst_name = _extract_access_ptr_buffer_name(args[2])
    if src_name is None or dst_name is None:
        raise _UnsupportedTileIRNode(
            f"decode_i2u_to_i8s: could not extract buffer names from access_ptr args (got {args[1]!r}, {args[2]!r})."
        )
    try:
        src_val = scope.lookup_buffer(src_name)
        dst_val = scope.lookup_buffer(dst_name)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"decode_i2u_to_i8s: buffer lookup failed: {exc}") from exc
    builder.create(DecodeI2(src=src_val, dst=dst_val))


@tile_op_impl("decode_fp4_to_bf16_twiddling")
def _lower_decode_fp4_twiddling(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower decode_fp4_to_bf16_twiddling via the DecodeFp4Twiddling IR op.

    Args layout: ``call_extern(name, src_ptr, dst_ptr[, N])`` — src/dst are
    access_ptr args; the optional trailing N is the group count from the C
    helper's signature (``const int N = 8``; each group is 4 packed bytes →
    8 bf16 outputs).  A literal N is forwarded for shape validation at emit
    time; a missing/non-literal N falls back to deriving n from the dst
    buffer shape (n_groups=0).
    """
    from tilelang.tileir.ir.ops import DecodeFp4Twiddling

    args = stmt.call_args
    if len(args) < 3:
        raise _UnsupportedTileIRNode("decode_fp4_to_bf16_twiddling: expected TIR call with at least 3 args (name, src_ptr, dst_ptr).")
    src_name = _extract_access_ptr_buffer_name(args[1])
    dst_name = _extract_access_ptr_buffer_name(args[2])
    if src_name is None or dst_name is None:
        raise _UnsupportedTileIRNode(
            f"decode_fp4_to_bf16_twiddling: could not extract buffer names from access_ptr args (got {args[1]!r}, {args[2]!r})."
        )
    n_groups = 0
    if len(args) > 3:
        from tvm import tirx as _tir

        n_arg = args[3]
        if isinstance(n_arg, _tir.IntImm):
            n_groups = int(n_arg)
        else:
            raise _UnsupportedTileIRNode(f"decode_fp4_to_bf16_twiddling: group count N must be a compile-time constant, got {n_arg!r}.")
    try:
        src_val = scope.lookup_buffer(src_name)
        dst_val = scope.lookup_buffer(dst_name)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"decode_fp4_to_bf16_twiddling: buffer lookup failed: {exc}") from exc
    builder.create(DecodeFp4Twiddling(src=src_val, dst=dst_val, n_groups=n_groups))


@tile_op_impl("DP4A")
def _lower_dp4a(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower DP4A via the Dp4a IR op.

    Extracts lhs/rhs/acc buffer values from the TIR call args (access_ptr structure).
    Args layout: call_extern("DP4A", lhs_ptr, rhs_ptr, acc_ptr) — 3 access_ptr args
    after the function name.
    """
    from tilelang.tileir.ir.ops import Dp4a

    # call_extern structure: args[0]=name, args[1]=lhs_ptr, args[2]=rhs_ptr, args[3]=acc_ptr
    args = stmt.call_args
    if len(args) < 4:
        raise _UnsupportedTileIRNode("DP4A: expected TIR call with 4 args (name, lhs_ptr, rhs_ptr, acc_ptr).")
    lhs_name = _extract_access_ptr_buffer_name(args[1])
    rhs_name = _extract_access_ptr_buffer_name(args[2])
    acc_name = _extract_access_ptr_buffer_name(args[3])
    if lhs_name is None or rhs_name is None or acc_name is None:
        raise _UnsupportedTileIRNode(
            f"DP4A: could not extract buffer names from access_ptr args (lhs={args[1]!r}, rhs={args[2]!r}, acc={args[3]!r})."
        )
    try:
        lhs_val = scope.lookup_buffer(lhs_name)
        rhs_val = scope.lookup_buffer(rhs_name)
        acc_val = scope.lookup_buffer(acc_name)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"DP4A: buffer lookup failed: {exc}") from exc
    builder.create(Dp4a(lhs=lhs_val, rhs=rhs_val, acc=acc_val))
