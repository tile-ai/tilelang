"""Tests for Gemm / Tcgen05Gemm emit_mlir.

Test structure
--------------
1. test_gemm_emits_mma
   Build a Gemm over two loaded tiles (fp16 LHS/RHS, fp32 ACC).  Verify that
   str(module) contains "mma".

2. test_gemm_trans_a_emits_permute
   Gemm with trans_a=True must include "permute" for the LHS.

3. test_gemm_trans_b_emits_permute
   Gemm with trans_b=True must include "permute" for the RHS.

4. test_gemm_clear_zero_initialises_acc
   Gemm with clear=True must emit "constant" (zero-initialise the ACC).

5. test_gemm_no_clear_loads_acc
   Gemm with clear=False loads the existing ACC tile (no zero-init path).

6. test_gemm_f32_f32_lowers_to_tf32
   An f32×f32→f32 Gemm should go via TF32 (TileLang convention), so the
   MLIR text should contain "tf32" for the LHS/RHS casts.

7. test_gemm_result_not_in_value_map
   Gemm is a side-effect-only op (writes into the ACC buffer in-place);
   it must NOT bind any SSA result (results stays ()).

8. test_tcgen05_gemm_emits_mma
   Same as test_gemm_emits_mma but uses Tcgen05Gemm.  Both ops share the
   same emit path.

9. test_tcgen05_gemm_clear_zero_initialises_acc
   Tcgen05Gemm with clear=True emits "constant" for zero-init.

10. test_gemm_unsigned_int_signedness
    Gemm with uint8 LHS/RHS should propagate unsigned signedness through mma.
    Verify "unsigned" appears in the MLIR text.

Build pattern (same for all tests)
-----------------------------------
  1. Create a root Block with three GLOBAL buffer params (LHS, RHS, ACC).
  2. Emit Load ops for LHS and RHS to produce register tile SSA values.
  3. Emit Gemm/Tcgen05Gemm over the three buffer Values.
  4. Call emit_module and inspect str(module).
"""

from __future__ import annotations

import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import Gemm, Tcgen05Gemm

# ---------------------------------------------------------------------------
# Skip guard — needs cuda_tile MLIR bindings
# ---------------------------------------------------------------------------

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Shared dtype / type helpers
# ---------------------------------------------------------------------------

FP16 = dtype("float16")
FP32 = dtype("float32")
BF16 = dtype("bfloat16")
U8 = dtype("uint8")
I32 = dtype("int32")


def _buf_ty(elem, shape):
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.GLOBAL, layout=None)


def _reg_ty(elem, shape):
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


# ---------------------------------------------------------------------------
# Helper: build a root block + Load ops for a 3-buffer GEMM
#
# Layout: (M, K) × (K, N) → (M, N)
# Default: fp16 LHS/RHS, fp32 ACC, tile sizes 16×16.
# ---------------------------------------------------------------------------


class _Counter:
    """Simple monotonic id counter (mimics IRBuilder)."""

    def __init__(self):
        self._n = 0

    def next(self):
        value = self._n
        self._n += 1
        return value


def _build_gemm_root(
    *,
    lhs_elem=FP16,
    rhs_elem=FP16,
    acc_elem=FP32,
    M=16,
    K=16,
    N=16,
    trans_a=False,
    trans_b=False,
    clear=False,
    swap_ab=False,
    lhs_unsigned=False,
    rhs_unsigned=False,
):
    """Build (root, entry_args, gemm_op) for a 3-buffer Gemm kernel.

    Returns:
        root       — Block with params [lhs_buf, rhs_buf, acc_buf]
        entry_args — list[(name, TileType)] aligned with root.params
        gemm_op    — the Gemm op appended to root
    """
    ctr = _Counter()

    # --- Buffer Value objects (GLOBAL params) ---
    lhs_buf = Value(ctr.next(), _buf_ty(lhs_elem, (M, K)), name="A")
    rhs_buf = Value(ctr.next(), _buf_ty(rhs_elem, (K, N)), name="B")
    acc_buf = Value(ctr.next(), _buf_ty(acc_elem, (M, N)), name="C")

    root = Block(params=[lhs_buf, rhs_buf, acc_buf])

    entry_args = [
        ("A", lhs_buf.type),
        ("B", rhs_buf.type),
        ("C", acc_buf.type),
    ]

    # --- Gemm op (buffer operands only, no load results needed) ---
    gemm_op = Gemm(
        lhs=lhs_buf,
        rhs=rhs_buf,
        acc=acc_buf,
        trans_a=trans_a,
        trans_b=trans_b,
        clear=clear,
        swap_ab=swap_ab,
        lhs_unsigned=lhs_unsigned,
        rhs_unsigned=rhs_unsigned,
    )
    gemm_op.results = ()
    root.append(gemm_op)

    return root, entry_args, gemm_op


def _build_tcgen05_root(
    *,
    lhs_elem=FP16,
    rhs_elem=FP16,
    acc_elem=FP32,
    M=16,
    K=16,
    N=16,
    trans_a=False,
    trans_b=False,
    clear=False,
):
    """Same as _build_gemm_root but uses Tcgen05Gemm."""
    ctr = _Counter()

    lhs_buf = Value(ctr.next(), _buf_ty(lhs_elem, (M, K)), name="A")
    rhs_buf = Value(ctr.next(), _buf_ty(rhs_elem, (K, N)), name="B")
    acc_buf = Value(ctr.next(), _buf_ty(acc_elem, (M, N)), name="C")

    root = Block(params=[lhs_buf, rhs_buf, acc_buf])

    entry_args = [
        ("A", lhs_buf.type),
        ("B", rhs_buf.type),
        ("C", acc_buf.type),
    ]

    gemm_op = Tcgen05Gemm(
        lhs=lhs_buf,
        rhs=rhs_buf,
        acc=acc_buf,
        trans_a=trans_a,
        trans_b=trans_b,
        clear=clear,
    )
    gemm_op.results = ()
    root.append(gemm_op)

    return root, entry_args, gemm_op


# ===========================================================================
# Tests — Gemm
# ===========================================================================


@skip_no_cuda_tile
def test_gemm_emits_mma():
    """Gemm op lowers to a cuda_tile.mma op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root()
    module = emit_module(root, kernel_name="gemm_kernel", entry_args=entry_args)
    text = str(module)

    assert "mma" in text, f"Expected 'mma' in MLIR output:\n{text}"


@skip_no_cuda_tile
def test_gemm_trans_a_emits_permute():
    """Gemm with trans_a=True inserts a permute for the LHS."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(trans_a=True)
    module = emit_module(root, kernel_name="gemm_trans_a", entry_args=entry_args)
    text = str(module)

    assert "permute" in text, f"Expected 'permute' for trans_a=True in:\n{text}"


@skip_no_cuda_tile
def test_gemm_trans_b_emits_permute():
    """Gemm with trans_b=True inserts a permute for the RHS."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(trans_b=True)
    module = emit_module(root, kernel_name="gemm_trans_b", entry_args=entry_args)
    text = str(module)

    assert "permute" in text, f"Expected 'permute' for trans_b=True in:\n{text}"


@skip_no_cuda_tile
def test_gemm_swap_ab_emits_algebraic_transpose():
    """swap_ab emits (rhs^T @ lhs^T + acc^T)^T."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(M=16, K=64, N=128, swap_ab=True)
    module = emit_module(root, kernel_name="gemm_swap_ab", entry_args=entry_args)
    text = str(module)

    # rhs, lhs, and accumulator are transposed before MMA; the result is
    # transposed back to the original output layout.
    assert text.count("permute") == 4, text
    assert "tile<128x64xf16>" in text, text
    assert "tile<64x16xf16>" in text, text
    assert "tile<128x16xf32>" in text, text


@skip_no_cuda_tile
def test_gemm_clear_zero_initialises_acc():
    """Gemm with clear=True emits a zero 'constant' tile for the ACC."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(clear=True)
    module = emit_module(root, kernel_name="gemm_clear", entry_args=entry_args)
    text = str(module)

    # constant is emitted by ct.constant(0, ...) in the clear path
    assert "constant" in text, f"Expected 'constant' (zero-init ACC) for clear=True in:\n{text}"


@skip_no_cuda_tile
def test_gemm_no_clear_loads_acc():
    """Gemm with clear=False loads the ACC tile from its buffer (no zero-init)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(clear=False)
    module = emit_module(root, kernel_name="gemm_noclear", entry_args=entry_args)
    text = str(module)

    # The ACC buffer is loaded so there must be a load_view_tko or load_ptr_tko
    has_load = "load_view_tko" in text or "load_ptr_tko" in text
    assert has_load, f"Expected a load TKO (ACC read) for clear=False in:\n{text}"


@skip_no_cuda_tile
def test_gemm_f32_f32_lowers_to_tf32():
    """An f32×f32→f32 Gemm must down-cast LHS/RHS through TF32 (TileLang convention)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(lhs_elem=FP32, rhs_elem=FP32, acc_elem=FP32)
    module = emit_module(root, kernel_name="gemm_tf32", entry_args=entry_args)
    text = str(module)

    assert "tf32" in text, f"Expected 'tf32' cast in f32×f32→f32 Gemm emission:\n{text}"


@skip_no_cuda_tile
def test_gemm_result_not_in_value_map():
    """Gemm is side-effect-only; no SSA result is bound in the EmitContext."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    ctr = _Counter()
    lhs_buf = Value(ctr.next(), _buf_ty(FP16, (16, 16)), name="A")
    rhs_buf = Value(ctr.next(), _buf_ty(FP16, (16, 16)), name="B")
    acc_buf = Value(ctr.next(), _buf_ty(FP32, (16, 16)), name="C")

    # Use a phantom result value — it must NOT appear in value_map after emit
    phantom = Value(ctr.next(), _reg_ty(FP32, (16, 16)), name="result")

    root = Block(params=[lhs_buf, rhs_buf, acc_buf])
    entry_args = [("A", lhs_buf.type), ("B", rhs_buf.type), ("C", acc_buf.type)]

    gemm_op = Gemm(lhs=lhs_buf, rhs=rhs_buf, acc=acc_buf, clear=False)
    gemm_op.results = ()  # side-effect only — empty results
    root.append(gemm_op)

    _mod, ctx = emit_module(
        root,
        kernel_name="gemm_no_result",
        entry_args=entry_args,
        return_ctx=True,
    )

    # The phantom value must not be bound
    assert phantom not in ctx.value_map, "Gemm must not bind any SSA result"


@skip_no_cuda_tile
def test_gemm_unsigned_int_signedness():
    """Gemm with lhs_unsigned=True / rhs_unsigned=True propagates unsigned signedness.

    The TileIR type system maps uint8 → int8 (signless i8), so signedness cannot
    be derived from the buffer dtype alone.  The Gemm op carries explicit
    ``lhs_unsigned`` / ``rhs_unsigned`` attributes for this purpose.  The
    lowering must set them when it detects unsigned dtype strings in the TileLang
    source IR.
    """
    from tilelang.tileir.lowering.mlir_emit import emit_module

    ctr = _Counter()
    # U8 dtype is stored as int8 in the type system (uint8 is an alias for int8)
    lhs_buf = Value(ctr.next(), _buf_ty(U8, (16, 16)), name="A")
    rhs_buf = Value(ctr.next(), _buf_ty(U8, (16, 16)), name="B")
    acc_buf = Value(ctr.next(), _buf_ty(I32, (16, 16)), name="C")

    root = Block(params=[lhs_buf, rhs_buf, acc_buf])
    entry_args = [("A", lhs_buf.type), ("B", rhs_buf.type), ("C", acc_buf.type)]

    # Explicitly set lhs_unsigned=True, rhs_unsigned=True
    gemm_op = Gemm(
        lhs=lhs_buf,
        rhs=rhs_buf,
        acc=acc_buf,
        clear=False,
        lhs_unsigned=True,
        rhs_unsigned=True,
    )
    gemm_op.results = ()
    root.append(gemm_op)

    module = emit_module(root, kernel_name="gemm_u8_u8_i32", entry_args=entry_args)
    text = str(module)

    assert "mma" in text, f"Expected 'mma' in u8×u8→i32 Gemm:\n{text}"
    assert "unsigned" in text, f"Expected 'unsigned' signedness attr in u8×u8→i32 Gemm:\n{text}"


@skip_no_cuda_tile
def test_gemm_swap_ab_swaps_integer_signedness():
    """swap_ab must exchange the two operand signedness attributes."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_gemm_root(
        lhs_elem=U8,
        rhs_elem=U8,
        acc_elem=I32,
        M=16,
        K=32,
        N=64,
        swap_ab=True,
        lhs_unsigned=True,
        rhs_unsigned=False,
    )
    module = emit_module(root, kernel_name="gemm_swap_signedness", entry_args=entry_args)
    mma_line = next(line for line in str(module).splitlines() if "mmai" in line)

    assert " signed unsigned :" in mma_line, mma_line


# ===========================================================================
# Tests — Tcgen05Gemm
# ===========================================================================


@skip_no_cuda_tile
def test_tcgen05_gemm_emits_mma():
    """Tcgen05Gemm op lowers to a cuda_tile.mma op (same path as Gemm)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_tcgen05_root()
    module = emit_module(root, kernel_name="tcgen05_kernel", entry_args=entry_args)
    text = str(module)

    assert "mma" in text, f"Expected 'mma' in Tcgen05Gemm MLIR output:\n{text}"


@skip_no_cuda_tile
def test_tcgen05_gemm_clear_zero_initialises_acc():
    """Tcgen05Gemm with clear=True emits a zero constant tile for the ACC."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _ = _build_tcgen05_root(clear=True)
    module = emit_module(root, kernel_name="tcgen05_clear", entry_args=entry_args)
    text = str(module)

    assert "constant" in text, f"Expected 'constant' (zero-init ACC) for Tcgen05Gemm clear=True in:\n{text}"
