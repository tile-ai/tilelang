"""Tests for Reduce / Cumsum / ThreadAllreduce emit_mlir.

Test structure
--------------
1. test_reduce_emits_reduce_op
   Build a Reduce (sum, axis=1) over a 16×16 fp16 src tile into a 16-element
   fp16 dst tile.  Assert that str(module) contains the reduce op mnemonic.

2. test_reduce_max_axis0
   Reduce with op="max" along axis=0.  Assert reduce mnemonic is present.

3. test_cumsum_emits_scan_op
   Build a Cumsum (axis=0) over a 16×16 fp32 tile.  Assert that str(module)
   contains the scan op mnemonic.

4. test_cumsum_reverse_emits_scan_op
   Build a Cumsum (axis=0, reverse=True).  Assert scan mnemonic is present.

5. test_thread_allreduce_raises_unsupported
   Instantiate ThreadAllreduce and call emit_mlir(ctx=None).
   Assert it raises _UnsupportedTileIRNode (NOT generic NotImplementedError).

Build pattern (tests 1-4)
--------------------------
  1. Create a root Block with two GLOBAL buffer params: src (shape, dtype) and
     dst (reduced shape, same dtype).
  2. Add a Reduce / Cumsum op over those buffer Values.
  3. Call emit_module and inspect str(module).

Note: the emit_mlir for Reduce/Cumsum needs to load+store tiles from buffers,
so both src and dst must be GLOBAL buffer params (same pattern as Gemm).
"""

from __future__ import annotations

import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import Reduce, Cumsum, ThreadAllreduce
from tilelang.tileir.errors import _UnsupportedTileIRNode

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


def _buf_ty(elem, shape):
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.GLOBAL, layout=None)


# ---------------------------------------------------------------------------
# Helper: build root block + Reduce op
# ---------------------------------------------------------------------------


def _build_reduce_root(
    *,
    src_shape=(16, 16),
    dst_shape=(16,),
    elem=FP16,
    reduce_op="sum",
    axis=1,
    clear=True,
):
    """Return (root, entry_args) for a single Reduce op."""
    src_val = Value(0, _buf_ty(elem, src_shape), name="src")
    dst_val = Value(1, _buf_ty(elem, dst_shape), name="dst")

    root = Block(params=[src_val, dst_val])
    entry_args = [
        ("src", src_val.type),
        ("dst", dst_val.type),
    ]

    op = Reduce(src=src_val, dst=dst_val, op=reduce_op, axis=axis, clear=clear)
    op.results = ()
    root.append(op)

    return root, entry_args


# ---------------------------------------------------------------------------
# Helper: build root block + Cumsum op
# ---------------------------------------------------------------------------


def _build_cumsum_root(
    *,
    shape=(16, 16),
    elem=FP32,
    axis=0,
    reverse=False,
):
    """Return (root, entry_args) for a single Cumsum op."""
    src_val = Value(0, _buf_ty(elem, shape), name="src")
    dst_val = Value(1, _buf_ty(elem, shape), name="dst")

    root = Block(params=[src_val, dst_val])
    entry_args = [
        ("src", src_val.type),
        ("dst", dst_val.type),
    ]

    op = Cumsum(src=src_val, dst=dst_val, axis=axis, reverse=reverse)
    op.results = ()
    root.append(op)

    return root, entry_args


# ===========================================================================
# Tests — Reduce
# ===========================================================================


@skip_no_cuda_tile
def test_reduce_emits_reduce_op():
    """Reduce sum along axis=1 lowers to a cuda_tile reduce op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_reduce_root(reduce_op="sum", axis=1)
    module = emit_module(root, kernel_name="reduce_sum", entry_args=entry_args)
    text = str(module)

    assert "reduce" in text, f"Expected 'reduce' in MLIR output:\n{text}"


@skip_no_cuda_tile
def test_reduce_max_axis0():
    """Reduce max along axis=0 includes the reduce op mnemonic."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_reduce_root(
        reduce_op="max",
        axis=0,
        src_shape=(16, 16),
        dst_shape=(16,),
    )
    module = emit_module(root, kernel_name="reduce_max", entry_args=entry_args)
    text = str(module)

    assert "reduce" in text, f"Expected 'reduce' in MLIR output:\n{text}"


# ===========================================================================
# Tests — Cumsum
# ===========================================================================


@skip_no_cuda_tile
def test_cumsum_emits_scan_op():
    """Cumsum along axis=0 lowers to a cuda_tile scan op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_cumsum_root(axis=0)
    module = emit_module(root, kernel_name="cumsum_fwd", entry_args=entry_args)
    text = str(module)

    assert "scan" in text, f"Expected 'scan' in MLIR output:\n{text}"


@skip_no_cuda_tile
def test_cumsum_reverse_emits_scan_op():
    """Cumsum with reverse=True still emits a scan op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_cumsum_root(axis=0, reverse=True)
    module = emit_module(root, kernel_name="cumsum_rev", entry_args=entry_args)
    text = str(module)

    assert "scan" in text, f"Expected 'scan' in MLIR output:\n{text}"


# ===========================================================================
# Test — ThreadAllreduce (must raise _UnsupportedTileIRNode)
# ===========================================================================


def test_thread_allreduce_raises_unsupported():
    """ThreadAllreduce.emit_mlir raises _UnsupportedTileIRNode (not NotImplementedError)."""
    op = ThreadAllreduce()
    op.results = ()

    with pytest.raises(_UnsupportedTileIRNode):
        op.emit_mlir(ctx=None)
