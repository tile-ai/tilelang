"""Tests for data-movement emit_mlir + entry buffer args.

Test structure
--------------
1. test_buffer_entry_arg_flattened
   A single global buffer entry arg (MemSpace.GLOBAL, fp16 2-D) is flattened into
   ptr + 2×shape + 2×stride MLIR args (5 args total).  Verifies str(module)
   contains "tile<ptr<f16>>" and the correct arg count.

2. test_fill_emits_constant
   Build a root with one GLOBAL buffer param + a Fill op.  Verify str(module)
   contains "constant" and the fill value.

3. test_load_emits_load_view_tko
   Build a root with one GLOBAL buffer param + a Load op.  Verify str(module)
   contains "load_view_tko" or "load_ptr_tko".

4. test_store_emits_store_tko
   Build a root with one GLOBAL buffer param + a Store op.  Verify str(module)
   contains "store_view_tko" or "store_ptr_tko".

5. test_copy_global_to_global
   Two GLOBAL buffers + a Copy op.  Verify str(module) contains both a load and
   a store op mnemonic.

6. test_partition_view_emits_make_partition_view
   One GLOBAL buffer + a PartitionView op.  Verify str(module) contains
   "make_partition_view".

7. test_load_result_bound_in_ctx
   Load op result Value is bound in the EmitContext after emission.

8. test_fill_no_result_op
   Fill is a side-effect-only op (no SSA results); emission should not crash.

9. test_token_seam_records_after_load
   After emitting a Load, ctx has a token recorded for the src buffer Value.
"""

from __future__ import annotations


import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings
from tilelang.tileir.errors import TileIRLoweringNotImplementedError
from tilelang.tileir.emission_utils import _make_tile_view

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.ir.ops import (
    Load,
    Store,
    Copy,
    TmaCopy,
    Fill,
    PartitionView,
    RepeatInterleave,
)

# ---------------------------------------------------------------------------
# Skip guard — needs cuda_tile MLIR bindings
# ---------------------------------------------------------------------------

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Shared type helpers
# ---------------------------------------------------------------------------

FP16 = dtype("float16")
I32 = dtype("int32")
I8 = dtype("int8")


def _buf_type(shape=(128, 128)) -> TileType:
    """2-D fp16 global buffer type."""
    return TileType(dtype=FP16, shape=tuple(shape), space=MemSpace.GLOBAL, layout=None)


def _reg_type(shape=(64, 64)) -> TileType:
    """fp16 register tile type (used for Store.val)."""
    return TileType(dtype=FP16, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


def _scalar_type() -> TileType:
    return TileType(dtype=I32, shape=(), space=MemSpace.REGISTER, layout=None)


def test_make_tile_view_rejects_non_power_of_two_shape_before_cuda_tile_builder():
    class FakeCudaTile:
        def make_partition_view(self, *_args, **_kwargs):
            raise AssertionError("invalid tile shape reached the CUDA Tile builder")

    with pytest.raises(TileIRLoweringNotImplementedError, match="power-of-two.*192"):
        _make_tile_view(FakeCudaTile(), object(), [1, 128, 1, 192], elem_view=False, loc=None)


# ---------------------------------------------------------------------------
# Helper: build a root block with one buffer param (and optional extra params)
# ---------------------------------------------------------------------------


def _single_buffer_root() -> tuple[Block, Value]:
    """Return (root_block, buf_val) with one 128x128 fp16 global buffer param."""
    buf_ty = _buf_type()
    buf_val = Value(0, buf_ty, name="A")
    block = Block(params=[buf_val])
    return block, buf_val


def _two_buffer_root() -> tuple[Block, Value, Value]:
    """Return (root_block, src_val, dst_val) with two 128x128 fp16 global buffer params."""
    buf_ty = _buf_type()
    src_val = Value(0, buf_ty, name="src")
    dst_val = Value(1, buf_ty, name="dst")
    block = Block(params=[src_val, dst_val])
    return block, src_val, dst_val


def _entry_args_for(root: Block) -> list[tuple[str, TileType]]:
    """Build the entry_args list aligned with root.params."""
    return [(p.name or f"p{i}", p.type) for i, p in enumerate(root.params)]


# ---------------------------------------------------------------------------
# 1. Buffer entry arg flattening
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_buffer_entry_arg_flattened():
    """A 2-D global buffer is flattened to 5 MLIR args: ptr + 2×shape + 2×stride."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, _buf = _single_buffer_root()
    module = emit_module(root, kernel_name="flat_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    # Pointer arg for fp16 buffer
    assert "tile<ptr<f16>>" in text, f"Expected ptr<f16> arg in:\n{text}"
    # 4 i32 shape/stride args
    assert text.count("tile<i32>") >= 4, f"Expected at least 4 tile<i32> args (2 shape + 2 stride) in:\n{text}"


@skip_no_cuda_tile
def test_buffer_entry_arg_tensor_view_created():
    """emit_module materialises a TensorView for the global buffer."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, _buf = _single_buffer_root()
    module = emit_module(root, kernel_name="view_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "make_tensor_view" in text, f"Expected make_tensor_view in:\n{text}"


# ---------------------------------------------------------------------------
# 2. Fill
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_fill_emits_constant():
    """Fill op lowers to a ct.constant tile with the fill value."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, buf_val = _single_buffer_root()

    b = IRBuilder()
    b._module_block = root
    b._block = root

    # Fill the 64x64 region with 0.0
    fill_op = Fill(dst=buf_val, value=0.0, tile_shape=(64, 64))
    fill_op.results = ()
    root.append(fill_op)

    module = emit_module(root, kernel_name="fill_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "constant" in text, f"Expected 'constant' (fill lowering) in:\n{text}"


@skip_no_cuda_tile
def test_fill_no_result_op():
    """Fill is side-effect-only; emission must not crash and op.results stays ()."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, buf_val = _single_buffer_root()

    fill_op = Fill(dst=buf_val, value=1.0, tile_shape=(64, 64))
    fill_op.results = ()
    root.append(fill_op)

    # Should not raise
    module = emit_module(root, kernel_name="fill_nores_kernel", entry_args=_entry_args_for(root))
    assert module is not None
    text = str(module)
    assert "constant" in text, f"Expected 'constant' in fill emission:\n{text}"
    assert ("store_view_tko" in text) or ("store_ptr_tko" in text), f"Expected store_view_tko or store_ptr_tko in fill emission:\n{text}"


# ---------------------------------------------------------------------------
# 3. Load
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_load_emits_load_tko():
    """Load op emits a load_view_tko or load_ptr_tko instruction."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import fresh_value

    class _Counter:
        def __init__(self):
            self._n = 0

        def next(self):
            value = self._n
            self._n += 1
            return value

    ctr = _Counter()
    root, buf_val = _single_buffer_root()
    result_ty = _reg_type((64, 64))
    result_val = fresh_value(ctr, result_ty, "loaded")

    load_op = Load(src=buf_val, tile_shape=(64, 64), indices=(0, 0))
    load_op.results = (result_val,)
    root.append(load_op)

    module = emit_module(root, kernel_name="load_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "load_view_tko" in text or "load_ptr_tko" in text, f"Expected load_view_tko or load_ptr_tko in:\n{text}"


@skip_no_cuda_tile
def test_load_result_bound_in_ctx():
    """The Load result Value is bound in EmitContext after emission."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import fresh_value

    class _Counter:
        def __init__(self):
            self._n = 0

        def next(self):
            value = self._n
            self._n += 1
            return value

    ctr = _Counter()
    root, buf_val = _single_buffer_root()
    result_ty = _reg_type((64, 64))
    result_val = fresh_value(ctr, result_ty, "loaded")

    load_op = Load(src=buf_val, tile_shape=(64, 64), indices=(0, 0))
    load_op.results = (result_val,)
    root.append(load_op)

    _module, ctx = emit_module(
        root,
        kernel_name="load_bound_kernel",
        entry_args=_entry_args_for(root),
        return_ctx=True,
    )
    assert result_val in ctx.value_map, f"Load result not bound in value_map; keys={list(ctx.value_map.keys())}"


# ---------------------------------------------------------------------------
# 4. Store
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_store_emits_store_tko():
    """Store op emits a store_view_tko or store_ptr_tko instruction."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import fresh_value

    class _Counter:
        def __init__(self):
            self._n = 0

        def next(self):
            value = self._n
            self._n += 1
            return value

    ctr = _Counter()
    root, buf_val = _single_buffer_root()
    tile_ty = _reg_type((64, 64))
    tile_val = fresh_value(ctr, tile_ty, "tile")

    # We need a "preloaded" tile — simulate a dummy load producing tile_val
    # by injecting into value_map manually.  Approach: use a custom root block
    # where tile_val is a second block param (pretend it's an MLIR value).
    # Simpler: make tile_val a block param so emit_module binds it.
    root2 = Block(params=[buf_val, tile_val])

    store_op = Store(dst=buf_val, val=tile_val, tile_shape=(64, 64), indices=(0, 0))
    store_op.results = ()
    root2.append(store_op)

    # Need to provide entry_args for both params.
    # tile_val is a register tile, not a global buffer — scalar tile arg.
    entry_args = [
        ("A", buf_val.type),
        ("tile", tile_val.type),
    ]

    module = emit_module(root2, kernel_name="store_kernel", entry_args=entry_args)
    text = str(module)

    assert "store_view_tko" in text or "store_ptr_tko" in text, f"Expected store_view_tko or store_ptr_tko in:\n{text}"


# ---------------------------------------------------------------------------
# 5. Copy (global → global)
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_copy_global_to_global():
    """Copy between two global buffers emits a load followed by a store."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, src_val, dst_val = _two_buffer_root()

    copy_op = Copy(src=src_val, dst=dst_val, tile_shape=(64, 64), src_indices=(0, 0), dst_indices=(0, 0))
    copy_op.results = ()
    root.append(copy_op)

    module = emit_module(root, kernel_name="copy_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    has_load = "load_view_tko" in text or "load_ptr_tko" in text
    has_store = "store_view_tko" in text or "store_ptr_tko" in text
    assert has_load and has_store, f"Expected both a load and a store TKO in:\n{text}"


# ---------------------------------------------------------------------------
# 6. PartitionView
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_partition_view_emits_make_partition_view():
    """PartitionView op lowers to make_partition_view."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import fresh_value

    class _Counter:
        def __init__(self):
            self._n = 0

        def next(self):
            value = self._n
            self._n += 1
            return value

    ctr = _Counter()
    root, buf_val = _single_buffer_root()
    # This unit test validates the emitted operation; a buffer-shaped value
    # supplies the result slot used by the generic test scaffold.
    result_ty = _buf_type((64, 64))
    result_val = fresh_value(ctr, result_ty, "pview")

    pv_op = PartitionView(src=buf_val, tile_shape=(64, 64))
    pv_op.results = (result_val,)
    root.append(pv_op)

    module = emit_module(root, kernel_name="pview_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "make_partition_view" in text, f"Expected make_partition_view in:\n{text}"


# ---------------------------------------------------------------------------
# 7. Token seam
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_token_seam_records_after_load():
    """After a Load, ctx has a token recorded for the src buffer param."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import fresh_value

    class _Counter:
        def __init__(self):
            self._n = 0

        def next(self):
            value = self._n
            self._n += 1
            return value

    ctr = _Counter()
    root, buf_val = _single_buffer_root()
    result_ty = _reg_type((64, 64))
    result_val = fresh_value(ctr, result_ty, "loaded")

    load_op = Load(src=buf_val, tile_shape=(64, 64), indices=(0, 0))
    load_op.results = (result_val,)
    root.append(load_op)

    _module, ctx = emit_module(
        root,
        kernel_name="tok_seam_kernel",
        entry_args=_entry_args_for(root),
        return_ctx=True,
    )

    # After a load, ctx should have recorded an output token for buf_val.
    # This is the token seam: ctx._token_map[buf_val] is not None.
    assert hasattr(ctx, "_token_map"), "EmitContext must have a _token_map dict"
    assert buf_val in ctx._token_map, f"No token recorded for buf_val after Load; _token_map={ctx._token_map}"
    assert ctx._token_map[buf_val] is not None, "Recorded token must not be None"


# ---------------------------------------------------------------------------
# 8. TmaCopy (same path as Copy, just different opcode)
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_tma_copy_same_as_copy():
    """TmaCopy follows the same emission path as Copy."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, src_val, dst_val = _two_buffer_root()

    tma_op = TmaCopy(src=src_val, dst=dst_val, tile_shape=(64, 64), src_indices=(0, 0), dst_indices=(0, 0))
    tma_op.results = ()
    root.append(tma_op)

    module = emit_module(root, kernel_name="tma_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    has_load = "load_view_tko" in text or "load_ptr_tko" in text
    has_store = "store_view_tko" in text or "store_ptr_tko" in text
    assert has_load and has_store, f"Expected load+store TKO in TmaCopy emission:\n{text}"


@skip_no_cuda_tile
def test_repeat_interleave_emits_reshape_broadcast_reshape():
    """RepeatInterleave expands one tile axis without scalar gather semantics."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    src_ty = TileType(dtype=I8, shape=(8, 4), space=MemSpace.REGISTER, layout=None)
    out_ty = TileType(dtype=I8, shape=(8, 8), space=MemSpace.REGISTER, layout=None)
    src = Value(0, src_ty, name="packed")
    out = Value(1, out_ty, name="unpacked")
    op = RepeatInterleave(src=src, axis=1, repeats=2)
    op.results = (out,)
    root = Block(params=[src])
    root.append(op)

    module, ctx = emit_module(
        root,
        kernel_name="repeat_interleave_kernel",
        entry_args=[("packed", src_ty)],
        return_ctx=True,
    )
    text = str(module)

    assert "broadcast" in text
    assert "tile<8x8xi8>" in text
    assert list(ctx.lookup(out).tile_type.shape) == [8, 8]
