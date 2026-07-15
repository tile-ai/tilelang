"""Emission tests for TileIR atomic operations."""

from __future__ import annotations

import pytest

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import AtomicRMW, AtomicCAS
from tilelang.tileir.errors import _UnsupportedTileIRNode
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

I32 = dtype("int32")
FP32 = dtype("float32")
TILE_SHAPE = (64,)


def _global_buf(elem=I32, shape=TILE_SHAPE) -> TileType:
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.GLOBAL, layout=None)


def _reg_tile(elem=I32, shape=TILE_SHAPE) -> TileType:
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


def _build_atomic_rmw_root(kind: str, dst_elem=I32, val_elem=I32, shape=TILE_SHAPE):
    """Return (root, entry_args) for a single AtomicRMW op."""
    dst_val = Value(0, _global_buf(dst_elem, shape), name="dst")
    val_val = Value(1, _reg_tile(val_elem, shape), name="val")

    root = Block(params=[dst_val, val_val])
    entry_args = [
        ("dst", dst_val.type),
        ("val", val_val.type),
    ]

    op = AtomicRMW(dst=dst_val, val=val_val, kind=kind)
    op.results = ()
    root.append(op)

    return root, entry_args


def _build_atomic_cas_root(dst_elem=I32, shape=TILE_SHAPE):
    """Return (root, entry_args, result_val) for a single AtomicCAS op."""
    dst_val = Value(0, _global_buf(dst_elem, shape), name="dst")
    expected_val = Value(1, _reg_tile(dst_elem, shape), name="expected")
    desired_val = Value(2, _reg_tile(dst_elem, shape), name="desired")

    root = Block(params=[dst_val, expected_val, desired_val])
    entry_args = [
        ("dst", dst_val.type),
        ("expected", expected_val.type),
        ("desired", desired_val.type),
    ]

    # AtomicCAS returns the old value.
    result_val = Value(3, _reg_tile(dst_elem, shape), name="old_val")
    op = AtomicCAS(dst=dst_val, expected=expected_val, desired=desired_val)
    op.results = (result_val,)
    root.append(op)

    return root, entry_args, result_val


# Relaxed atomics on view-backed global buffers use the partition-view
# reduction path. Verify its operation and encoded reduction mode.


def _atomic_mode_int(name: str) -> int:
    """Return the integer ``mode`` attribute the dialect prints for an
    ``AtomicRMWMode`` member (e.g. ADD/ADDF/MAX/MIN)."""
    from cuda_tile._mlir.dialects import cuda_tile as ct

    member = getattr(ct.AtomicRMWMode, name)
    # The MLIR text prints the enum as its underlying integer; recover it via
    # the ordering of the enum so the assertion is robust to dialect changes.
    return list(type(member).__members__.values()).index(member)


def _assert_atomic_red_view_mode(text: str, mode_name: str):
    assert "atomic_red_view_tko" in text, f"Expected 'atomic_red_view_tko' (relaxed view-backed atomic) in MLIR:\n{text}"
    assert "atomic_rmw_tko" not in text, f"Did not expect pointer-based 'atomic_rmw_tko' for a view-backed relaxed atomic:\n{text}"
    mode_int = _atomic_mode_int(mode_name)
    assert f"mode = {mode_int}" in text, f"Expected 'mode = {mode_int}' ({mode_name}) in MLIR:\n{text}"


@skip_no_cuda_tile
def test_atomic_rmw_add():
    """AtomicRMW kind='add' on int32 dst emits atomic_red_view_tko with ADD mode."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_atomic_rmw_root(kind="add")
    module = emit_module(root, kernel_name="atomic_add", entry_args=entry_args)
    _assert_atomic_red_view_mode(str(module), "ADD")


@skip_no_cuda_tile
def test_atomic_rmw_max():
    """AtomicRMW kind='max' on int32 dst emits atomic_red_view_tko with MAX mode."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_atomic_rmw_root(kind="max")
    module = emit_module(root, kernel_name="atomic_max", entry_args=entry_args)
    _assert_atomic_red_view_mode(str(module), "MAX")


@skip_no_cuda_tile
def test_atomic_rmw_min():
    """AtomicRMW kind='min' on int32 dst emits atomic_red_view_tko with MIN mode."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_atomic_rmw_root(kind="min")
    module = emit_module(root, kernel_name="atomic_min", entry_args=entry_args)
    _assert_atomic_red_view_mode(str(module), "MIN")


@skip_no_cuda_tile
def test_atomic_rmw_add_float():
    """AtomicRMW kind='add' on fp32 dst uses the ADDF (float) reduction mode."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_atomic_rmw_root(kind="add", dst_elem=FP32, val_elem=FP32)
    module = emit_module(root, kernel_name="atomic_addf", entry_args=entry_args)
    _assert_atomic_red_view_mode(str(module), "ADDF")


def test_atomic_rmw_unsupported_kind():
    """AtomicRMW with an unknown kind raises _UnsupportedTileIRNode (not NotImplementedError)."""
    if not _HAS_CUDA_TILE:
        pytest.skip("cuda_tile unavailable; skip full integration check")
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_atomic_rmw_root(kind="xor")  # unsupported
    with pytest.raises(_UnsupportedTileIRNode, match="unsupported kind"):
        emit_module(root, kernel_name="atomic_bad", entry_args=entry_args)


@skip_no_cuda_tile
def test_atomic_cas():
    """AtomicCAS emits the atomic_cas mnemonic (cuda_tile dialect has atomic_cas_tko)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, _result_val = _build_atomic_cas_root()
    module = emit_module(root, kernel_name="atomic_cas", entry_args=entry_args)
    text = str(module)

    assert "atomic_cas" in text, f"Expected 'atomic_cas' in MLIR output:\n{text}"


@skip_no_cuda_tile
def test_atomic_cas_result_bound():
    """AtomicCAS result (old value) is bound in the EmitContext after emission."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args, result_val = _build_atomic_cas_root()
    module, ctx = emit_module(root, kernel_name="atomic_cas_bind", entry_args=entry_args, return_ctx=True)

    # The result Value should be bound in the value_map.
    assert result_val in ctx.value_map, f"Expected AtomicCAS result Value id={result_val.id} in ctx.value_map after emission"
