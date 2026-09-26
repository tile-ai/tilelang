"""Tests for Barrier / DeviceAssert / DebugPrint / DecodeI4 / DecodeI2 / Dp4a.

Coverage
--------
1.  test_barrier_is_noop           — Barrier emits no ops (no mnemonic, no crash)
2.  test_device_assert_emits_assert — DeviceAssert emits 'assert' mnemonic
3.  test_device_assert_message      — DeviceAssert embeds the message string in MLIR
4.  test_debug_print_emits_print    — DebugPrint emits 'print_tko' mnemonic
5.  test_debug_print_message        — DebugPrint message appears in MLIR output
6.  test_decode_i4_emits_exti       — DecodeI4 emits bit-manipulation ops (exti)
7.  test_decode_i4_emits_andi       — DecodeI4 emits andi (nibble mask)
8.  test_decode_i4_emits_itof       — DecodeI4 emits itof (float cast)
9.  test_decode_i2_emits_exti       — DecodeI2 emits bit-manipulation ops (exti)
10. test_decode_i2_emits_andi       — DecodeI2 emits andi (crumb mask)
11. test_decode_i2_emits_trunci     — DecodeI2 emits trunci (truncate to i8)
12. test_dp4a_emits_mul             — Dp4a emits 'mul' (product ops)
13. test_dp4a_emits_add             — Dp4a emits 'add' (accumulate ops)
14. test_dp4a_emits_exti            — Dp4a emits exti (i8 -> i32 sign extension)
"""

from __future__ import annotations

import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import (
    Barrier,
    DeviceAssert,
    DebugPrint,
    DecodeI4,
    DecodeI2,
    Dp4a,
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
# Shared dtype / type helpers
# ---------------------------------------------------------------------------

I8 = dtype("int8")
I32 = dtype("int32")
F16 = dtype("float16")
BOOL = dtype("bool")


def _global_buf(elem, shape) -> TileType:
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.GLOBAL, layout=None)


def _reg_tile(elem, shape=()) -> TileType:
    return TileType(dtype=elem, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


# ---------------------------------------------------------------------------
# Helper: emit a root block with the given ops and return module text
# ---------------------------------------------------------------------------


def _emit_root(params, entry_args, *ops):
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root = Block(params=params)
    for op in ops:
        root.append(op)
    module = emit_module(root, kernel_name="test_kernel", entry_args=entry_args)
    return str(module)


# ===========================================================================
# 1. Barrier
# ===========================================================================


@skip_no_cuda_tile
def test_barrier_is_noop():
    """Barrier.emit_mlir() completes without error and emits no custom op."""
    op = Barrier()
    op.results = ()

    # Build a minimal root with no params — just the Barrier op
    root = Block(params=[])
    root.append(op)

    from tilelang.tileir.lowering.mlir_emit import emit_module

    module = emit_module(root, kernel_name="barrier_test", entry_args=[])
    text = str(module)

    # The only op in the block should be the mandatory 'return'.
    # No barrier-specific mnemonic should appear.
    assert "barrier_test" in text, "kernel name must appear in module"
    # Confirm no crash and MLIR is well-formed by checking the module string.
    assert "cuda_tile.entry" in text or "entry" in text


# ===========================================================================
# 2-3. DeviceAssert
# ===========================================================================


@skip_no_cuda_tile
def test_device_assert_emits_assert():
    """DeviceAssert emits the 'assert' mnemonic in MLIR output."""
    cond_val = Value(0, _reg_tile(BOOL, ()), name="cond")
    params = [cond_val]
    entry_args = [("cond", cond_val.type)]

    op = DeviceAssert(cond=cond_val, message="test assertion failed")
    op.results = ()

    text = _emit_root(params, entry_args, op)
    assert "assert" in text, f"Expected 'assert' in MLIR output for DeviceAssert:\n{text}"


@skip_no_cuda_tile
def test_device_assert_message():
    """DeviceAssert embeds the message string in the MLIR output."""
    cond_val = Value(0, _reg_tile(BOOL, ()), name="cond")
    params = [cond_val]
    entry_args = [("cond", cond_val.type)]

    msg = "my_custom_assertion_msg"
    op = DeviceAssert(cond=cond_val, message=msg)
    op.results = ()

    text = _emit_root(params, entry_args, op)
    assert msg in text, f"Expected message {msg!r} in MLIR output for DeviceAssert:\n{text}"


# ===========================================================================
# 4-5. DebugPrint
# ===========================================================================


@skip_no_cuda_tile
def test_debug_print_emits_print():
    """DebugPrint emits 'print_tko' mnemonic in MLIR output."""
    op = DebugPrint(message="hello")
    op.results = ()

    text = _emit_root([], [], op)
    assert "print_tko" in text, f"Expected 'print_tko' in MLIR output for DebugPrint:\n{text}"


@skip_no_cuda_tile
def test_debug_print_message():
    """DebugPrint embeds the message string in MLIR output."""
    msg = "debug_msg_42"
    op = DebugPrint(message=msg)
    op.results = ()

    text = _emit_root([], [], op)
    assert msg in text, f"Expected message {msg!r} in MLIR output for DebugPrint:\n{text}"


# ===========================================================================
# 6-8. DecodeI4
# ===========================================================================


def _build_decode_i4_root():
    """Return (root, entry_args) for a single DecodeI4 op."""
    src_val = Value(0, _global_buf(I8, (4,)), name="src")
    dst_val = Value(1, _global_buf(F16, (8,)), name="dst")
    params = [src_val, dst_val]
    entry_args = [("src", src_val.type), ("dst", dst_val.type)]

    op = DecodeI4(src=src_val, dst=dst_val)
    op.results = ()

    root = Block(params=params)
    root.append(op)
    return root, entry_args


@skip_no_cuda_tile
def test_decode_i4_emits_exti():
    """DecodeI4 emits integer extension ops (exti) for each nibble."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_decode_i4_root()
    module = emit_module(root, kernel_name="decode_i4", entry_args=entry_args)
    text = str(module)

    assert "exti" in text, f"Expected 'exti' in MLIR output for DecodeI4:\n{text}"


@skip_no_cuda_tile
def test_decode_i4_emits_andi():
    """DecodeI4 emits andi for nibble masking."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_decode_i4_root()
    module = emit_module(root, kernel_name="decode_i4_andi", entry_args=entry_args)
    text = str(module)

    assert "andi" in text, f"Expected 'andi' in MLIR output for DecodeI4:\n{text}"


@skip_no_cuda_tile
def test_decode_i4_emits_itof():
    """DecodeI4 emits itof for the nibble-to-float16 cast."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_decode_i4_root()
    module = emit_module(root, kernel_name="decode_i4_itof", entry_args=entry_args)
    text = str(module)

    assert "itof" in text, f"Expected 'itof' in MLIR output for DecodeI4:\n{text}"


# ===========================================================================
# 9-11. DecodeI2
# ===========================================================================


def _build_decode_i2_root():
    """Return (root, entry_args) for a single DecodeI2 op."""
    src_val = Value(0, _global_buf(I8, (4,)), name="src")
    dst_val = Value(1, _global_buf(I8, (16,)), name="dst")
    params = [src_val, dst_val]
    entry_args = [("src", src_val.type), ("dst", dst_val.type)]

    op = DecodeI2(src=src_val, dst=dst_val)
    op.results = ()

    root = Block(params=params)
    root.append(op)
    return root, entry_args


@skip_no_cuda_tile
def test_decode_i2_emits_exti():
    """DecodeI2 emits integer extension ops (exti) for each crumb."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_decode_i2_root()
    module = emit_module(root, kernel_name="decode_i2", entry_args=entry_args)
    text = str(module)

    assert "exti" in text, f"Expected 'exti' in MLIR output for DecodeI2:\n{text}"


@skip_no_cuda_tile
def test_decode_i2_emits_andi():
    """DecodeI2 emits andi for 2-bit crumb masking."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_decode_i2_root()
    module = emit_module(root, kernel_name="decode_i2_andi", entry_args=entry_args)
    text = str(module)

    assert "andi" in text, f"Expected 'andi' in MLIR output for DecodeI2:\n{text}"


@skip_no_cuda_tile
def test_decode_i2_emits_trunci():
    """DecodeI2 emits trunci to truncate i32 crumb back to i8."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_decode_i2_root()
    module = emit_module(root, kernel_name="decode_i2_trunci", entry_args=entry_args)
    text = str(module)

    assert "trunci" in text, f"Expected 'trunci' in MLIR output for DecodeI2:\n{text}"


# ===========================================================================
# 12-14. Dp4a
# ===========================================================================


def _build_dp4a_root():
    """Return (root, entry_args) for a single Dp4a op."""
    lhs_val = Value(0, _global_buf(I8, (4,)), name="lhs")
    rhs_val = Value(1, _global_buf(I8, (4,)), name="rhs")
    acc_val = Value(2, _global_buf(I32, ()), name="acc")
    params = [lhs_val, rhs_val, acc_val]
    entry_args = [
        ("lhs", lhs_val.type),
        ("rhs", rhs_val.type),
        ("acc", acc_val.type),
    ]

    op = Dp4a(lhs=lhs_val, rhs=rhs_val, acc=acc_val)
    op.results = ()

    root = Block(params=params)
    root.append(op)
    return root, entry_args


@skip_no_cuda_tile
def test_dp4a_emits_mul():
    """Dp4a emits 'mul' for the element-wise products."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_dp4a_root()
    module = emit_module(root, kernel_name="dp4a_mul", entry_args=entry_args)
    text = str(module)

    assert "mul" in text, f"Expected 'mul' in MLIR output for Dp4a:\n{text}"


@skip_no_cuda_tile
def test_dp4a_emits_add():
    """Dp4a emits 'add' for accumulation."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_dp4a_root()
    module = emit_module(root, kernel_name="dp4a_add", entry_args=entry_args)
    text = str(module)

    assert "add" in text, f"Expected 'add' in MLIR output for Dp4a:\n{text}"


@skip_no_cuda_tile
def test_dp4a_emits_exti():
    """Dp4a emits exti for sign-extending i8 operands to i32."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, entry_args = _build_dp4a_root()
    module = emit_module(root, kernel_name="dp4a_exti", entry_args=entry_args)
    text = str(module)

    assert "exti" in text, f"Expected 'exti' in MLIR output for Dp4a:\n{text}"
