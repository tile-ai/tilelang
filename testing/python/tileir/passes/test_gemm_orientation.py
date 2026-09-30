"""Tests for Hopper GEMM orientation selection."""

from __future__ import annotations

from tilelang.tileir.ir.ops import Gemm
from tilelang.tileir.ir.types import MemSpace, TileType, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.passes.base import PassContext
from tilelang.tileir.passes.gemm_orientation import gemm_orientation_pass


def _buffer(value_id: int, name: str, shape: tuple[int, int], dtype_name: str) -> Value:
    return Value(
        value_id,
        TileType(dtype(dtype_name), shape, MemSpace.REGISTER, layout=None),
        name=name,
    )


def _gemm_root(m: int, n: int, k: int = 64, input_dtype: str = "float16") -> tuple[Block, Gemm]:
    lhs = _buffer(0, "lhs", (m, k), input_dtype)
    rhs = _buffer(1, "rhs", (k, n), input_dtype)
    acc = _buffer(2, "acc", (m, n), "float32")
    op = Gemm(lhs=lhs, rhs=rhs, acc=acc)
    root = Block()
    root.append(op)
    return root, op


def _run(root: Block, arch: str | None) -> PassContext:
    ctx = PassContext()
    ctx.results["target_arch"] = arch
    gemm_orientation_pass(root, ctx)
    return ctx


def test_hopper_skinny_m_gemm_uses_swapped_orientation():
    root, op = _gemm_root(m=16, n=128)

    ctx = _run(root, "sm_90")

    assert op.swap_ab is True
    assert ctx.results["gemm_orientation"]["swapped"] == 1


def test_hopper_fp8_m64_wide_gemm_uses_swapped_orientation():
    root, op = _gemm_root(m=64, n=128, input_dtype="float8_e4m3fn")

    ctx = _run(root, "sm_90a")

    assert op.swap_ab is True
    assert ctx.results["gemm_orientation"]["swapped"] == 1


def test_hopper_fp16_m64_wide_gemm_keeps_native_orientation():
    root, op = _gemm_root(m=64, n=128)

    ctx = _run(root, "sm_90a")

    assert op.swap_ab is False
    assert ctx.results["gemm_orientation"]["swapped"] == 0


def test_hopper_non_skinny_gemm_keeps_native_orientation():
    root, op = _gemm_root(m=128, n=128)

    ctx = _run(root, "sm_90")

    assert op.swap_ab is False
    assert ctx.results["gemm_orientation"]["swapped"] == 0


def test_non_hopper_gemm_keeps_native_orientation():
    root, op = _gemm_root(m=16, n=128)

    _run(root, "sm_100")

    assert op.swap_ab is False


def test_missing_arch_keeps_native_orientation():
    root, op = _gemm_root(m=16, n=128)

    _run(root, None)

    assert op.swap_ab is False
