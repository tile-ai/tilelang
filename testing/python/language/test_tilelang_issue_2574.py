"""Regression test for issue #2574.

T.atomic_load with memory_order="release"/"acq_rel" and T.atomic_store
with memory_order="consume"/"acquire"/"acq_rel" should raise a clear
ValueError instead of compiling a kernel that dies with a device-side assert.
"""

import pytest
import tilelang
import tilelang.language as T


def test_atomic_load_rejects_release():
    @T.prim_func
    def main(Flag: T.Tensor((1,), "int32"), Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1) as bx:
            Out[0] = T.atomic_load(Flag[0], memory_order="release")

    with pytest.raises(ValueError, match="illegal for a load"):
        tilelang.compile(main, target="cuda")


def test_atomic_load_rejects_acq_rel():
    @T.prim_func
    def main(Flag: T.Tensor((1,), "int32"), Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1) as bx:
            Out[0] = T.atomic_load(Flag[0], memory_order="acq_rel")

    with pytest.raises(ValueError, match="illegal for a load"):
        tilelang.compile(main, target="cuda")


@pytest.mark.parametrize("order", ["consume", "acquire", "acq_rel"])
def test_atomic_store_rejects_illegal_orders(order):
    @T.prim_func
    def main(Flag: T.Tensor((1,), "int32"), Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1) as bx:
            T.atomic_store(Flag[0], 42, memory_order=order)
            Out[0] = Flag[0]

    with pytest.raises(ValueError, match="illegal for a store"):
        tilelang.compile(main, target="cuda")


@pytest.mark.parametrize("order", ["relaxed", "acquire", "seq_cst"])
def test_atomic_load_accepts_legal_orders(order):
    @T.prim_func
    def main(Flag: T.Tensor((1,), "int32"), Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1) as bx:
            Out[0] = T.atomic_load(Flag[0], memory_order=order)

    # Should NOT raise
    tilelang.compile(main, target="cuda")


@pytest.mark.parametrize("order", ["relaxed", "release", "seq_cst"])
def test_atomic_store_accepts_legal_orders(order):
    @T.prim_func
    def main(Flag: T.Tensor((1,), "int32"), Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1) as bx:
            T.atomic_store(Flag[0], 42, memory_order=order)
            Out[0] = Flag[0]

    # Should NOT raise
    tilelang.compile(main, target="cuda")
