"""Trace the public PTO select boundary without compiling or launching NPU code."""

import pytest

pytestmark = pytest.mark.pto

pytest.importorskip("ptodsl")

import tilelang.contrib.ptodsl as tl
from ptodsl import pto


@pytest.mark.parametrize("condition", [False, True])
def test_public_select_materializes_static_bool_as_i1(condition):
    @pto.jit(kernel_kind="vector", mode="explicit")
    def kernel(output: pto.ptr(pto.f32, "gm")):
        pto.init_core()
        with pto.simt(1, 1, 1):
            selected = pto.select(tl.as_logical_bool(condition), pto.const(3.0, dtype=pto.f32), pto.const(4.0, dtype=pto.f32))
            pto.store(selected, output, 0)

    module = kernel.compile()
    text = module.mlir_text()
    assert "pto.select" in text
    assert f"pto.constant {'true' if condition else 'false'}" in text


def test_public_select_preserves_unsigned_runtime_branch_type():
    @pto.jit(kernel_kind="vector", mode="explicit")
    def kernel(source: pto.ptr(pto.ui32, "gm"), output: pto.ptr(pto.ui32, "gm")):
        pto.init_core()
        with pto.simt(1, 1, 1):
            value = pto.load(source, 0)
            selected = pto.select(value > pto.const(0, dtype=pto.ui32), value, pto.const(0xFFFFFFFF, dtype=pto.ui32))
            pto.store(selected // pto.const(2, dtype=pto.ui32), output, 0)

    text = kernel.compile().mlir_text()
    assert "pto.select" in text
    assert "ui32" in text
