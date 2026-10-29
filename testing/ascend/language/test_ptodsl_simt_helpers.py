"""TileLang helpers must not mutate the native PTODSL scalar operators."""

import importlib

import pytest


def test_simt_import_and_reload_preserve_native_unsigned_select():
    pytest.importorskip("ptodsl")
    from ptodsl import _scalar, pto
    from ptodsl._context import make_context
    from ptodsl._surface_values import unwrap_surface_value
    from ptoas.mlir.ir import InsertionPoint, Location, Module

    native_select = _scalar.select
    simt = importlib.import_module("tilelang.contrib.ptodsl.simt")
    assert _scalar.select is native_select
    importlib.reload(simt)
    assert _scalar.select is native_select

    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            cond = pto.const(1, dtype=pto.i1)
            lhs = pto.const(0xFFFFFFFF, dtype=pto.ui32)
            rhs = pto.const(1, dtype=pto.ui32)
            selected = _scalar.select(cond, lhs, rhs)
            assert str(unwrap_surface_value(selected).type) == "ui32"
            assert str(unwrap_surface_value(pto.min(selected, rhs)).type) == "ui32"
