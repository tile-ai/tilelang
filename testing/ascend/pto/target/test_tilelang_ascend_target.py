import pytest

from tilelang import tvm
from tilelang.ascend.codegen import PTO_CODEGEN
from tilelang.backend import create_backend_context
from tilelang.ascend.target import (
    target_is_pto,
)


@pytest.mark.pto
def test_pto_execution_backend_selects_pto_target_variant():
    context = create_backend_context("ascend", "c", "pto")

    assert context.name == "pto"
    assert context.target.kind.name == "ascend"
    assert target_is_pto(context.target)
    assert context.execution_backend.name == "pto"
    assert context.module.get_device_codegen(context.target) is PTO_CODEGEN
    assert tvm.ffi.get_global_func("target.build.tilelang_pto_without_compile", allow_missing=True) is not None


@pytest.mark.pto
def test_pto_execution_backend_preserves_ascend_target_options():
    context = create_backend_context({"kind": "ascend", "arch": "dav-3510"}, "c", "pto")

    assert context.name == "pto"
    assert str(context.target.attrs["arch"]) == "dav-3510"
    assert target_is_pto(context.target)
