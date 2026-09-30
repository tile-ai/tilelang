"""PTO target selection reaches the PTO device code generator."""

import pytest

import tilelang
from tilelang.ascend import language as T
from tilelang.backend import create_backend_context


@T.prim_func
def copy_kernel(src: T.Tensor((32,), "float32"), dst: T.Tensor((32,), "float32")):
    with T.Kernel(1):
        buf = T.alloc_shared((32,), "float32")
        T.copy(src, buf)
        T.copy(buf, dst)


def test_pto_target_selects_codegen_and_execution_backend():
    context = create_backend_context("pto", "c", "auto")
    assert context.target.kind.name == "ascend"
    assert "pto" in context.target.keys
    assert context.module.name == "pto"
    assert context.module.get_device_codegen(context.target).name == "pto"
    assert context.execution_backend.name == "pto"

    with pytest.raises(ValueError, match="Invalid execution backend"):
        create_backend_context("ascend", "c", "pto")


def test_pto_lower_emits_ptodsl_source():
    ascend_source = tilelang.lower(copy_kernel, target="ascend").kernel_source
    pto_source = tilelang.lower(copy_kernel, target="pto").kernel_source
    pto_source_with_compile = tilelang.lower(copy_kernel, target="pto", enable_device_compile=True).kernel_source

    assert "from ptodsl import pto" not in ascend_source
    assert "from ptodsl import pto" in pto_source
    assert "@pto.jit(" in pto_source
    assert pto_source_with_compile == pto_source
