"""The callable T.allocate survives imports of its implementation module."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import tvm_tirx

import tilelang
import tilelang.language as T


@pytest.mark.parametrize(
    "first",
    [
        "tilelang.language",
        "tilelang.language.allocate",
        "tilelang.language.common",
        "tilelang.cpu.language",
        "tilelang.cuda.language",
        "tilelang.layout.ascend",
        "tilelang._typing",
        "tilelang.language._allocation",
        "tilelang.language.proxy",
        "tilelang.language.eager.builder",
        "tilelang.engine.param",
        "tilelang.ascend.language",
        "tilelang.rocm.language",
        "tilelang.metal.language",
        "tilelang.webgpu.language",
    ],
)
def test_allocate_callable_with_helper_and_dialect_import_orders(first):
    source = f"""
        import importlib
        importlib.import_module({first!r})
        import tilelang.language as T
        assert callable(T.allocate)
        helper = importlib.import_module("tilelang.language.allocate")
        assert callable(helper.alloc_local)
        assert callable(T.allocate)
        import tvm_ffi
        assert tvm_ffi.get_global_func("tl.gemm.lower", allow_missing=True) is None
        assert "tilelang.cuda.op.gemm.gemm_mma" not in __import__("sys").modules
        from tilelang.cuda import language as cuda
        from tilelang.language import common
        assert T.allocate is cuda.allocate is common.allocate
        from tilelang.language import allocate
        assert allocate is T.allocate
    """
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_allocate_keeps_native_storage_and_cpu_numerics():
    @T.prim_func
    def main(a: T.Tensor((8,), "int32"), b: T.Tensor((8,), "int32")):
        temporary = T.allocate((8,), "int32", scope="local")
        for index in T.serial(8):
            temporary[index] = a[index] + 5
        for index in T.serial(8):
            b[index] = temporary[index] * 2

    executable = tilelang.compile(main, target="c", execution_backend="tvm_ffi")
    source = np.arange(8, dtype="int32") - 3
    a = tvm_tirx.runtime.tensor(source)
    b = tvm_tirx.runtime.tensor(np.full(8, -999, dtype="int32"))
    executable(a, b)
    np.testing.assert_array_equal(b.numpy(), (source + 5) * 2)


def test_allocate_helper_reload_preserves_callable_and_explicit_user_override():
    source = """
        import importlib
        import tilelang.language as T
        original = T.allocate
        custom = lambda *args, **kwargs: "custom"
        T.allocate = custom
        helper = importlib.import_module("tilelang.language.allocate")
        assert T.allocate is custom
        importlib.reload(helper)
        assert T.allocate is custom
        T.allocate = original
        assert callable(helper.alloc_shared)
        from tilelang.cuda import language as cuda
        from tilelang.language import common
        assert T.allocate is cuda.allocate is common.allocate
        importlib.reload(T)
        assert T.allocate is cuda.allocate is common.allocate
        assert callable(importlib.import_module("tilelang.language.allocate").alloc_var)
    """
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
