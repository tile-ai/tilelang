"""Language manifests and default overrides must survive cold import order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("backend", ["common", "cpu", "cuda", "rocm", "metal", "webgpu", "ascend"])
@pytest.mark.parametrize("backend_first", [False, True])
def test_cold_language_import_order(backend, backend_first):
    name = "tilelang.language.common" if backend == "common" else f"tilelang.{backend}.language"
    imports = ["tilelang.language", name]
    if backend_first:
        imports.reverse()
    source = f"""
import importlib
for name in {imports!r}:
    importlib.import_module(name)
import tilelang.language as T
import tilelang.cuda.language as C
assert T.Kernel is C.Kernel
assert T.copy is C.copy
assert T.Parallel is C.Parallel
assert T.gemm is C.gemm
assert T.__all__ == C.__all__
assert set(T.__all__) <= set(dir(T))
assert T.__tilelang_dialect__ == "cuda"
assert "Tensor" in T.__all__
assert "Kernel" in T.__all__
assert "alloc_tmem" in T.__all__
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["tilelang.utils.language", "tilelang._typing", "tilelang.language.proxy", "tilelang.language.allocate"])
def test_cold_language_helper_import(name):
    source = f"""
import importlib
importlib.import_module({name!r})
import tilelang.language as T
import tilelang.cuda.language as C
assert T.Kernel is C.Kernel
assert T.copy is C.copy
assert T.__all__ == C.__all__
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
