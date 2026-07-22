import os
import subprocess
import sys

import pytest

import tilelang
import tilelang.language as core_language
from tilelang.language.dialect import (
    list_language_dialects,
    resolve_language_module,
)


def test_builtin_language_dialects_are_registered():
    dialects = {spec.name: spec.module for spec in list_language_dialects()}
    assert dialects["core"] == "tilelang.language"
    assert dialects["cuda"] == "tilelang.cuda.language"
    assert dialects["rocm"] == "tilelang.rocm.language"
    assert dialects["cpu"] == "tilelang.cpu.language"
    assert dialects["metal"] == "tilelang.metal.language"
    assert dialects["webgpu"] == "tilelang.webgpu.language"

    assert resolve_language_module("cu").__tilelang_dialect__ == "cuda"
    assert resolve_language_module("hip").__tilelang_dialect__ == "rocm"


def test_cuda_language_overlay_exports_core_and_cuda_symbols():
    from tilelang.cuda import language as T
    from tilelang.cuda import debug as cuda_debug

    assert T.__tilelang_dialect__ == "cuda"
    assert T.copy is core_language.copy
    assert T.tcgen05_mma is T.tcgen05_gemm
    assert T.tcgen05_mma_blockscaled is T.tcgen05_gemm_blockscaled
    assert T.wgmma_mma is T.wgmma_gemm
    assert T.device_assert is cuda_debug.device_assert
    assert hasattr(T, "TCGEN05TensorCoreIntrinEmitter")
    assert hasattr(T, "WGMMATensorCoreIntrinEmitter")


def test_rocm_language_overlay_exports_core_and_rocm_symbols():
    from tilelang.rocm import language as T

    assert T.__tilelang_dialect__ == "rocm"
    assert T.copy is core_language.copy
    assert T.mfma is T.tvm_mfma
    assert T.mfma_store is T.tvm_mfma_store
    assert hasattr(T, "MatrixCoreIntrinEmitter")
    assert hasattr(T, "make_mfma_swizzle_layout")


def test_set_default_language_dialect_updates_tilelang_language_attribute():
    desc = type(tilelang.env).__dict__["TILELANG_DEFAULT_DIALECT"]
    original = tilelang.language
    original_forced_value = desc._forced_value
    try:
        selected = tilelang.set_default_language_dialect("cuda")
        assert selected.__tilelang_dialect__ == "cuda"
        assert tilelang.language is selected

        selected = tilelang.set_default_language_dialect("core")
        assert selected.__tilelang_dialect__ == "core"
        assert tilelang.language is selected

        selected = core_language.set_default_language_dialect("cuda")
        assert selected.__tilelang_dialect__ == "cuda"
        assert tilelang.language is selected
    finally:
        desc._forced_value = original_forced_value
        tilelang.language = original


def test_default_language_dialect_env_controls_from_tilelang_import_language():
    script = """
from tilelang import language as T
print("dialect", getattr(T, "__tilelang_dialect__", "missing"))
print("module", T.__name__)
print("has_tcgen05_mma", hasattr(T, "tcgen05_mma"))
"""
    env = dict(os.environ)
    env["TILELANG_DEFAULT_DIALECT"] = "cuda"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "dialect cuda" in result.stdout
    assert "module tilelang.cuda.language" in result.stdout
    assert "has_tcgen05_mma True" in result.stdout


def test_unknown_language_dialect_fails_clearly():
    with pytest.raises(ValueError, match="Unknown TileLang language dialect"):
        resolve_language_module("not_a_backend")
