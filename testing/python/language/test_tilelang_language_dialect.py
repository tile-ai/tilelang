import importlib

import pytest

import tilelang
import tilelang.language as default_language
import tilelang.language.common as common_language


CUDA_ONLY_NAMES = {
    "ClusterKernel",
    "CUDASourceCodeKernel",
    "pdl_trigger",
    "rng_init",
    "tcgen05_mma",
    "tma_copy",
    "wgmma_mma",
    "ws",
}

ROCM_ONLY_NAMES = {
    "MatrixCoreIntrinEmitter",
    "mfma",
    "rdna_wmma",
    "tvm_mfma",
}


def test_default_language_is_static_cuda_facade():
    from tilelang.cuda import language as cuda_language

    assert tilelang.language is default_language
    assert importlib.import_module("tilelang.language") is default_language
    assert default_language.__tilelang_dialect__ == "cuda"
    assert cuda_language.__tilelang_dialect__ == "cuda"
    assert set(default_language.__all__) == set(cuda_language.__all__)

    for name in default_language.__all__:
        assert getattr(default_language, name) is getattr(cuda_language, name)


def test_common_language_preserves_special_dsl_exports():
    assert common_language.__tilelang_dialect__ == "common"
    assert "__log" in common_language.__all__
    assert hasattr(common_language, "__log")


def test_cuda_language_composes_common_and_cuda_symbols():
    from tilelang.cuda import language as T
    from tilelang.cuda import debug as cuda_debug

    assert T.copy is common_language.copy
    assert T.tcgen05_mma is T.tcgen05_gemm
    assert T.tcgen05_mma_blockscaled is T.tcgen05_gemm_blockscaled
    assert T.wgmma_mma is T.wgmma_gemm
    assert T.device_assert is cuda_debug.device_assert
    assert set(T.__all__) >= CUDA_ONLY_NAMES
    assert hasattr(T, "TCGEN05TensorCoreIntrinEmitter")
    assert hasattr(T, "WGMMATensorCoreIntrinEmitter")


def test_rocm_language_composes_common_and_rocm_symbols():
    from tilelang.rocm import language as T

    assert T.__tilelang_dialect__ == "rocm"
    assert T.copy is common_language.copy
    assert T.mfma is T.tvm_mfma
    assert T.mfma_store is T.tvm_mfma_store
    assert set(T.__all__) >= ROCM_ONLY_NAMES
    assert hasattr(T, "MatrixCoreIntrinEmitter")
    assert hasattr(T, "make_mfma_swizzle_layout")


@pytest.mark.parametrize(
    "module_name",
    [
        "tilelang.cpu.language",
        "tilelang.metal.language",
        "tilelang.rocm.language",
        "tilelang.webgpu.language",
    ],
)
def test_non_cuda_dialects_do_not_export_cuda_symbols(module_name):
    module = importlib.import_module(module_name)
    assert CUDA_ONLY_NAMES.isdisjoint(module.__all__)


@pytest.mark.parametrize(
    "module_name",
    [
        "tilelang.cpu.language",
        "tilelang.cuda.language",
        "tilelang.metal.language",
        "tilelang.webgpu.language",
    ],
)
def test_non_rocm_dialects_do_not_export_rocm_symbols(module_name):
    module = importlib.import_module(module_name)
    assert ROCM_ONLY_NAMES.isdisjoint(module.__all__)


def test_common_only_dialects_match_common_surface():
    from tilelang.cpu import language as cpu_language
    from tilelang.webgpu import language as webgpu_language

    assert set(cpu_language.__all__) == set(common_language.__all__)
    assert set(webgpu_language.__all__) == set(common_language.__all__)


def test_cuda_whole_module_implementations_live_under_cuda():
    from tilelang.cuda import language as T

    assert T.pdl_trigger.__module__ == "tilelang.cuda.language.pdl"
    assert T.cluster_sync.__module__ == "tilelang.cuda.language.cluster"
    assert T.rng_init.__module__ == "tilelang.cuda.language.random"
    assert T.ws.__module__ == "tilelang.cuda.language.warpgroup"


@pytest.mark.parametrize(
    ("legacy_module", "cuda_module", "symbol"),
    [
        ("tilelang.language.cluster", "tilelang.cuda.language.cluster", "cluster_sync"),
        ("tilelang.language.pdl", "tilelang.cuda.language.pdl", "pdl_trigger"),
        ("tilelang.language.print_op", "tilelang.cuda.language.print", "print"),
        ("tilelang.language.random", "tilelang.cuda.language.random", "rng_init"),
        ("tilelang.language.warpgroup", "tilelang.cuda.language.warpgroup", "ws"),
    ],
)
def test_legacy_cuda_module_paths_are_identity_preserving_wrappers(legacy_module, cuda_module, symbol):
    legacy = importlib.import_module(legacy_module)
    cuda = importlib.import_module(cuda_module)
    assert getattr(legacy, symbol) is getattr(cuda, symbol)
