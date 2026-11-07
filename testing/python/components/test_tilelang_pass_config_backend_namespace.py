"""Tests for backend-owned pass-config namespaces.

The CUDA-owned members are exposed from ``tilelang.cuda`` while the historical
``tilelang.transform.PassConfigKey`` spellings remain available during the
introduction release.  Both spellings must normalize to the same string key so
that pass contexts, JIT kernels, and cache keys are unaffected.
"""

import warnings
from enum import Enum

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cache.kernel_cache import KernelCache
from tilelang.cuda import PassConfigKey as CudaPassConfigKey
from tilelang.transform.pass_config import PassConfigKey, normalize_pass_configs


EXPECTED_CUDA_MEMBERS = {
    "TL_CUDA_COMPILER",
    "TL_PTXAS_REGISTER_USAGE_LEVEL",
    "TL_ENABLE_FP32X2_REDUCTION",
    "TL_DISABLE_WGMMA",
    "TL_DISABLE_SHUFFLE_ELECT",
    "CUDA_KERNELS_OUTPUT_DIR",
}
CUDA_OWNED_MEMBERS = list(CudaPassConfigKey.__members__)


def test_cuda_namespace_exports_a_static_enum():
    import tilelang.cuda.pass_config as cuda_pass_config

    assert CudaPassConfigKey is cuda_pass_config.PassConfigKey
    assert issubclass(CudaPassConfigKey, str)
    assert set(CUDA_OWNED_MEMBERS) == EXPECTED_CUDA_MEMBERS


def test_every_cuda_member_has_a_common_compatibility_alias():
    for name, member in CudaPassConfigKey.__members__.items():
        assert name in PassConfigKey.__members__
        assert member.value == PassConfigKey[name].value


def test_legacy_and_cuda_keys_normalize_identically():
    for name in CUDA_OWNED_MEMBERS:
        legacy = normalize_pass_configs({PassConfigKey[name]: 1})
        cuda = normalize_pass_configs({CudaPassConfigKey[name]: 1})

        assert legacy == cuda
        assert legacy[CudaPassConfigKey[name].value] == 1


def test_common_and_cuda_keys_coexist():
    normalized = normalize_pass_configs(
        {
            PassConfigKey.TL_ENABLE_FAST_MATH: True,
            CudaPassConfigKey.TL_CUDA_COMPILER: "nvrtc",
            "tirx.disable_cse_tir": True,
        }
    )

    assert normalized["tl.enable_fast_math"] is True
    assert normalized["tl.cuda_compiler"] == "nvrtc"
    assert normalized["tirx.disable_cse_tir"] is True


def test_raw_string_keys_still_work():
    legacy = normalize_pass_configs({PassConfigKey.TL_CUDA_COMPILER: "nvrtc"})
    raw = normalize_pass_configs({"tl.cuda_compiler": "nvrtc"})

    assert legacy == raw


def test_introduction_release_has_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        normalize_pass_configs({PassConfigKey.TL_CUDA_COMPILER: "nvrtc"})

    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_non_string_enum_values_are_rejected():
    class NonStringKey(Enum):
        VALUE = 1

    with pytest.raises(TypeError, match="string enum values"):
        normalize_pass_configs({NonStringKey.VALUE: True})


def test_cache_key_is_namespace_independent():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=1):
            A[0] = A[0]

    def cache_key(configs):
        # `_generate_key` only needs the instance for `_get_base_key`; skipping
        # `__init__` avoids a cache lookup or a real compilation.
        cache = object.__new__(KernelCache)
        return cache._generate_key(
            func=main,
            out_idx=[],
            args=[],
            pass_configs=normalize_pass_configs(configs),
        )

    legacy = cache_key({PassConfigKey.TL_CUDA_COMPILER: "nvrtc"})
    cuda = cache_key({CudaPassConfigKey.TL_CUDA_COMPILER: "nvrtc"})

    assert legacy == cuda


def test_cuda_key_is_accepted_by_pass_context():
    from tilelang import tvm

    with tvm.transform.PassContext(config={CudaPassConfigKey.TL_CUDA_COMPILER: "nvrtc"}):
        current = tvm.transform.PassContext.current().config

    assert str(current["tl.cuda_compiler"]) == "nvrtc"


def test_normalized_config_is_accepted_by_pass_context():
    from tilelang import tvm

    configs = normalize_pass_configs(
        {
            PassConfigKey.TL_ENABLE_FAST_MATH: True,
            CudaPassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 4,
        }
    )

    with tvm.transform.PassContext(config=configs):
        current = tvm.transform.PassContext.current().config

    assert bool(current["tl.enable_fast_math"])
    assert int(current["tl.ptxas_register_usage_level"]) == 4


if __name__ == "__main__":
    tilelang.testing.main()
