"""Tests for backend-owned pass-config namespaces.

The CUDA-owned members are exposed from ``tilelang.cuda`` while the historical
``tilelang.transform.PassConfigKey`` spellings remain available during the
introduction release.  Both spellings must normalize to the same string key so
that pass contexts, JIT kernels, and cache keys are unaffected.
"""

import warnings
from enum import Enum

import pytest

import tilelang.language as T
import tilelang.testing
from tilelang.cache.kernel_cache import KernelCache
from tilelang.cuda import PassConfigKey as CudaPassConfigKey
from tilelang.transform.pass_config import PassConfigKey, normalize_pass_configs


def test_cuda_namespace_exports_a_static_enum():
    import tilelang.cuda.pass_config as cuda_pass_config

    assert CudaPassConfigKey is cuda_pass_config.PassConfigKey
    assert issubclass(CudaPassConfigKey, str)
    assert CudaPassConfigKey.__members__


def test_every_cuda_member_has_a_common_compatibility_alias():
    for name, member in CudaPassConfigKey.__members__.items():
        assert name in PassConfigKey.__members__
        assert member.value == PassConfigKey[name].value


def test_normalization_is_namespace_independent():
    for name, member in CudaPassConfigKey.__members__.items():
        with warnings.catch_warnings():
            # TL_DISABLE_TMA_LOWER is deprecated; its warning is exercised by
            # the existing TMA-lowering tests.
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = normalize_pass_configs({PassConfigKey[name]: 1})
            cuda = normalize_pass_configs({member: 1})
            raw = normalize_pass_configs({member.value: 1})

        assert legacy == cuda == raw
        assert legacy[member.value] == 1


def test_introduction_release_has_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        normalize_pass_configs(
            {
                PassConfigKey.TL_CUDA_COMPILER: "nvrtc",
                PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
            }
        )

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


if __name__ == "__main__":
    tilelang.testing.main()
