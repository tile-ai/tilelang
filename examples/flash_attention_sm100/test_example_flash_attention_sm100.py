import pytest

import tilelang.testing

import gqa_bwd_bshd
import gqa_fwd_bshd
import mha_bwd_bshd
import mha_fwd_bshd


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("variant", ["ss", "ts", "wasp"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_example_mha_fwd_bshd(variant, is_causal):
    mha_fwd_bshd.main(is_causal=is_causal, variant=variant)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("variant", ["ss", "ts", "wasp"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_example_gqa_fwd_bshd(variant, is_causal):
    gqa_fwd_bshd.main(is_causal=is_causal, groups=2, variant=variant)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("variant", ["ss", "ts"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_example_mha_bwd_bshd(variant, is_causal):
    mha_bwd_bshd.main(is_causal=is_causal, variant=variant)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("variant", ["ss", "ts"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_example_gqa_bwd_bshd(variant, is_causal):
    gqa_bwd_bshd.main(is_causal=is_causal, groups=2, variant=variant)


if __name__ == "__main__":
    tilelang.testing.main()
