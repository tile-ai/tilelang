"""CPU regression test for SM arch string parsing in the carver.

``check_sm_version`` gates every compute-capability decision the carver makes:
``arch.sm_version`` feeds ``is_volta_arch`` / ``is_ampere_arch`` /
``is_ada_arch`` / ``is_hopper_arch`` / ``has_mma_support``, and
``matmul_analysis`` compares the same value against 70, 80 and 90 to pick the
tensorcore path.

nvcc and CUTLASS spell the arch-specific feature set of Hopper and newer with a
trailing letter (``sm_90a``, ``sm_100a``, ``sm_103a``), and that is the value
that reaches ``target.attrs["arch"]``. The old ``str.isdigit()`` guard rejected
those strings and collapsed them to the -1 sentinel, so a real ``sm_90a``
target read as older than sm_70 and silently lost its Hopper dispatch.

No GPU is needed: the defect and the fix are pure string parsing.
"""

import tilelang.testing
from tilelang.carver.arch import cuda as carver_cuda
from tilelang.carver.arch.cuda import CUDA, check_sm_version, has_mma_support, is_ada_arch, is_ampere_arch, is_hopper_arch, is_volta_arch
from tilelang.carver import matmul_analysis


def test_check_sm_version_numeric_arch():
    assert check_sm_version("sm_70") == 70
    assert check_sm_version("sm_80") == 80
    assert check_sm_version("sm_89") == 89
    assert check_sm_version("sm_90") == 90
    # The bare numeric form was already accepted and must stay accepted.
    assert check_sm_version("90") == 90


def test_check_sm_version_lettered_arch():
    assert check_sm_version("sm_90a") == 90
    assert check_sm_version("sm_100a") == 100
    assert check_sm_version("sm_103a") == 103
    assert check_sm_version("sm_120f") == 120


def test_check_sm_version_rejects_non_cuda_arch():
    # Not CUDA arches, so the -1 sentinel is the correct answer here.
    assert check_sm_version("gfx942") == -1
    assert check_sm_version("gfx90a") == -1
    assert check_sm_version("sm_") == -1
    assert check_sm_version("") == -1


def _cuda_arch_with_sm_version(sm_version: int) -> CUDA:
    """Build a CUDA arch carrying ``sm_version`` without touching a device.

    ``CUDA.__init__`` queries cuda device 0, which no CPU runner has, but the
    predicates below only read ``sm_version`` plus ``isinstance(arch, CUDA)``.
    """
    arch = CUDA.__new__(CUDA)
    arch.sm_version = sm_version
    return arch


def test_lettered_arch_keeps_capability_dispatch():
    hopper = _cuda_arch_with_sm_version(check_sm_version("sm_90a"))
    assert is_hopper_arch(hopper)
    assert has_mma_support(hopper)
    assert not is_volta_arch(hopper)
    assert not is_ampere_arch(hopper)
    assert not is_ada_arch(hopper)

    blackwell = _cuda_arch_with_sm_version(check_sm_version("sm_100a"))
    assert has_mma_support(blackwell)


def test_matmul_analysis_shares_one_parser():
    # matmul_analysis used to carry its own copy of the same buggy body, so a
    # fix in one place left the other wrong. Pin the single definition.
    assert matmul_analysis.check_sm_version is carver_cuda.check_sm_version


if __name__ == "__main__":
    tilelang.testing.main()
