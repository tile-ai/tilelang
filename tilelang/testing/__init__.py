import sys
import inspect
import pytest
import random
import torch
import numpy as np
from tilelang.contrib import nvcc
from tilelang.backend.target import determine_target
from tilelang.cuda.target import target_is_cuda
from tilelang.rocm.target import target_is_cdna, target_is_gfx950
from tvm.testing.utils import Feature, requires_cuda, requires_package, requires_llvm, requires_metal, requires_rocm, _compose

from tilelang.utils.tensor import torch_assert_close as torch_assert_close
from .perf_regression import process_func, regression

__all__ = [
    "requires_package",
    "requires_cuda",
    "requires_metal",
    "requires_rocm",
    "requires_llvm",
    "requires_cdna",
    "requires_cuda_or_cdna",
    "requires_gfx950",
    "main",
    "ascend_backend_compiled",
    "requires_ascend",
    "requires_cuda_compute_version",
    "process_func",
    "regression",
] + [f"requires_cuda_compute_version_{op}" for op in ("ge", "gt", "le", "lt", "eq")]


def _check_is_gfx950() -> bool:
    try:
        target = determine_target("auto", return_object=True)
        return target_is_gfx950(target)
    except (ValueError, RuntimeError):
        return False


def _check_is_cdna() -> bool:
    try:
        target = determine_target("auto", return_object=True)
        return target_is_cdna(target)
    except (ValueError, RuntimeError):
        return False


def _check_is_cuda_or_cdna() -> bool:
    try:
        target = determine_target("auto", return_object=True)
        return target_is_cuda(target) or target_is_cdna(target)
    except (ValueError, RuntimeError):
        return False


def ascend_backend_compiled() -> bool:
    """Whether this build includes the Ascend backend (USE_ASCEND).

    The Ascend sources are gated behind USE_ASCEND, so a CUDA-only build has no
    Ascend tile ops and no Ascend pass-config options. The pass-config key is the
    cheapest reliable probe: it is registered from src/ascend.
    """
    try:
        import tvm

        from tilelang.transform import PassConfigKey

        with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: True}):
            pass
        return True
    except Exception:
        return False


def _ascend_device_available() -> bool:
    """Whether an Ascend NPU is usable on this machine."""
    try:
        from tilelang.ascend.target import check_ascend_availability

        return check_ascend_availability()
    except Exception:
        return False


# Registered on the same open registry as requires_cuda/requires_rocm: the
# Feature constructor publishes itself into Feature._all_features, which is what
# gives the `ascend` pytest marker, the compile/run split, and `-m ascend`.
#
# target_kind_enabled is deliberately unset. For CUDA/Metal/ROCm it gates on
# TVM_TEST_TARGETS, whose default list has no ascend entry, so setting it would
# skip every Ascend test unless that variable named ascend. target_kind_hardware
# is unset too: it would call tvm.device("ascend").exist, and ascend is a
# TileLang target kind, not a registered TVM runtime ("Unknown optional runtime
# ascend"). Device availability is checked through torch_npu instead.
requires_ascend = Feature(
    "ascend",
    "Ascend",
    compile_time_check=ascend_backend_compiled,
    run_time_check=_ascend_device_available,
)

def requires_cdna(func):
    """Skip the test unless the ROCm device is a CDNA GPU."""
    is_cdna = _check_is_cdna()
    marks = [
        pytest.mark.skipif(
            not is_cdna,
            reason="Requires CDNA ROCm target",
        ),
        *requires_rocm.marks(),
    ]
    return _compose([func], marks)


def requires_cuda_or_cdna(func):
    """Skip the test unless the device is CUDA or CDNA ROCm."""
    is_cuda_or_cdna = _check_is_cuda_or_cdna()
    marks = [
        pytest.mark.skipif(
            not is_cuda_or_cdna,
            reason="Requires CUDA or CDNA ROCm target",
        ),
    ]
    return _compose([func], marks)


def requires_gfx950(func):
    """Skip the test unless the ROCm device is gfx950 (CDNA4 / MI350)."""
    is_gfx950 = _check_is_gfx950()
    marks = [
        pytest.mark.skipif(
            not is_gfx950,
            reason="Requires gfx950 (CDNA4/MI350)",
        ),
        *requires_rocm.marks(),
    ]
    return _compose([func], marks)


# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def requires_cuda_compute_version(major_version, minor_version=0, mode="ge"):
    """Mark a test as requiring at least a compute architecture

    Unit test marked with this decorator will run only if the CUDA
    compute architecture of the GPU is at least `(major_version,
    minor_version)`.

    This also marks the test as requiring a cuda support.

    Parameters
    ----------
    major_version: int

        The major version of the (major,minor) version tuple.

    minor_version: int

        The minor version of the (major,minor) version tuple.

    mode: str

        The mode of the comparison.
        - "ge": greater than or equal to
        - "gt": greater than
        - "le": less than or equal to
        - "lt": less than
    """
    min_version = (major_version, minor_version)
    try:
        arch = nvcc.get_target_compute_version()
        compute_version = nvcc.parse_compute_version(arch)
    except ValueError:
        # No GPU present.  This test will be skipped from the
        # requires_cuda() marks as well.
        compute_version = (0, 0)

    min_version_str = ".".join(str(v) for v in min_version)
    compute_version_str = ".".join(str(v) for v in compute_version)

    def compare(compute_version, min_version, mode) -> bool:
        if mode == "ge":
            return compute_version >= min_version
        elif mode == "gt":
            return compute_version > min_version
        elif mode == "le":
            return compute_version <= min_version
        elif mode == "lt":
            return compute_version < min_version
        elif mode == "eq":
            return compute_version == min_version
        else:
            raise ValueError(f"Invalid mode: {mode}")

    requires = [
        pytest.mark.skipif(
            not compare(compute_version, min_version, mode),
            reason=f"Requires CUDA compute {mode} {min_version_str}, but have {compute_version_str}",
        ),
        *requires_cuda.marks(),
    ]

    def inner(func):
        return _compose([func], requires)

    return inner


def requires_cuda_compute_version_ge(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="ge")


def requires_cuda_compute_version_gt(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="gt")


def requires_cuda_compute_version_eq(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="eq")


def requires_cuda_compute_version_lt(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="lt")


def requires_cuda_compute_version_le(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="le")
