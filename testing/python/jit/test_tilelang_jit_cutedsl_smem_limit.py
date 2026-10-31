"""A CuTeDSL kernel that needs more dynamic shared memory than the device allows
per block must fail with the same message as the CUDA backend, not with an
opaque launch error code."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


@tilelang.jit(target="cutedsl")
def staged_copy_kernel(n):
    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((n,), "float32")
            T.copy(A, S)
            T.copy(S, B)

    return kernel


@tilelang.testing.requires_cuda
def test_cutedsl_rejects_dynamic_smem_above_device_limit():
    _require_cutedsl()
    # 256 KiB is above the opt-in cap of every architecture (at most 227 KiB),
    # so the launch must be rejected without querying PyTorch device properties.
    n = 256 * 1024 // 4
    a = torch.randn(n, device="cuda")
    b = torch.empty_like(a)
    with pytest.raises(RuntimeError, match=r"Failed to set the allowed dynamic shared memory size to \d+ .*at most"):
        staged_copy_kernel(n)(a, b)


@tilelang.testing.requires_cuda
def test_cutedsl_dynamic_smem_within_device_limit_runs():
    _require_cutedsl()
    n = 16 * 1024
    a = torch.randn(n, device="cuda")
    b = torch.empty_like(a)
    staged_copy_kernel(n)(a, b)
    torch.testing.assert_close(b, a)


if __name__ == "__main__":
    tilelang.testing.main()
