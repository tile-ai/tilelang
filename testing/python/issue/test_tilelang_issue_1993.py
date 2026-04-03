import torch
import pytest

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def _issue1993_dynamic_grid():
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def kernel(out: T.Tensor[(num_tokens,), T.float32]):
        with T.Kernel(num_tokens, threads=1) as pid:
            out[pid] = T.float32(1.0)

    return kernel


@tilelang.jit
def _issue1993_static_grid():

    @T.prim_func
    def kernel(out: T.Tensor[(4,), T.float32]):
        with T.Kernel(0, threads=1) as pid:
            out[pid] = T.float32(1.0)

    return kernel


@tilelang.testing.requires_cuda
def test_issue_1993_dynamic_zero_grid_dim():
    """Regression test for issue #1993.

    When a dynamic grid dimension resolves to 0 at runtime, the runtime
    should raise an error instead of silently clamping to 1 and launching
    the kernel (which would write through a NULL pointer and crash with
    CUDA_ERROR_ILLEGAL_ADDRESS).
    """
    kernel = _issue1993_dynamic_grid()

    # Positive case: should work correctly
    out = torch.zeros(4, dtype=torch.float32, device="cuda")
    kernel(out)
    torch.cuda.synchronize()
    assert out.eq(1.0).all()

    # Zero case: should raise an error, not crash with illegal memory access
    out_empty = torch.zeros(0, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):  # noqa: B017
        kernel(out_empty)
        torch.cuda.synchronize()


@tilelang.testing.requires_cuda
def test_issue_1993_static_zero_grid_dim():
    """Regression test for issue #1993.

    When T.Kernel(0) is used with a static constant, the runtime should
    raise an error instead of silently clamping to 1 and executing a
    spurious CTA.
    """
    kernel = _issue1993_static_grid()

    out = torch.zeros(4, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):  # noqa: B017
        kernel(out)
        torch.cuda.synchronize()

    # Buffer should remain untouched
    assert out.eq(0.0).all()


if __name__ == "__main__":
    tilelang.testing.main()
