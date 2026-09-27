"""Runtime tests for multiple VF-bearing kernels and kernel frames."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

from example_simtvf_multi_kernel import (
    ref_add,
    ref_scale,
    vector_add,
    vector_scale,
)


def make_multi_kernel_program(N):
    @T.prim_func
    def main(
        a: T.Buffer((N,), "float32"),
        b: T.Buffer((N,), "float32"),
        c: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=128):
            for i in T.Parallel(N):
                c[i] = a[i] + b[i]

        with T.Kernel(1), T.SimtVF(threads=128):
            for i in T.Parallel(N):
                c[i] = c[i] * T.float32(2.0)

    return main


@pytest.mark.parametrize("target", ["ascend"])  # PTO backend temporarily disabled for this multi-kernel test.
def test_simtvf_multi_kernel(target):
    N = 1024
    SCALE = 3.0
    device = torch.device("npu")

    add_kernel = tilelang.compile(vector_add(N), target=target, out_idx=-1)
    scale_kernel = tilelang.compile(vector_scale(N, SCALE), target=target, out_idx=-1)
    multi_kernel = tilelang.compile(make_multi_kernel_program(N), target=target, out_idx=-1)

    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)

    c_add = add_kernel(a, b)
    c_scale = scale_kernel(a)
    c_multi = multi_kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(c_add, ref_add(a, b))
    assert torch.equal(c_scale, ref_scale(a, SCALE))
    assert torch.equal(c_multi, (a + b) * 2.0)


if __name__ == "__main__":
    tilelang.testing.main()
