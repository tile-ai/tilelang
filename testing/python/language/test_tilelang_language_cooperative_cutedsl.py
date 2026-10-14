import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

try:
    from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

    check_cutedsl_available()
except (ImportError, AssertionError):
    pytest.skip("CuTeDSL not installed", allow_module_level=True)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 0)
@pytest.mark.parametrize("threads", [32, (32, 2), (32, 1, 2), (16, 2, 2)])
def test_cutedsl_grid_sync_multidimensional_threads(threads):
    @T.prim_func
    def kernel(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32")):
        with T.Kernel(2, threads=threads) as bx:
            tx = T.get_thread_binding()
            ty = T.get_thread_binding(1)
            tz = T.get_thread_binding(2)
            # Give block 0 time to reach the barrier before block 1 writes.
            # Checking only tid.x elects multiple leaders in a 2D/3D block,
            # so block 0 can release itself and read the initial A[1].
            if bx == 1:
                start = T.call_extern("int64", "cute::arch::clock64")
                while T.call_extern("int64", "cute::arch::clock64") - start < 1000000:
                    T.evaluate(T.call_extern("int64", "cute::arch::clock64"))
            if tx == 0 and ty == 0 and tz == 0:
                A[bx] = bx + 1
            T.sync_grid()
            if tx == 0 and ty == 0 and tz == 0:
                B[bx] = A[0] + A[1]

    compiled = tilelang.compile(kernel, target="cutedsl")
    A = torch.zeros(2, dtype=torch.int32, device="cuda")
    B = torch.zeros_like(A)
    for _ in range(8):
        A.zero_()
        B.zero_()
        compiled(A, B)
        torch.testing.assert_close(B, torch.full_like(B, 3))


if __name__ == "__main__":
    tilelang.testing.main()
