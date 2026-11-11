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


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 0)
@pytest.mark.parametrize("launcher", ["cpp", "cutlass"])
def test_cutedsl_grid_sync_concurrent_launches(launcher, monkeypatch):
    """Each launch synchronizes only its own grid, also when another launch of
    the same kernel runs concurrently on another stream (#3386)."""
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", launcher)
    if launcher == "cutlass":
        monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")

    @T.prim_func
    def kernel(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32"), delay: T.int64):
        with T.Kernel(2, threads=32) as bx:
            tx = T.get_thread_binding()
            if bx == 1:
                start = T.call_extern("int64", "cute::arch::clock64")
                while T.call_extern("int64", "cute::arch::clock64") - start < delay:
                    T.evaluate(T.call_extern("int64", "cute::arch::clock64"))
            if tx == 0:
                A[bx] = bx + 1
            T.sync_grid()
            if tx == 0:
                B[bx] = A[0] + A[1]

    compiled = tilelang.compile(kernel, target="cutedsl")
    slow, fast = torch.cuda.Stream(), torch.cuda.Stream()
    bufs = [torch.zeros(2, dtype=torch.int32, device="cuda") for _ in range(4)]
    for _ in range(4):
        for buf in bufs:
            buf.zero_()
        torch.cuda.synchronize()
        # While the slow launch's block 0 waits at the barrier for its block 1,
        # the fast launch's blocks arrive at the same barrier.
        with torch.cuda.stream(slow):
            compiled(bufs[0], bufs[1], 20_000_000)
        with torch.cuda.stream(fast):
            compiled(bufs[2], bufs[3], 0)
        torch.cuda.synchronize()
        torch.testing.assert_close(bufs[1], torch.full_like(bufs[1], 3))
        torch.testing.assert_close(bufs[3], torch.full_like(bufs[3], 3))


if __name__ == "__main__":
    tilelang.testing.main()
