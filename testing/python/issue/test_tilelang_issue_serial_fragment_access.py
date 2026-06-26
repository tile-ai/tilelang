# ruff: noqa
import pytest
import tilelang
import tilelang.testing
import tilelang.language as T


# A T.serial loop runs sequentially inside each CUDA thread, but a fragment's
# elements are distributed across threads, so indexing a fragment by a serial
# loop variable has no valid thread-ownership mapping. The three cases below
# (issues #2393, #2395, #2396) share this root cause and must be rejected with a
# clear diagnostic instead of crashing nvcc / silently miscomputing. A scalar
# fragment[0] store, which the rejection must NOT break, is also covered.


@tilelang.testing.requires_cuda
def test_issue_2393_serial_reduce_gemm_fragment_is_rejected():
    """#2393: a serial reduce of a GEMM-output (MMA-layout) fragment used to fail
    with an unhelpful internal error ("contains inner var j")."""
    M, N, KD = 64, 64, 64

    @tilelang.jit
    def k(Q: T.Tensor((M, KD), T.float16), Kk: T.Tensor((N, KD), T.float16)):
        out = T.empty((M,), T.float32)
        with T.Kernel(1, threads=128) as bx:
            Qs = T.alloc_shared((M, KD), T.float16)
            Ks = T.alloc_shared((N, KD), T.float16)
            acc = T.alloc_fragment((M, N), T.float32)
            s = T.alloc_fragment((M,), T.float32)
            T.copy(Q, Qs)
            T.copy(Kk, Ks)
            T.clear(acc)
            T.gemm(Qs, Ks, acc, transpose_B=True)
            T.fill(s, 0.0)
            for i in T.Parallel(M):
                for j in T.serial(N):
                    s[i] += acc[i, j]
            T.copy(s, out)
        return out

    import torch

    Q = torch.randn(M, KD, dtype=torch.float16, device="cuda")
    Kk = torch.randn(N, KD, dtype=torch.float16, device="cuda")
    with pytest.raises(Exception) as excinfo:
        k(Q, Kk)
    msg = str(excinfo.value)
    assert "thread-distributed fragment" in msg and "T.reduce_sum" in msg


@tilelang.testing.requires_cuda
def test_issue_2395_serial_write_fragment_then_copy_is_rejected():
    """#2395: serial write into a fragment then T.copy out used to emit an
    owner-thread guard built from serial loop vars outside the loop that binds
    them, so nvcc failed with "identifier undefined"."""
    M = N = 16

    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            Af = T.alloc_fragment((M, N), "float32")
            Bf = T.alloc_fragment((M, N), "float32")
            T.copy(A[0, 0], Af)
            for i in T.serial(M):
                for j in T.serial(N):
                    Bf[i, j] = Af[i, j] * 2.0
            T.copy(Bf, C[0, 0])

    with pytest.raises(Exception) as excinfo:
        tilelang.compile(kernel, target="cuda")
    assert "serial" in str(excinfo.value) and "T.Parallel" in str(excinfo.value)


@tilelang.testing.requires_cuda
def test_issue_2396_serial_fragment_to_global_is_rejected():
    """#2396: serial write of a fragment to global used to compile and run with
    silently wrong results (concurrent threads race-write each cell)."""
    M = N = 16

    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            Af = T.alloc_fragment((M, N), "float32")
            T.copy(A[0, 0], Af)
            for i in T.serial(M):
                for j in T.serial(N):
                    C[i, j] = Af[i, j] * 2.0

    with pytest.raises(Exception) as excinfo:
        tilelang.compile(kernel, target="cuda")
    assert "serial" in str(excinfo.value) and "T.Parallel" in str(excinfo.value)


@tilelang.testing.requires_cuda
def test_scalar_fragment_store_is_accepted():
    """The rejection must not break the case this pass exists to handle: a bare
    scalar fragment[0] store (not inside a tile op, not thread-bound) is still
    wrapped and compiles to the correct result."""
    import torch

    @T.prim_func
    def kernel(C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((1,), "float32")
            frag[0] = 1.0
            C[0] = frag[0]

    kfn = tilelang.compile(kernel, target="cuda")
    c = torch.empty(1, dtype=torch.float32, device="cuda")
    kfn(c)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.ones(1, dtype=torch.float32, device="cuda"))


if __name__ == "__main__":
    tilelang.testing.main()
