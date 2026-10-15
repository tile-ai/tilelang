# ruff: noqa
import pytest
import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.testing.requires_cuda
def test_issue_2393_serial_reduce_gemm_fragment_is_rejected():
    """#2393: a serial reduce of a GEMM-output (MMA-layout) fragment used to fail
    with an unhelpful internal error ("contains inner var j"). Now rejected with
    a clear diagnostic suggesting T.reduce_sum."""
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


if __name__ == "__main__":
    tilelang.testing.main()
