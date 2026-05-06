import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner import set_autotune_inputs


@tilelang.autotune(configs=[{"threads": 128}, {"threads": 256}], warmup=1, rep=1, timeout=60)
@tilelang.jit
def add_scalar(N: int = 4096, BLOCK_N: int = 512, threads: int = 128):
    @T.prim_func
    def kernel(A: T.Tensor((N,), T.float32), s: T.float32):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=threads) as pid_n:
            A_local = T.alloc_fragment((BLOCK_N,), T.float32)
            T.copy(A[pid_n * BLOCK_N], A_local)
            for i in T.Parallel(BLOCK_N):
                A_local[i] += s
            T.copy(A_local, A[pid_n * BLOCK_N])

    return kernel


def test_autotune_scalar_inputs_require_explicit_supply():
    with pytest.raises(ValueError, match=r"set_autotune_inputs"):
        add_scalar()


def test_autotune_scalar_inputs_with_set_autotune_inputs():
    tune_a = torch.randn((4096,), device="cuda", dtype=torch.float32)
    tune_s = 0.1
    with set_autotune_inputs(tune_a, tune_s):
        kernel = add_scalar()

    a = torch.randn((4096,), device="cuda", dtype=torch.float32)
    before = a.clone()
    kernel(a, tune_s)

    torch.testing.assert_close(a, before + tune_s, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
