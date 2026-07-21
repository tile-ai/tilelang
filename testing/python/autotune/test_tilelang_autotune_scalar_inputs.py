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


# Reproduces the segfault from calling an autotuned kernel whose tunable
# parameters default to ``None`` (the common ``block_M=None`` style) before any
# config is applied. The old validation path built a prim_func with those
# ``None`` values and crashed inside TVM (e.g. ``ceildiv`` over a None extent).
# Tuning must elaborate a concrete config first and only then validate.
@tilelang.autotune(
    configs=[{"BLOCK_N": 128, "threads": 128}, {"BLOCK_N": 256, "threads": 256}],
    warmup=1,
    rep=1,
    timeout=60,
)
@tilelang.jit(out_idx=[2])
def vector_add_none_tunable(N: int = 4096, BLOCK_N: int = None, threads: int = None):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        C: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=threads) as pid_n:
            A_local = T.alloc_fragment((BLOCK_N,), T.float32)
            B_local = T.alloc_fragment((BLOCK_N,), T.float32)
            T.copy(A[pid_n * BLOCK_N], A_local)
            T.copy(B[pid_n * BLOCK_N], B_local)
            for i in T.Parallel(BLOCK_N):
                A_local[i] = A_local[i] + B_local[i]
            T.copy(A_local, C[pid_n * BLOCK_N])

    return kernel


@tilelang.testing.requires_cuda
def test_autotune_none_tunable_does_not_segfault():
    # Calling with only the positional shape arg leaves BLOCK_N/threads at
    # their None defaults; tuning must bind a concrete config before any TIR is
    # constructed. A segfault here means the fix regressed.
    kernel = vector_add_none_tunable()
    assert kernel is not None


if __name__ == "__main__":
    tilelang.testing.main()
