import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def polynomial_kernel(
    B: int = 1,
    M: int = 512,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
    threads: int = 256,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)

    @T.prim_func
    def kernel(
        A: T.Tensor((B, M, M), dtype),
        C: T.Tensor((B, M, M), dtype),
        C0: T.float32,
        C1: T.float32,
        C2: T.float32,
    ):
        with T.Kernel(B, total_blocks, threads=threads) as (pid_b, pid):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R

            pid_m = T.alloc_var(T.int32)
            pid_m = T.cast(
                T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0) * 0.5 - 0.5,
                T.int32,
            )
            pid_m_col = base_pid - pid_m * (pid_m + 1) // 2
            pid_n = pid_m_col * R + pid % R

            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))

            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_b, pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(A[pid_b, pid_n * BLOCK_N, k * BLOCK_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            s_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            t_shared = T.alloc_shared((BLOCK_N, BLOCK_M), dtype)
            T.copy(A[pid_b, pid_m * BLOCK_M, pid_n * BLOCK_N], s_shared)

            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    s_shared[i, j] = T.cast(
                        C1 * T.cast(s_shared[i, j], T.float32) + C2 * C_local[i, j],
                        dtype,
                    )
                T.copy(s_shared, C[pid_b, pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(s_shared, t_shared)
                T.copy(t_shared, C[pid_b, pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    s_shared[i, j] = T.cast(
                        C1 * T.cast(s_shared[i, j], T.float32)
                        + C2 * C_local[i, j]
                        + T.if_then_else(
                            pid_m * BLOCK_M + i == pid_n * BLOCK_N + j,
                            C0,
                            0,
                        ),
                        dtype,
                    )
                T.copy(s_shared, C[pid_b, pid_m * BLOCK_M, pid_n * BLOCK_N])

    return kernel


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_scalarized_epilogue_keeps_shared_memory_barrier():
    shape = 512
    torch.manual_seed(0)
    x = torch.randn(shape, shape, device="cuda", dtype=torch.float32)
    x /= x.norm()
    gram = (x @ x.T).half()

    a = 5.8918421975177555
    b = -16.720493089530553
    c = 11.910583457755012
    reference = a * torch.eye(shape, device="cuda") + b * gram.float() + c * (gram.float() @ gram.float().T)

    compiled = tilelang.jit(
        polynomial_kernel,
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )(B=1, M=shape, dtype="float16")

    output = torch.empty_like(gram)
    for _ in range(3):
        output.fill_(float("nan"))
        compiled(gram, output, a, b, c)
        torch.cuda.synchronize()
        assert output.isfinite().all()
        relative_error = (output.float() - reference).norm() / reference.norm()
        assert relative_error < 0.002
