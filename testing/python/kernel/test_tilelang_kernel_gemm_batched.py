from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul_batched(
    batch,
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (batch, K, M) if trans_A else (batch, M, K)
    B_shape = (batch, N, K) if trans_B else (batch, K, N)
    A_shared_shape = (batch, block_K, block_M) if trans_A else (batch, block_M, block_K)
    B_shared_shape = (batch, block_N, block_K) if trans_B else (batch, block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((batch, M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.disable_warp_group_reg_alloc()
            for b in T.serial(batch):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    if trans_A:
                        T.copy(A[b, k * block_K, by * block_M], A_shared[b, :, :])
                    else:
                        T.copy(A[b, by * block_M, k * block_K], A_shared[b, :, :])
                    if trans_B:
                        T.copy(B[b, bx * block_N, k * block_K], B_shared[b, :, :])
                    else:
                        T.copy(B[b, k * block_K, bx * block_N], B_shared[b, :, :])
                    T.gemm(A_shared[b, :, :], B_shared[b, :, :], C_local, trans_A, trans_B)
                T.copy(C_local, C[b, by * block_M, bx * block_N])

    return main


def run_gemm_batched(
    batch,
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
):
    program = matmul_batched(
        batch,
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.transpose(-1, -2)
        if trans_B:
            B = B.transpose(-1, -2)
        if in_dtype == T.float32:
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.contiguous().view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.contiguous().view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
def test_gemm_f16f16f16_nn():
    run_gemm_batched(
        2,
        64,
        64,
        32,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        64,
        64,
        32,
        0,
    )


if __name__ == "__main__":
    tilelang.testing.main()
