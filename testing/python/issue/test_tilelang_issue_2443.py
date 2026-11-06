import torch
import tilelang
import tilelang.testing
import tilelang.language as T

M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32


def make_kernel(disable_ws: bool):
    pass_configs = {}
    if disable_ws:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def gemm_bias_relu(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            bias: T.Tensor((N,), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                bias_shared = T.alloc_shared((block_N,), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                # PROLOGUE 1-D copy (correct slice index, correct extent).
                T.copy(bias[bx * block_N : (bx + 1) * block_N], bias_shared)

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.max(C_local[i, j] + bias_shared[j], 0)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return gemm_bias_relu(M, N, K, block_M, block_N, block_K)


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_gemm_bias_relu_prologue_shared_copy_with_warp_specialization():
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda")
    ref = torch.relu(a.float() @ b.float() + bias.float()).half()

    for disable_ws in (False, True):
        c = make_kernel(disable_ws)(a, b, bias)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    torch.cuda.synchronize()


if __name__ == "__main__":
    tilelang.testing.main()
