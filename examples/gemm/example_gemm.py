import tilelang
import tilelang.language as T


# @tilelang.jit(out_idx=[-1], debug_root_path="./debug_output", pass_configs={
#     tilelang.PassConfigKey.TL_AST_PRINT_ENABLE: True,
#     tilelang.PassConfigKey.TL_PRINT_IR_WHEN_CHANGE: True
# })
# def matmul(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
#     @T.prim_func
#     def gemm(
#         A: T.Tensor((M, K), dtype),
#         B: T.Tensor((K, N), dtype),
#         C: T.Tensor((M, N), dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
#             A_shared = T.alloc_shared((block_M, block_K), dtype)
#             B_shared = T.alloc_shared((block_K, block_N), dtype)
#             C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

#             T.clear(C_local)
#             for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
#                 T.copy(A[by * block_M, k * block_K], A_shared)
#                 T.copy(B[k * block_K, bx * block_N], B_shared)
#                 T.gemm(A_shared, B_shared, C_local)

#             T.copy(C_local, C[by * block_M, bx * block_N])

#     return gemm



@tilelang.jit(out_idx=[-1])
def matmul_nt(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype), 
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        #### this bx, by is the usual cuda block idx. that the continues will be the bx;
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable rasterization for better L2 cache locality
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm_v2(A_shared, B_shared, C_local, transpose_B=True, k_pack=2)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return gemm
    

### TODO(zty): 1. add a new kernel for NT layout? or just transpose B and continues can do?

def main(transpose_b=False):
    M, N, K = 8192, 8192, 8192
    if not transpose_b:
        assert False, "not implemented"
        kernel = matmul(M, N, K, 128, 128, 32)
    else:
        kernel = matmul_nt(M, N, K, 256, 256, 64)

    import torch

    a = torch.randn(M, K).cuda().bfloat16()
    
    if not transpose_b:
        b = torch.randn(K, N).cuda().bfloat16()
        c = kernel(a, b)
        ref_c = a @ b
    else:
        # NT layout: B is (N, K) contiguous
        b_nt = torch.randn(N, K).cuda().bfloat16()
        c = kernel(a, b_nt)
        ref_c = a @ b_nt.T

    # print("c:")
    # print(c)
    # print("ref_c:")
    # print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    # print("CUDA Source:")
    # print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti", rep=500)
    # latency = profiler.do_bench()

    print(f"tilelang Latency: {latency}ms")
    print(f"tilelang flops: {2 * M * N * K / latency / 1e9} TFLOPS")
    print(f"shape: {M}x{N}x{K}")


def run_regression_perf():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    # main()
    main(transpose_b=True)
