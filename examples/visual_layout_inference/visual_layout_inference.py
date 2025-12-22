import tilelang
import tilelang.language as T


# use pass_configs to enable layout visualization
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "svg",
    },
)
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(128, 128, 128, 32, 32, 32)

    import torch

    a = torch.randn(128, 128).cuda().half()
    b = torch.randn(128, 128).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # print the layout visualization result and save figures to ./tmp.
    """
    C_local inferenced layout:
    Shape: [32, 32] -> [8]
    Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
    Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
    """


if __name__ == "__main__":
    main()
