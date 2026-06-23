import tilelang
import tilelang.language as T
import torch

M, N, K = 128, 128, 33  # K = BLOCK_K + 1 -> 1-element residual K-tile
BM, BN, BK = 64, 64, 32


@T.prim_func
def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        T.clear(Cl)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], As)
            T.copy(B[k * BK, bx * BN], Bs)
            T.gemm(As, Bs, Cl)
        T.copy(Cl, C[by * BM, bx * BN])


tilelang.disable_cache()
kernel = tilelang.compile(
    main,
    out_idx=[2],
    execution_backend="cython",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)  # crashes here
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = kernel(a, b)
print("ok:", torch.allclose(c, a.float() @ b.float(), rtol=1e-2, atol=1e-1))
