# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

from tilelang.layout import make_metadata_layout
from tilelang.utils.sparse import compress
from tilelang.contrib import nvcc

import torch

# arch = nvcc.get_target_compute_version()
arch = "8.0"


SPARSITY_MAP = {
    torch.float16: (2, 4),
}

ARCH_INFO = {
    "8.0": (16, "int16"),
    "9.0": (4, "uint8")
}

def generate_sparse_tensor_float32(M: int, K: int, dtype: torch.dtype, device='cpu', trans_A=False):
    elem, group = SPARSITY_MAP[dtype]
    if K % group != 0:
        raise ValueError(
            f"Last dimension must be divisible by {group} for {elem}:{group} sparsity.")

    if trans_A:
        full_tensor = torch.randn(K * M, dtype=torch.float32, device=device).view(K, M)
        mask = torch.zeros_like(full_tensor, dtype=torch.bool)
        for j in range(M):
            for i in range(0, K, group):
                flat_idx = torch.randint(0, group, (elem,), dtype=torch.int64)
                for k in range(1, len(flat_idx)):
                    while flat_idx[k] in flat_idx[:k]:
                        flat_idx[k] = torch.randint(0, group, (1,), dtype=torch.int64)
                for idx in flat_idx:
                    mask[i + idx, j] = True
    else:
        full_tensor = torch.randn((M, K), dtype=torch.float32, device=device).view(M, K)
        mask = torch.zeros_like(full_tensor, dtype=torch.bool)
        for i in range(M):
            for j in range(0, K, group):
                flat_idx = torch.randint(0, group, (elem,), dtype=torch.int64)
                for k in range(1, len(flat_idx)):
                    while flat_idx[k] in flat_idx[:k]:
                        flat_idx[k] = torch.randint(0, group, (1,), dtype=torch.int64)
                for idx in flat_idx:
                    mask[i, j + idx] = True

    return full_tensor * mask



@tilelang.jit(out_idx=[-1])
def matmul_sp_fp16(M, N, K, block_M, block_N, block_K, accum_dtype="float"):
    e_factor, e_dtype = ARCH_INFO[arch]
    @T.prim_func
    def gemm_sp_fp16(
            A_sparse: T.Tensor((M, K // 2), 'float16'),
            E: T.Tensor((M, K // e_factor), e_dtype),
            B: T.Tensor((K, N), 'float16'),
            C: T.Tensor((M, N), 'float16'),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), 'float16')
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            B_shared = T.alloc_shared((block_K, block_N), 'float16')
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout({
                E:
                    make_metadata_layout(
                        E, mma_dtype="float16", backend="cutlass", block_k=block_K, arch=arch),
                E_shared:
                    make_metadata_layout(
                        E_shared,
                        mma_dtype="float16",
                        backend="cutlass",
                        block_k=block_K,
                        arch=arch),
            })
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, False, False)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_sp_fp16


def main():
    block_k = 32
    kernel = matmul_sp_fp16(1024, 1024, 1024, 128, 128, block_k)

    a = generate_sparse_tensor_float32(1024, 1024, torch.float16, device='cuda').half()
    b = torch.randn(1024, 1024).cuda().half()

    a_sparse, e = compress(a, transposed=False, block_k=block_k, arch=arch)
    c = kernel(a_sparse, e, b)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Precision check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
