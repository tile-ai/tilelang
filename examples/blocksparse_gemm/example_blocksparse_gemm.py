# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch

torch.random.manual_seed(0)


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    block_mask_shape = (M // block_M, N // block_N, K // block_K)

    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            BlockMask: T.Buffer(block_mask_shape, "bool"),
            C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                if BlockMask[by, bx, k]:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


func = matmul(1024, 1024, 1024, 128, 128, 32)

print(func)
kernel = tilelang.compile(func, out_idx=-1)

a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()
# block_mask = torch.zeros(1024 // 128, 1024 // 128, 1024 // 32).cuda().bool()
# block_mask = torch.ones(1024 // 128, 1024 // 128, 1024 // 32).cuda().bool()
# random mask
block_mask = torch.randint(0, 2, (1024 // 128, 1024 // 128, 1024 // 32)).cuda().bool()

c = kernel(a, b, block_mask)

ref_c = torch.zeros_like(c)
for i in range(1024 // 128):
    for j in range(1024 // 128):
        accu = torch.zeros((128, 128), dtype=torch.float32, device=a.device)
        for k in range(1024 // 32):
            if block_mask[i, j, k]:
                accu += (
                    a[i * 128:(i + 1) * 128, k * 32:(k + 1) * 32].to(torch.float32)
                    @ b[k * 32:(k + 1) * 32, j * 128:(j + 1) * 128].to(torch.float32))
        ref_c[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = accu.to(torch.float16)

# ref_c = a @ b
print(c)
print(ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print(kernel.get_kernel_source())
