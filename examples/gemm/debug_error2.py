import os
os.system('find ~/.tilelang/ -name "*.so" -print -delete')
os.environ["TL_DISABLE_G2S_HOIST"] = "1"
import tilelang
import tilelang.language as T
import torch

# Test 1: single k iteration (K=64, block_K=64 -> 1 iteration)
@tilelang.jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_SCATTERED_WARP_LAYOUT: True,
})
def matmul_nt_k1(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=4, num_xcds=8)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm_v2(A_shared, B_shared, C_local, transpose_B=True, k_pack=2)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm

print("=== Test 1: K=64 (1 iteration) ===")
M, N, K = 256, 256, 64
kernel1 = matmul_nt_k1(M, N, K, 256, 256, 64)
a = torch.randn(M, K).cuda().bfloat16()
b = torch.randn(N, K).cuda().bfloat16()
c = kernel1(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()
abs_diff = (c - ref_c).abs()
mismatched = (abs_diff > 0.01).sum().item()
print(f"Mismatched: {mismatched}/{M*N} ({100*mismatched/(M*N):.1f}%)")
if mismatched > 0:
    row_has_error = (abs_diff > 0.01).any(dim=1)
    error_rows = torch.where(row_has_error)[0]
    print(f"Error rows: {error_rows.tolist()}")

print("\n=== Test 2: K=128 (2 iterations) ===")
M, N, K = 256, 256, 128
kernel2 = matmul_nt_k1(M, N, K, 256, 256, 64)
a = torch.randn(M, K).cuda().bfloat16()
b = torch.randn(N, K).cuda().bfloat16()
c = kernel2(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()
abs_diff = (c - ref_c).abs()
mismatched = (abs_diff > 0.01).sum().item()
print(f"Mismatched: {mismatched}/{M*N} ({100*mismatched/(M*N):.1f}%)")
if mismatched > 0:
    row_has_error = (abs_diff > 0.01).any(dim=1)
    error_rows = torch.where(row_has_error)[0]
    print(f"Error rows: {error_rows.tolist()}")

print("\n=== Test 3: K=192 (3 iterations) ===")
M, N, K = 256, 256, 192
kernel3 = matmul_nt_k1(M, N, K, 256, 256, 64)
a = torch.randn(M, K).cuda().bfloat16()
b = torch.randn(N, K).cuda().bfloat16()
c = kernel3(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()
abs_diff = (c - ref_c).abs()
mismatched = (abs_diff > 0.01).sum().item()
print(f"Mismatched: {mismatched}/{M*N} ({100*mismatched/(M*N):.1f}%)")
if mismatched > 0:
    row_has_error = (abs_diff > 0.01).any(dim=1)
    error_rows = torch.where(row_has_error)[0]
    print(f"Error rows: {error_rows.tolist()}")

# Test with scattered warp layout OFF
@tilelang.jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_SCATTERED_WARP_LAYOUT: False,
})
def matmul_nt_noscatter(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=4, num_xcds=8)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm_v2(A_shared, B_shared, C_local, transpose_B=True, k_pack=2)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm

print("\n=== Test 4: K=1024, NO scattered warp layout ===")
M, N, K = 256, 256, 1024
kernel4 = matmul_nt_noscatter(M, N, K, 256, 256, 64)
a = torch.randn(M, K).cuda().bfloat16()
b = torch.randn(N, K).cuda().bfloat16()
c = kernel4(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()
abs_diff = (c - ref_c).abs()
mismatched = (abs_diff > 0.01).sum().item()
print(f"Mismatched: {mismatched}/{M*N} ({100*mismatched/(M*N):.1f}%)")
if mismatched > 0:
    row_has_error = (abs_diff > 0.01).any(dim=1)
    error_rows = torch.where(row_has_error)[0]
    print(f"Error rows: {error_rows.tolist()}")
