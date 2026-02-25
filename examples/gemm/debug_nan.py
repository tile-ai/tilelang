import os
os.environ["TL_DISABLE_G2S_HOIST"] = "1"
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_SCATTERED_WARP_LAYOUT: True,
})
def matmul_nt(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
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

M, N, K = 256, 256, 64
kernel = matmul_nt(M, N, K, 256, 256, 64)

# Use small known values to trace
a = torch.ones(M, K).cuda().bfloat16()
b = torch.ones(N, K).cuda().bfloat16()
c = kernel(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()

print("=== All-ones test (K=64, expected all 64.0) ===")
print(f"c[0,0]={c[0,0].item()}, ref={ref_c[0,0].item()}")
print(f"c[64,0]={c[64,0].item()}, ref={ref_c[64,0].item()}")
print(f"c[128,0]={c[128,0].item()}, ref={ref_c[128,0].item()}")

nan_count = torch.isnan(c).sum().item()
inf_count = torch.isinf(c).sum().item()
print(f"\nNaN count: {nan_count}/{M*N}")
print(f"Inf count: {inf_count}/{M*N}")

# Check which rows have NaN
nan_rows = torch.isnan(c).any(dim=1)
nan_row_indices = torch.where(nan_rows)[0]
print(f"Rows with NaN: {nan_row_indices.tolist()}")

# Check first few values
print(f"\nc[0:8, 0:4] =")
print(c[0:8, 0:4])
print(f"\nc[60:68, 0:4] =")
print(c[60:68, 0:4])
print(f"\nc[128:136, 0:4] =")
print(c[128:136, 0:4])

# Use random data but check for NaN
print("\n=== Random data test ===")
a = torch.randn(M, K).cuda().bfloat16()
b = torch.randn(N, K).cuda().bfloat16()
c = kernel(a, b)
ref_c = a @ b.T
torch.cuda.synchronize()

nan_count = torch.isnan(c).sum().item()
print(f"NaN count: {nan_count}/{M*N}")
nan_rows = torch.isnan(c).any(dim=1)
nan_row_indices = torch.where(nan_rows)[0]
if len(nan_row_indices) > 0:
    print(f"Rows with NaN: {nan_row_indices.tolist()}")

# Filter out NaN and check remaining
finite_mask = torch.isfinite(c) & torch.isfinite(ref_c)
if finite_mask.sum() > 0:
    diff = (c[finite_mask] - ref_c[finite_mask]).abs()
    mismatched = (diff > 0.01).sum().item()
    print(f"Finite elements: {finite_mask.sum().item()}")
    print(f"Mismatched (finite only): {mismatched}")

# Check if the C store address overlaps — maybe some threads overwrite others
# C store: C[((tid&255)>>6)*16384 + (i_9>>3)*4096 + (tid&15)*256 + (tid>>8)*128 + (i_9&7)*16 + ((tid&63)>>4)*4]
# Each store writes 4 bf16 at consecutive addresses (uint2 = 8 bytes)
print("\n=== Checking C store address uniqueness ===")
c_addrs = set()
duplicates = 0
for tid in range(512):
    for i_9 in range(32):
        addr = (((tid & 255) >> 6) * 16384 +
                (i_9 >> 3) * 4096 +
                (tid & 15) * 256 +
                (tid >> 8) * 128 +
                (i_9 & 7) * 16 +
                ((tid & 63) >> 4) * 4)
        for j in range(4):
            a = addr + j
            if a in c_addrs:
                duplicates += 1
                if duplicates <= 5:
                    print(f"  DUPLICATE at C[{a}] from tid={tid}, i_9={i_9}")
            c_addrs.add(a)

print(f"Total C store addresses: {512*32*4}")
print(f"Unique addresses: {len(c_addrs)}")
print(f"Duplicates: {duplicates}")

# Check coverage
expected = set(range(M*N))
missing = expected - c_addrs
extra = c_addrs - expected
print(f"Missing addresses: {len(missing)}")
print(f"Out-of-range addresses: {len(extra)}")
if missing:
    print(f"  First few missing: {sorted(missing)[:20]}")
