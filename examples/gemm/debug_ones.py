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

a = torch.ones(M, K).cuda().bfloat16()
b = torch.ones(N, K).cuda().bfloat16()

# Run multiple times to check consistency
for run in range(3):
    c = kernel(a, b)
    torch.cuda.synchronize()

    print(f"=== Run {run} ===")
    # Check each row
    wrong_rows = []
    for m in range(M):
        expected = 64.0
        row = c[m, :]
        if torch.isnan(row).any() or not torch.allclose(row, torch.full_like(row, expected), rtol=1e-2, atol=1e-2):
            # Get the actual value
            vals = row[:4].tolist()
            nan_count = torch.isnan(row).sum().item()
            wrong_rows.append((m, vals, nan_count))

    if wrong_rows:
        print(f"  Wrong rows ({len(wrong_rows)}):")
        for m, vals, nan_count in wrong_rows:
            warp_group = m // 64
            row_in_wg = m % 64
            sub_m = row_in_wg // 32
            i_5 = (row_in_wg % 32) // 16
            lane = row_in_wg % 16
            print(f"    Row {m} (wg={warp_group}, sub_m={sub_m}, i_5={i_5}, lane={lane}): "
                  f"vals={vals}, nan_count={nan_count}")
    else:
        print("  All correct!")

# Now test with identity-like A to trace which M rows are actually computed
print("\n=== Row identification test ===")
# Make A such that each row has a unique sum: A[m,:] = m+1
a2 = torch.zeros(M, K).cuda().bfloat16()
for m in range(M):
    a2[m, :] = (m + 1) * 0.01  # Small values to avoid overflow
b2 = torch.ones(N, K).cuda().bfloat16()
# ref: C[m,n] = sum_k A[m,k]*B[n,k] = (m+1)*0.01 * K = (m+1) * 0.64
c2 = kernel(a2, b2)
ref_c2 = a2 @ b2.T
torch.cuda.synchronize()

print("Expected c2[m,0] = (m+1) * 0.64:")
for m in [0, 1, 2, 15, 16, 31, 32, 48, 60, 61, 62, 63, 64, 128, 255]:
    expected = (m + 1) * 0.01 * K
    actual = c2[m, 0].item()
    ref = ref_c2[m, 0].item()
    ok = "OK" if abs(actual - ref) < 0.1 else "WRONG"
    # If wrong, try to figure out what row's data we got
    if ok == "WRONG" and not torch.isnan(c2[m:m+1, 0]):
        # actual = (row+1) * 0.64, so row = actual/0.64 - 1
        inferred_row = actual / (0.01 * K) - 1
        print(f"  c2[{m},0] = {actual:.4f} (expected {ref:.4f}) [{ok}] "
              f"→ looks like data from row {inferred_row:.1f}")
    else:
        print(f"  c2[{m},0] = {actual:.4f} (expected {ref:.4f}) [{ok}]")
