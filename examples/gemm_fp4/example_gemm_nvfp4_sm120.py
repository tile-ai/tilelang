"""Minimal NVFP4 GEMM on SM120 using block-scaled MMA.

This example exercises ``mma.sync.aligned.kind::mxf4nvf4.block_scale`` for
FP4 E2M1 inputs with UE4M3 scale factors.  It intentionally starts with a
single SM120 MMA shape (m16n8k64) before wiring NVFP4 into the generic GEMM API.
"""

import os
import torch
import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout


FP4_E2M1_TO_FLOAT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# UE4M3 encoding for scale 1.0, packed four times into one uint32 register.
UE4M3_ONE_X4 = 0x38383838


def unpack_fp4_to_uint8(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack (M, K//2) int8 -> (M, K) uint8, one FP4 value per byte."""
    flat = packed_int8.to(torch.uint8).reshape(M, K // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(M, K).contiguous()


def unpack_fp4_to_float(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack FP4 E2M1 data to float32 through a lookup table."""
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    unpacked = unpack_fp4_to_uint8(packed_int8, M, K).to(torch.int64)
    return lut[unpacked]


def matmul_nvfp4_sm120(M=16, N=8, K=64, out_dtype=T.float32, accum_dtype=T.float32):
    block_M, block_N, block_K = 16, 8, 64
    threads = 32

    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=block_M,
        warp_col_tiles=block_N,
        chunk=block_K,
        is_blockscaled=True,
        scale_dtype=T.float8_e4m3,
        scale_vec_size=16,
    )

    local_size_a = emitter.local_size_a
    local_size_b = emitter.local_size_b
    local_size_c = emitter.local_size_out

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "uint8"),
        B: T.Tensor((N, K), "uint8"),
        SFA: T.Tensor((T.ceildiv(M, block_M), T.ceildiv(K, block_K)), "uint32"),
        SFB: T.Tensor((T.ceildiv(N, block_N), T.ceildiv(K, block_K)), "uint32"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "uint8")
            B_shared = T.alloc_shared((block_N, block_K), "uint8")
            C_shared = T.alloc_shared((1, 1, block_M, block_N), out_dtype)
            A_local = T.alloc_local((local_size_a,), T.float4_e2m1fn)
            B_local = T.alloc_local((local_size_b,), T.float4_e2m1fn)
            C_local = T.alloc_local((local_size_c,), accum_dtype)
            SFA_local = T.alloc_local((1,), "uint32")
            SFB_local = T.alloc_local((1,), "uint32")

            T.annotate_layout(
                {
                    A_shared: make_swizzled_layout(A_shared),
                    B_shared: make_swizzled_layout(B_shared),
                }
            )

            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                SFA_local[0] = SFA[by, ko]
                SFB_local[0] = SFB[bx, ko]
                emitter.ldmatrix_a(A_local, A_shared, 0)
                emitter.ldmatrix_b(B_local, B_shared, 0)
                emitter.mma_blockscaled(A_local, B_local, C_local, SFA_local, SFB_local)

            emitter.stmatrix(C_local, C_shared)
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[0, 0, i, j]

    return main


if __name__ == "__main__":
    M, N, K = 16, 8, 64
    print(f"Running NVFP4 SM120 block-scaled MMA: M={M}, N={N}, K={K}")

    kernel = tilelang.compile(
        matmul_nvfp4_sm120(M, N, K),
        out_idx=[4],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded!")

    if os.environ.get("TL_NVFP4_DUMP_CUDA", "0") != "0":
        with open(os.path.join(os.path.dirname(__file__), "gemm_nvfp4_sm120.cu"), "w") as f:
            f.write(kernel.get_kernel_source())

    torch.manual_seed(0)
    a_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    a_unpacked = unpack_fp4_to_uint8(a_packed, M, K)
    b_unpacked = unpack_fp4_to_uint8(b_packed, N, K)

    sfa = torch.full((1, 1), UE4M3_ONE_X4, device="cuda", dtype=torch.uint32)
    sfb = torch.full((1, 1), UE4M3_ONE_X4, device="cuda", dtype=torch.uint32)

    a_zero = torch.zeros((M, K), device="cuda", dtype=torch.uint8)
    b_zero = torch.zeros((N, K), device="cuda", dtype=torch.uint8)
    c_zero = kernel(a_zero, b_zero, sfa, sfb)
    print(f"[ZERO] max_abs={c_zero.abs().max().item():.4f}")

    a_one = torch.full((M, K), 2, device="cuda", dtype=torch.uint8)
    b_one = torch.full((N, K), 2, device="cuda", dtype=torch.uint8)
    c_one = kernel(a_one, b_one, sfa, sfb)
    print(f"[ONE] first={c_one[0, 0].item():.4f}, expected={float(K):.4f}")

    c = kernel(a_unpacked, b_unpacked, sfa, sfb)
    ref = unpack_fp4_to_float(a_packed, M, K) @ unpack_fp4_to_float(b_packed, N, K).T
    diff = (c.float() - ref).abs()
    print(f"[NUMERICAL] max_abs_diff={diff.max().item():.4f}, rel_err={diff.sum().item() / (ref.abs().sum().item() + 1e-10):.6f}")
