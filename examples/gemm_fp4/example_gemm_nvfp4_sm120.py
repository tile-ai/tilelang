"""Minimal NVFP4 GEMM on SM120 using block-scaled MMA.

FlashInfer/TRT-LLM use "NVFP4" as packed FP4 E2M1 data plus FP8 E4M3/UE4M3
scale factors with ``sf_vec_size=16``.  The corresponding SM120 Tensor Core
instruction is ``mma.sync.aligned.kind::mxf4nvf4.block_scale``: both MMA
operands are FP4, the scale factors are FP8 E4M3, and accumulation is FP32.

That is distinct from the non-scaled A8W4 examples, which use FP8 x FP4
``kind::f8f6f4`` MMA.  FP16/BF16 usually appear around NVFP4 as source/output
types for quantization or epilogues, not as the direct mxf4nvf4 MMA operands.

This example intentionally starts with one SM120 MMA atom (m16n8k64) and a
uniform scale register before wiring NVFP4 into the generic GEMM API.
"""

import os
import time

import torch
import tilelang
import tilelang.language as T
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


def unpack_fp4_to_uint8(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack (M, K//2) uint8 -> (M, K) uint8 FP4 codes for the reference."""
    flat = packed_int8.to(torch.uint8).reshape(M, K // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(M, K).contiguous()


def unpack_fp4_to_float(packed_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unpack FP4 E2M1 data to float32 through a lookup table."""
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    unpacked = unpack_fp4_to_uint8(packed_int8, M, K).to(torch.int64)
    return lut[unpacked]


def pack_e4m3x4_to_u32(values: tuple[float, float, float, float]) -> int:
    """Pack four FP8 E4M3 scale values into the uint32 register used by MMA.SF."""
    raw = torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    return sum(int(byte) << (8 * i) for i, byte in enumerate(raw.tolist()))


UE4M3_ONE_X4 = pack_e4m3x4_to_u32((1.0, 1.0, 1.0, 1.0))
UE4M3_HALF_X4 = pack_e4m3x4_to_u32((0.5, 0.5, 0.5, 0.5))
UE4M3_TWO_X4 = pack_e4m3x4_to_u32((2.0, 2.0, 2.0, 2.0))


def matmul_nvfp4_sm120(M=16, N=8, K=64, out_dtype=T.float32, accum_dtype=T.float32):
    block_M, block_N, block_K = 16, 8, 64
    threads = 32
    packed_K = K // 2
    packed_block_K = block_K // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, packed_K), "uint8"),
        B: T.Tensor((N, packed_K), "uint8"),
        SFA: T.Tensor((T.ceildiv(M, block_M), T.ceildiv(K, block_K)), "uint32"),
        SFB: T.Tensor((T.ceildiv(N, block_N), T.ceildiv(K, block_K)), "uint32"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, packed_block_K), "uint8")
            B_shared = T.alloc_shared((block_N, packed_block_K), "uint8")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
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
                T.copy(A[by * block_M, ko * packed_block_K], A_shared)
                T.copy(B[bx * block_N, ko * packed_block_K], B_shared)
                SFA_local[0] = SFA[by, ko]
                SFB_local[0] = SFB[bx, ko]
                T.nvfp4_gemm(A_shared, B_shared, SFA_local, SFB_local, C_local, transpose_B=True, clear_accum=(ko == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    M, N, K = 1024, 1024, 256
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
    a_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    
    # To fit for mma.m16n8k64, we need to pad the scale factors to the nearest multiples accordingly.
    scale_shape_a = ((M + 15) // 16, (K + 63) // 64)
    scale_shape_b = ((N + 7) // 8, (K + 63) // 64)
    sfa = torch.full(scale_shape_a, UE4M3_ONE_X4, device="cuda", dtype=torch.uint32)
    sfb = torch.full(scale_shape_b, UE4M3_ONE_X4, device="cuda", dtype=torch.uint32)

    a_zero = torch.zeros((M, K // 2), device="cuda", dtype=torch.uint8)
    b_zero = torch.zeros((N, K // 2), device="cuda", dtype=torch.uint8)
    c_zero = kernel(a_zero, b_zero, sfa, sfb)
    print(f"[ZERO] max_abs={c_zero.abs().max().item():.4f}")

    one_pair = (2 | (2 << 4))  # two FP4 E2M1 1.0 values packed in one byte
    a_one = torch.full((M, K // 2), one_pair, device="cuda", dtype=torch.uint8)
    b_one = torch.full((N, K // 2), one_pair, device="cuda", dtype=torch.uint8)
    c_one = kernel(a_one, b_one, sfa, sfb)
    print(f"[ONE] first={c_one[0, 0].item():.4f}, expected={float(K):.4f}")

    # Exercise the scale path explicitly.  SFA=0.5 and SFB=2.0 should preserve
    # the effective product for all-one FP4 inputs.
    sfa_half = torch.full(scale_shape_a, UE4M3_HALF_X4, device="cuda", dtype=torch.uint32)
    sfb_two = torch.full(scale_shape_b, UE4M3_TWO_X4, device="cuda", dtype=torch.uint32)
    c_scaled = kernel(a_one, b_one, sfa_half, sfb_two)
    print(f"[SCALE] first={c_scaled[0, 0].item():.4f}, expected={float(K):.4f}")

    c = kernel(a_packed, b_packed, sfa, sfb)
    ref = unpack_fp4_to_float(a_packed, M, K) @ unpack_fp4_to_float(b_packed, N, K).T
    diff = (c.float() - ref).abs()
    print(f"[NUMERICAL] max_abs_diff={diff.max().item():.4f}, rel_err={diff.sum().item() / (ref.abs().sum().item() + 1e-10):.6f}")

    warmup = int(os.environ.get("TL_NVFP4_WARMUP", "20"))
    iters = int(os.environ.get("TL_NVFP4_ITERS", "100"))
    for _ in range(warmup):
        kernel(a_packed, b_packed, sfa, sfb)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        kernel(a_packed, b_packed, sfa, sfb)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    tflops = 2 * M * N * K / (elapsed_ms / 1e3) / 1e12
    print(f"[BENCH] latency={elapsed_ms:.4f} ms, TFLOPS={tflops:.2f}, iters={iters}")
