"""Minimal NVFP4 GEMM on SM100/SM110 using TCGEN05 block-scaled MMA.

This validates the frontend-to-PTX path for
``tcgen05.mma.kind::mxf4nvf4.block_scale``.  A/B are declared as native
``T.float4_e2m1fn`` tensors, then viewed as packed bytes for the current SMEM
staging path.  SFA/SFB are uint32 words, each packing four FP8 E4M3 scale
values for one K64 tile.
"""

import os
import time

import torch
import tilelang
import tilelang.language as T


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


def pack_e4m3x4_to_u32(values: tuple[float, float, float, float]) -> int:
    raw = torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    return sum(int(byte) << (8 * i) for i, byte in enumerate(raw.tolist()))


def unpack_fp4_to_float(packed, rows, cols):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    flat = packed.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def matmul_nvfp4_sm100(
    M=128,
    N=128,
    K=64,
    block_M=128,
    block_N=128,
    block_K=64,
    out_dtype=T.float32,
    accum_dtype=T.float32,
    threads=128,
):
    packed_K = K // 2
    packed_block_K = block_K // 2
    k_tiles = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
        B: T.Tensor((N, K), T.float4_e2m1fn),
        SFA: T.Tensor((T.ceildiv(K, block_K) * M,), "uint32"),
        SFB: T.Tensor((T.ceildiv(K, block_K) * N,), "uint32"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_bytes = T.view(A, (M, packed_K), "uint8")
            B_bytes = T.view(B, (N, packed_K), "uint8")

            A_shared = T.alloc_shared((block_M, packed_block_K), "uint8")
            B_shared = T.alloc_shared((block_N, packed_block_K), "uint8")
            SFA_shared = T.alloc_shared((block_M,), "uint32")
            SFB_shared = T.alloc_shared((block_N,), "uint32")

            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            SFA_tmem = T.alloc_tmem([block_M, 4], "uint32")
            SFB_tmem = T.alloc_tmem([block_M, 4], "uint32")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            mbar = T.alloc_barrier(1)

            tx = T.get_thread_binding()
            T.use_swizzle(8)

            for ko in T.serial(k_tiles):
                T.copy(A_bytes[by * block_M, ko * packed_block_K], A_shared)
                T.copy(B_bytes[bx * block_N, ko * packed_block_K], B_shared)
                T.copy(SFA[ko * M + by * block_M], SFA_shared)
                T.copy(SFB[ko * N + bx * block_N], SFB_shared)
                T.sync_threads()

                if tx < 32:
                    T.tcgen05_sf_warp_transpose(SFA_shared)
                    T.tcgen05_sf_warp_transpose(SFB_shared)
                    T.fence_proxy_async()
                    T.tcgen05_cp_warpx4(SFA_shared, SFA_tmem)
                    T.tcgen05_cp_warpx4(SFB_shared, SFB_tmem)

                if 32 <= tx and tx < 64:
                    T.tcgen05_gemm_blockscaled(
                        A_shared,
                        B_shared,
                        C_tmem,
                        SFA_tmem,
                        SFB_tmem,
                        transpose_B=True,
                        mbar=mbar,
                        clear_accum=(ko == 0),
                        is_nvfp4=True,
                    )
                T.mbarrier_wait_parity(mbar, ko % 2)
                T.sync_threads()

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    M = int(os.environ.get("TL_NVFP4_SM100_M", "128"))
    N = int(os.environ.get("TL_NVFP4_SM100_N", "128"))
    K = int(os.environ.get("TL_NVFP4_SM100_K", "64"))
    one_scale = pack_e4m3x4_to_u32((1.0, 1.0, 1.0, 1.0))

    print(f"Running SM100 NVFP4 GEMM: M={M}, N={N}, K={K}")
    kernel = tilelang.compile(
        matmul_nvfp4_sm100(M, N, K),
        out_idx=[4],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    sfa = torch.full(((K + 63) // 64 * M,), one_scale, device="cuda", dtype=torch.uint32)
    sfb = torch.full(((K + 63) // 64 * N,), one_scale, device="cuda", dtype=torch.uint32)

    c = kernel(a, b, sfa, sfb)
    ref = unpack_fp4_to_float(a, M, K) @ unpack_fp4_to_float(b, N, K).T
    diff = (c.float() - ref).abs()
    print(f"[NUMERICAL] max_abs_diff={diff.max().item():.4f}, rel_err={diff.sum().item() / (ref.abs().sum().item() + 1e-10):.6f}")

    warmup = int(os.environ.get("TL_NVFP4_SM100_WARMUP", "20"))
    iters = int(os.environ.get("TL_NVFP4_SM100_ITERS", "100"))
    for _ in range(warmup):
        kernel(a, b, sfa, sfb)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        kernel(a, b, sfa, sfb)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    tflops = 2 * M * N * K / (elapsed_ms / 1e3) / 1e12
    print(f"[BENCH] latency={elapsed_ms:.4f} ms, TFLOPS={tflops:.2f}, iters={iters}")
