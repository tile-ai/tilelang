"""FP4 / NVFP4 GEMM on SM100/SM110 (B200 / DRIVE Thor) using TCGEN05 MMA.

Default: plain FP4 (float4_e2m1fn) GEMM via tcgen05.mma.kind::f8f6f4 (FP4 staged
gap-expanded in the ALIGN16B unpacksmem SMEM layout).

With ``--nvfp4``: NVFP4 block-scaled GEMM via tcgen05.mma.kind::mxf4nvf4.block_scale
-- FP4 data plus one E4M3 scale factor per 16 elements along K. FP4 operands are
staged DENSE (two e2m1 per byte) as uint8, which the >=8-bit swizzle path lays out
to match the dense e2m1 K-major operand the mxf4nvf4 MMA expects.

Supported: SM100 (B100/B200), SM101/SM110 (DRIVE Thor), SM103 (B300).
"""

import argparse
import os
import time

import torch
import tilelang
import tilelang.language as T


FP4_E2M1_TO_FLOAT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def unpack_fp4_to_float(packed, rows, cols, high_first=False):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    flat = packed.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    first, second = (hi, lo) if high_first else (lo, hi)
    unpacked = torch.stack([first, second], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def pack_e4m3x4_to_u32(values):
    raw = torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    return sum(int(byte) << (8 * i) for i, byte in enumerate(raw.tolist()))


# ---------------------------------------------------------------------------
# Plain FP4 (kind::f8f6f4): A/B are native float4_e2m1fn, transpose_B (TN).
# ---------------------------------------------------------------------------
def matmul_fp4_sm100(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype,
                     accum_dtype, num_stages=1, threads=128):

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.tcgen05_gemm(
                    A_shared, B_shared, C_tmem,
                    transpose_A=False, transpose_B=True,
                    mbar=mbar, clear_accum=(k == 0),
                )
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# NVFP4 (kind::mxf4nvf4.block_scale): FP4 + E4M3 block scale (one per 16 along K).
# FP4 staged DENSE as uint8[M, K/2]; one K64 MMA atom per tile (block_K=64) so the
# 4 scales of that tile pack into a single uint32 SF word.
# ---------------------------------------------------------------------------
def matmul_nvfp4_sm100(M, N, K, block_M, block_N, block_K, out_dtype, accum_dtype, threads=128):
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
            # DENSE-packed FP4 SMEM staging (two e2m1 per byte). The >=8-bit swizzle
            # path gives this the bank swizzle matching CUTLASS's dense e2m1 K-major
            # operand for mxf4nvf4.
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
                T.sync_threads()

                if 32 <= tx and tx < 64:
                    T.tcgen05_gemm_blockscaled(
                        A_shared, B_shared, C_tmem, SFA_tmem, SFB_tmem,
                        transpose_B=True, mbar=mbar,
                        clear_accum=(ko == 0), is_nvfp4=True,
                    )
                T.mbarrier_wait_parity(mbar, ko % 2)
                T.sync_threads()

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def make_random_fp4(rows, cols, mode):
    if mode == "positive":
        lo = torch.randint(0, 8, (rows, cols // 2), device="cuda", dtype=torch.uint8)
        hi = torch.randint(0, 8, (rows, cols // 2), device="cuda", dtype=torch.uint8)
        return (lo | (hi << 4)).to(torch.int8)
    if mode == "low_nibble":
        return torch.randint(0, 16, (rows, cols // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    if mode == "high_nibble":
        return (torch.randint(0, 16, (rows, cols // 2), device="cuda", dtype=torch.uint8) << 4).to(torch.int8)
    if mode == "random":
        return torch.randint(0, 256, (rows, cols // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    raise ValueError(f"Unsupported input mode={mode}")


def _arch_and_flags(args):
    device_major, device_minor = torch.cuda.get_device_capability()
    arch = args.arch or f"sm_{device_major}{device_minor}"
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_include = args.cuda_include or f"{cuda_root}/targets/{args.cuda_target}/include"
    flags = [f"--target-directory={args.cuda_target}", f"-I{cuda_include}"]
    return arch, flags


def run_fp4(args):
    M, N, K = args.m, args.n, args.k
    block_M, block_N, block_K = 128, 64, 128
    arch, device_flags = _arch_and_flags(args)
    print(f"Running SM100 FP4 GEMM (kind::f8f6f4): M={M}, N={N}, K={K}, arch={arch}, input_mode={args.input_mode}")

    func = matmul_fp4_sm100(M, N, K, block_M, block_N, block_K, T.float4_e2m1fn, T.float32, T.float32)
    jit_kernel = tilelang.compile(
        func, out_idx=[2], target={"kind": "cuda", "arch": arch},
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: device_flags,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(42)
    a_packed = make_random_fp4(M, K, args.input_mode)
    b_packed = make_random_fp4(N, K, args.input_mode)

    a_zero = torch.zeros(M, K // 2, device="cuda", dtype=torch.int8)
    b_zero = torch.zeros(N, K // 2, device="cuda", dtype=torch.int8)
    assert jit_kernel(a_zero, b_zero).abs().max().item() == 0.0, "Zero test failed"
    print("[PASS] zeros in -> zeros out")

    c = jit_kernel(a_packed, b_packed)
    ref = unpack_fp4_to_float(a_packed, M, K) @ unpack_fp4_to_float(b_packed, N, K).T
    diff = (c.float() - ref).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    print("[PASS] numerical verification" if max_diff < 1.0 else "[WARN] large diff")

    _bench(lambda: jit_kernel(a_packed, b_packed), M, N, K, args)


def run_nvfp4(args):
    M, N, K = args.m, args.n, args.k
    block_M, block_N, block_K = 128, 128, 64  # block_K=64: 4 scales == 1 SF word per tile
    arch, device_flags = _arch_and_flags(args)
    sf_tiles = (K + block_K - 1) // block_K
    blocks_per_tile = block_K // 16

    print(f"Running SM100 NVFP4 GEMM (kind::mxf4nvf4.block_scale): M={M}, N={N}, K={K}, arch={arch}, "
          f"scale=random")

    kernel = tilelang.compile(
        matmul_nvfp4_sm100(M, N, K, block_M, block_N, block_K, T.float32, T.float32),
        out_idx=[4], target={"kind": "cuda", "arch": arch},
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: device_flags,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)

    def make_sf(rows):
        # Random per-(row, K16-block) E4M3 scales. The reference decodes the actual
        # E4M3-quantized values from the same bytes, so verification stays exact.
        n_blocks = sf_tiles * blocks_per_tile  # == K // 16
        vals = torch.rand(rows, n_blocks, device="cuda") * 1.75 + 0.25  # ~[0.25, 2.0)
        e4 = vals.to(torch.float8_e4m3fn).view(torch.uint8)
        out = torch.zeros(sf_tiles * rows, 4, device="cuda", dtype=torch.uint8)
        for t in range(sf_tiles):
            out[t * rows:(t + 1) * rows, :blocks_per_tile] = e4[:, t * blocks_per_tile:(t + 1) * blocks_per_tile]
        return out.contiguous().view(torch.uint32).reshape(sf_tiles * rows)

    def decode_sf_full(sf, rows):
        raw = sf.view(torch.uint8).reshape(sf_tiles, rows, 4)[:, :, :blocks_per_tile].contiguous()
        sc = raw.view(torch.float8_e4m3fn).float()
        sc = sc.permute(1, 0, 2).reshape(rows, sf_tiles * blocks_per_tile)
        return sc.repeat_interleave(16, dim=1)

    sfa, sfb = make_sf(M), make_sf(N)
    c = kernel(a, b, sfa, sfb)
    sa_full, sb_full = decode_sf_full(sfa, M), decode_sf_full(sfb, N)
    ref = (unpack_fp4_to_float(a, M, K) * sa_full) @ (unpack_fp4_to_float(b, N, K) * sb_full).T
    diff = (c.float() - ref).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    print("[PASS] numerical verification" if max_diff < 1.0 else "[WARN] large diff")

    _bench(lambda: kernel(a, b, sfa, sfb), M, N, K, args)


def _bench(fn, M, N, K, args):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / max(args.iters, 1) * 1000
    print(f"Latency: {elapsed:.4f} ms")
    if elapsed > 0:
        print(f"TFLOPS:  {2 * M * N * K / (elapsed / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP4 / NVFP4 GEMM on SM100/SM110")
    parser.add_argument("--nvfp4", action="store_true",
                        help="run NVFP4 block-scaled GEMM with random E4M3 scales (default: plain FP4)")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--input-mode", default=os.environ.get("TL_FP4_INPUT_MODE", "random"),
                        help="[fp4] random|positive|low_nibble|high_nibble")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("TL_FP4_WARMUP", "20")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("TL_FP4_ITERS", "100")))
    parser.add_argument("--arch", default=os.environ.get("TL_FP4_ARCH"))
    parser.add_argument("--cuda-target", default=os.environ.get("TL_FP4_CUDA_TARGET_DIR", "aarch64-linux"))
    parser.add_argument("--cuda-include", default=os.environ.get("TL_FP4_CUDA_INCLUDE"))
    args = parser.parse_args()

    # Size defaults differ by variant; env vars TL_FP4_{M,N,K} override, --m/--n/--k win.
    default_mnk = 128 if args.nvfp4 else 256
    args.m = args.m if args.m is not None else int(os.environ.get("TL_FP4_M", default_mnk))
    args.n = args.n if args.n is not None else int(os.environ.get("TL_FP4_N", default_mnk))
    args.k = args.k if args.k is not None else int(os.environ.get("TL_FP4_K", default_mnk))

    if args.nvfp4:
        run_nvfp4(args)
    else:
        run_fp4(args)
