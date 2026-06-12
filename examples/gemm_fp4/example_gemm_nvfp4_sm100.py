"""Minimal NVFP4 GEMM on SM100/SM110 using TCGEN05 block-scaled MMA.

This validates the frontend-to-PTX path for
``tcgen05.mma.kind::mxf4nvf4.block_scale``.  A/B are declared and staged as
native ``T.float4_e2m1fn`` tensors so the shared operand gets the swizzled
(align16b -> SWIZZLE_128B) layout the MMA requires.  SFA/SFB are uint32 words,
each packing four FP8 E4M3 scale values for one K64 tile.
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


def unpack_fp4_to_float(packed, rows, cols, high_first=False):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    flat = packed.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    first, second = (hi, lo) if high_first else (lo, hi)
    unpacked = torch.stack([first, second], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def pack_repeated_nibble(codes, cols):
    packed = torch.empty((codes.numel(), cols // 2), device=codes.device, dtype=torch.uint8)
    byte = (codes.to(torch.uint8) & 0x0F) | ((codes.to(torch.uint8) & 0x0F) << 4)
    packed[:] = byte[:, None]
    return packed


def pack_k_pattern(row_count, codes):
    lo = codes[0::2].to(torch.uint8) & 0x0F
    hi = codes[1::2].to(torch.uint8) & 0x0F
    row = lo | (hi << 4)
    return row.repeat(row_count, 1)


def matmul_nvfp4_sm100(
    M=128,
    N=128,
    K=128,
    block_M=128,
    block_N=128,
    block_K=128,
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
            # DENSE-packed FP4 SMEM staging for tcgen05 mxf4nvf4: stage the operands
            # as uint8 [M, K/2] (two e2m1 per byte). With TMA on, infer_shared_layout's
            # >=8-bit branch gives this a half_bank (SW64) swizzle for K/2==64 bytes,
            # matching CUTLASS's dense e2m1 SW64 K-major operand. (mxf4nvf4 consumes
            # FP4 dense, unlike f8f6f4 which uses the 1-byte-per-FP4 align16b layout.)
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
    K = int(os.environ.get("TL_NVFP4_SM100_K", "128"))
    block_K = 128  # mxf4nvf4 needs a swizzled (SWIZZLE_128B) operand; 64B rows -> block_K=128
    device_major, device_minor = torch.cuda.get_device_capability()
    arch = os.environ.get("TL_NVFP4_SM100_ARCH", f"sm_{device_major}{device_minor}")
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_target = os.environ.get("TL_NVFP4_SM100_CUDA_TARGET_DIR", "aarch64-linux")
    cuda_include = os.environ.get("TL_NVFP4_SM100_CUDA_INCLUDE", f"{cuda_root}/targets/{cuda_target}/include")
    one_scale = pack_e4m3x4_to_u32((1.0, 1.0, 1.0, 1.0))

    print(f"Running SM100 NVFP4 GEMM: M={M}, N={N}, K={K}, arch={arch}")
    kernel = tilelang.compile(
        matmul_nvfp4_sm100(M, N, K),
        out_idx=[4],
        target={"kind": "cuda", "arch": arch},
        pass_configs={
            # mxf4nvf4 needs a swizzled K-major operand; sub-byte FP4 only gets the
            # tcgen05mma swizzled SMEM layout on the TMA path (see infer_shared_layout).
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: [
                f"--target-directory={cuda_target}",
                f"-I{cuda_include}",
            ],
        },
    )
    print("Compilation succeeded!")
    if os.environ.get("TL_NVFP4_DUMP_CUDA", "0") != "0":
        with open(os.path.join(os.path.dirname(__file__), "gemm_nvfp4_sm100.cu"), "w") as f:
            f.write(kernel.get_kernel_source())

    torch.manual_seed(0)
    pattern = os.environ.get("TL_NVFP4_SM100_PATTERN")
    if pattern == "ones":
        a = torch.full((M, K // 2), 0x22, device="cuda", dtype=torch.uint8)
        b = torch.full((N, K // 2), 0x22, device="cuda", dtype=torch.uint8)
    elif pattern == "a_ones_b_n":
        a = torch.full((M, K // 2), 0x22, device="cuda", dtype=torch.uint8)
        codes = (torch.arange(N, device="cuda", dtype=torch.uint8) % 7) + 1
        b = pack_repeated_nibble(codes, K)
    elif pattern == "a_ones_b_k":
        a = torch.full((M, K // 2), 0x22, device="cuda", dtype=torch.uint8)
        codes = (torch.arange(K, device="cuda", dtype=torch.uint8) % 7) + 1
        b = pack_k_pattern(N, codes)
    else:
        a = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
        b = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    # SF words must match the kernel signature: ceildiv(K, block_K) words per row.
    # (Uniform scale=1.0 here, so the per-K64-atom SF advance is exercised in M2.)
    sf_tiles = (K + block_K - 1) // block_K
    sfa = torch.full((sf_tiles * M,), one_scale, device="cuda", dtype=torch.uint32)
    sfb = torch.full((sf_tiles * N,), one_scale, device="cuda", dtype=torch.uint32)

    c = kernel(a, b, sfa, sfb)
    ref = unpack_fp4_to_float(a, M, K) @ unpack_fp4_to_float(b, N, K).T
    diff = (c.float() - ref).abs()
    print(f"[NUMERICAL] max_abs_diff={diff.max().item():.4f}, rel_err={diff.sum().item() / (ref.abs().sum().item() + 1e-10):.6f}")
    if os.environ.get("TL_NVFP4_SM100_PRINT_SAMPLE"):
        print(f"[SAMPLE] c00={c[0, 0].item():.4f}, c01={c[0, 1].item():.4f}, ref00={ref[0, 0].item():.4f}")
        print(
            f"[SAMPLE] c_min={c.float().min().item():.4f}, c_max={c.float().max().item():.4f}, "
            f"ref_min={ref.min().item():.4f}, ref_max={ref.max().item():.4f}, "
            f"num_equal={(diff == 0).sum().item()}/{diff.numel()}"
        )
        nz = c.float() != 0
        row_nz = nz.sum(dim=1)
        col_nz = nz.sum(dim=0)
        print(f"[SAMPLE] row_nz_first16={row_nz[:16].tolist()}")
        print(f"[SAMPLE] row_nz_last16={row_nz[-16:].tolist()}")
        print(f"[SAMPLE] col_nz_first32={col_nz[:32].tolist()}")
        print(f"[SAMPLE] col_nz_last32={col_nz[-32:].tolist()}")
        print(f"[SAMPLE] c_row0_first16={c[0, :16].float().tolist()}")
        print(f"[SAMPLE] c_row0_mid48_80={c[0, 48:80].float().tolist()}")
        print(f"[SAMPLE] c_row0_last16={c[0, -16:].float().tolist()}")
        print(f"[SAMPLE] c_col0_first16={c[:16, 0].float().tolist()}")
        print(f"[SAMPLE] ref_row0_first32={ref[0, :32].float().tolist()}")
    for a_high_first in (False, True):
        for b_high_first in (False, True):
            ref_probe = unpack_fp4_to_float(a, M, K, a_high_first) @ unpack_fp4_to_float(b, N, K, b_high_first).T
            probe_diff = (c.float() - ref_probe).abs()
            print(
                f"[PROBE] a_high_first={int(a_high_first)}, b_high_first={int(b_high_first)}, "
                f"max_abs_diff={probe_diff.max().item():.4f}, "
                f"rel_err={probe_diff.sum().item() / (ref_probe.abs().sum().item() + 1e-10):.6f}"
            )

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
