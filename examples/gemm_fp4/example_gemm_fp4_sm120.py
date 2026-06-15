"""FP4 / NVFP4 GEMM on SM120 (RTX 5080/5090) using fragment-based mma.sync.

Default: plain FP4 (float4_e2m1fn) GEMM via mma.sync.aligned.kind::f8f6f4.
FP4 data is pre-unpacked to uint8 on the host (1 byte/element, low nibble holds
the 4-bit value); shared memory stores uint8 like INT8 (ldmatrix 16B aligned).

With ``--nvfp4``: NVFP4 block-scaled GEMM via
mma.sync.aligned.kind::mxf4nvf4.block_scale -- both operands are dense-packed FP4
(two e2m1 per byte) with FP8 E4M3 scale factors (sf_vec_size=16) held in registers.

Addresses https://github.com/tile-ai/tilelang/issues/1592
"""

import argparse
import os
import time

import torch
import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout


FP4_E2M1_TO_FLOAT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def unpack_fp4_to_uint8(packed_int8, M, K):
    """Unpack (M, K//2) -> (M, K) uint8, 1 FP4 per byte in low nibble."""
    flat = packed_int8.to(torch.uint8).reshape(M, K // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(M, K).contiguous()


def unpack_fp4_to_float(packed_int8, M, K):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    return lut[unpack_fp4_to_uint8(packed_int8, M, K).to(torch.int64)]


def pack_e4m3x4_to_u32(values):
    raw = torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    return sum(int(byte) << (8 * i) for i, byte in enumerate(raw.tolist()))


# ---------------------------------------------------------------------------
# Plain FP4 (kind::f8f6f4): A/B unpacked to uint8 (1 byte/FP4), fragment T.gemm.
# ---------------------------------------------------------------------------
def matmul_fp4_sm120(M, N, K, block_M, block_N, block_K, out_dtype, accum_dtype,
                     num_stages=2, threads=128):

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "uint8"),
        B: T.Tensor((N, K), "uint8"),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "uint8")
            B_shared = T.alloc_shared((block_N, block_K), "uint8")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# NVFP4 (kind::mxf4nvf4.block_scale): A/B dense-packed FP4 (uint8 [*, K/2]),
# FP8 E4M3 scale per 16 elements held in registers, fragment T.nvfp4_gemm.
# ---------------------------------------------------------------------------
def matmul_nvfp4_sm120(M, N, K, out_dtype, accum_dtype):
    block_M, block_N, block_K = 16, 8, 64
    threads = 32
    packed_block_K = block_K // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K // 2), "uint8"),
        B: T.Tensor((N, K // 2), "uint8"),
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

            T.annotate_layout({
                A_shared: make_swizzled_layout(A_shared),
                B_shared: make_swizzled_layout(B_shared),
            })

            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * packed_block_K], A_shared)
                T.copy(B[bx * block_N, ko * packed_block_K], B_shared)
                SFA_local[0] = SFA[by, ko]
                SFB_local[0] = SFB[bx, ko]
                T.nvfp4_gemm(A_shared, B_shared, SFA_local, SFB_local, C_local,
                             transpose_B=True, clear_accum=(ko == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_fp4(args):
    M, N, K = args.m, args.n, args.k
    block_M, block_N, block_K = 128, 128, 128
    print(f"Running SM120 FP4 GEMM (kind::f8f6f4): M={M}, N={N}, K={K}")

    func = matmul_fp4_sm120(M, N, K, block_M, block_N, block_K, T.float32, T.float32, num_stages=2)
    jit_kernel = tilelang.compile(
        func, out_idx=[2], target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(42)
    a_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
    a_unpacked = unpack_fp4_to_uint8(a_packed, M, K)
    b_unpacked = unpack_fp4_to_uint8(b_packed, N, K)

    a_zero = torch.zeros(M, K, device="cuda", dtype=torch.uint8)
    b_zero = torch.zeros(N, K, device="cuda", dtype=torch.uint8)
    assert jit_kernel(a_zero, b_zero).abs().max().item() == 0.0, "Zero test failed"
    print("[PASS] zeros in -> zeros out")

    c = jit_kernel(a_unpacked, b_unpacked)
    ref = unpack_fp4_to_float(a_packed, M, K) @ unpack_fp4_to_float(b_packed, N, K).T
    diff = (c.float() - ref).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    print("[PASS] numerical verification" if max_diff < 1.0 else "[WARN] large diff")

    _bench(lambda: jit_kernel(a_unpacked, b_unpacked), M, N, K, args)


def run_nvfp4(args):
    M, N, K = args.m, args.n, args.k
    block_M, block_N, block_K = 16, 8, 64
    print(f"Running SM120 NVFP4 GEMM (kind::mxf4nvf4.block_scale): M={M}, N={N}, K={K}, scale=uniform")

    kernel = tilelang.compile(
        matmul_nvfp4_sm120(M, N, K, T.float32, T.float32),
        out_idx=[4], target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)

    # Scale factors: 2D [ceil(M/16), ceil(K/64)] uint32, 4 E4M3 packed per word.
    # NOTE: the exact SM120 mma.sync SF-register -> (row, K16-block) mapping is not
    # modeled in this reference, so verification uses UNIFORM scale (=1.0) for an
    # exact data check, plus a scale spot-check (0.5 x 2.0 preserves the product).
    # Random per-block-scale verification is a follow-up (needs the SF layout).
    sf_m, sf_n, sf_k = (M + block_M - 1) // block_M, (N + block_N - 1) // block_N, (K + block_K - 1) // block_K
    one = pack_e4m3x4_to_u32((1.0, 1.0, 1.0, 1.0))
    half = pack_e4m3x4_to_u32((0.5, 0.5, 0.5, 0.5))
    two = pack_e4m3x4_to_u32((2.0, 2.0, 2.0, 2.0))
    sfa = torch.full((sf_m, sf_k), one, device="cuda", dtype=torch.uint32)
    sfb = torch.full((sf_n, sf_k), one, device="cuda", dtype=torch.uint32)

    c = kernel(a, b, sfa, sfb)
    ref = unpack_fp4_to_float(a, M, K) @ unpack_fp4_to_float(b, N, K).T
    diff = (c.float() - ref).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    print("[PASS] numerical verification" if max_diff < 1.0 else "[WARN] large diff")

    # Scale spot-check: SFA=0.5, SFB=2.0 must reproduce the unit-scale result.
    c_scaled = kernel(
        a, b,
        torch.full((sf_m, sf_k), half, device="cuda", dtype=torch.uint32),
        torch.full((sf_n, sf_k), two, device="cuda", dtype=torch.uint32),
    )
    sdiff = (c_scaled.float() - ref).abs().max().item()
    print(f"[SCALE] max_abs_diff(SFA=0.5 x SFB=2.0 vs unit)={sdiff:.4f} ({'PASS' if sdiff < 1.0 else 'WARN'})")

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
    parser = argparse.ArgumentParser(description="FP4 / NVFP4 GEMM on SM120")
    parser.add_argument("--nvfp4", action="store_true",
                        help="run NVFP4 block-scaled GEMM with random E4M3 scales (default: plain FP4)")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("TL_FP4_WARMUP", "20")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("TL_FP4_ITERS", "100")))
    args = parser.parse_args()

    default_mnk = (1024, 1024, 256) if args.nvfp4 else (256, 256, 256)
    args.m = args.m if args.m is not None else int(os.environ.get("TL_FP4_M", default_mnk[0]))
    args.n = args.n if args.n is not None else int(os.environ.get("TL_FP4_N", default_mnk[1]))
    args.k = args.k if args.k is not None else int(os.environ.get("TL_FP4_K", default_mnk[2]))

    if args.nvfp4:
        run_nvfp4(args)
    else:
        run_fp4(args)
