"""SM120 W6A8 block-scaled GEMM example (kind::mxf8f6f4, mixed fp8 x fp6).

Mirrors CUTLASS example 79c (the only SM120 fp6 example there:
mxfp8 e4m3 activations x mxfp6 e3m2 weights -> bf16): A holds FP8
activations (e4m3 by default, e5m2 via --a-dtype), B holds MXFP6 weights as
a packed uint8 blob - an LSB-first 6-bit stream, 4 elements per 3 bytes
(0.75 B/elem, the compression point; torch has no fp6 dtype). Every 32
consecutive K elements share one UE8M0 scale byte; the instruction is
``m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X``.

B's global->shared staging is a SIMT producer writing the b6x16_p32 padded
form (12 payload bytes + 4 padding per 16-element group); the su6 ldmatrix
unpacks the containers into registers (bits[5:0], no shift). The
16U6_ALIGN16B TMA producer is tracked as follow-up work. Engine-vs-engine
(CUTLASS) bitwise comparison lives in
maint/gemm/gemm_sm120/correctness_evaluation_w6a8_vs_cutlass.py.

Run from the repository root:

    python examples/gemm_sm120/sm120_w6a8_mxf8f6f4_gemm.py --m 2048 --n 2048 --k 2048 --verify
"""

import argparse
from pathlib import Path

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_TORCH_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_TILELANG_FP8 = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


def swizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows: int = 128, block_words: int = 2):
    """Pack semantic scale words in SM120 BlockScaledBasicChunk K-major order."""

    if block_rows != 128 or block_words not in (1, 2, 4):
        raise ValueError(
            "SM120 BlockScaledBasicChunk K-major scale packing requires "
            f"block_rows=128 and block_words in (1, 2, 4), got block_rows={block_rows}, block_words={block_words}"
        )
    if not isinstance(words, torch.Tensor):
        raise TypeError(f"words must be a torch.Tensor, got {type(words)!r}")
    if words.dtype != torch.uint32:
        raise TypeError(f"words must have dtype torch.uint32, got {words.dtype}")
    if words.ndim != 2:
        raise ValueError(f"words must be a 2D tensor, got shape {tuple(words.shape)}")

    rows, cols = words.shape
    if cols % block_words != 0:
        raise ValueError(
            f"blockscaled_chunk_kmajor scale storage requires K-word columns multiple of {block_words}, got {tuple(words.shape)}"
        )
    if rows % block_rows != 0:
        padded_rows = (rows + block_rows - 1) // block_rows * block_rows
        padded = torch.zeros((padded_rows, cols), dtype=words.dtype, device=words.device)
        padded[:rows] = words
        words = padded
        rows = padded_rows

    row_blocks = rows // block_rows
    source = words.contiguous().reshape(row_blocks, 4, 32, cols)
    return source.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1.0e-3) / 1.0e12


@tilelang.jit
def sm120_w6a8_mxf8f6f4_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    a_dtype_name: str = "e4m3",
    b_flavor: str = "e3m2",
    out_dtype=T.bfloat16,
):
    assert N % 8 == 0, "N must be a multiple of 8 (16-byte aligned output rows)"
    if M % block_M != 0 and N % block_N != 0:
        raise ValueError(
            "simultaneous M and N tail tiles are not supported yet (a copy-lowering "
            "boundary bug is tracked upstream); pad M or N to a multiple of 128"
        )
    # The fp6 packer works in 16-element groups and the scale words cover
    # K=128; keep K a multiple of 128 (also future-proof for the 16U6 TMA).
    assert K % 128 == 0, "MXFP6 staging requires K to be a multiple of 128 elements"
    assert b_flavor in ("e2m3", "e3m2")
    assert K % block_K == 0
    # Same single-atom scale-staging constraint as the other SM120 examples.
    assert block_M == 128
    assert block_N == 128
    assert block_K in (128, 256), "one uint32 ue8m0 scale word covers K=128"
    assert num_stages >= 2

    a_dtype = getattr(T, _TILELANG_FP8[a_dtype_name])
    b_smem_dtype = getattr(T, {"e2m3": "float6_e2m3fn_unpacked", "e3m2": "float6_e3m2fn_unpacked"}[b_flavor])
    accum_dtype = T.float32
    sf_words_per_block_k = block_K // 128
    sf_granularity_k = 32
    M_pad = -(-M // block_M) * block_M
    N_pad = -(-N // block_N) * block_N
    k_blocks = K // block_K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), a_dtype),
        B_blob: T.Tensor((N, K * 3 // 4), T.uint8),
        SFA: T.Tensor((M_pad * k_blocks, sf_words_per_block_k), T.uint32),
        SFB: T.Tensor((N_pad * k_blocks, sf_words_per_block_k), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype)
            B_shared = T.alloc_shared((block_N, block_K), b_smem_dtype)
            SFA_shared = T.alloc_shared((block_M, sf_words_per_block_k), T.uint32)
            SFB_shared = T.alloc_shared((block_N, sf_words_per_block_k), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                # SIMT producer of the b6x16_p32 padded form: 12 payload
                # bytes + 4 padding bytes per 16-element group.
                for j, g, b in T.Parallel(block_N, block_K // 16, 12):
                    B_shared[j, 16 * g + b] = T.reinterpret(b_smem_dtype, B_blob[bx * block_N + j, ko * (block_K * 3 // 4) + 12 * g + b])

                for r, w in T.Parallel(block_M, sf_words_per_block_k):
                    SFA_shared[r, w] = SFA[(by * k_blocks + ko) * block_M + r, w]
                for r, w in T.Parallel(block_N, sf_words_per_block_k):
                    SFB_shared[r, w] = SFB[(bx * k_blocks + ko) * block_N + r, w]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=False,
                    k_start=ko * block_K,
                    sf_a_granularity_k=sf_granularity_k,
                    sf_b_granularity_k=sf_granularity_k,
                    sf_layout="blockscaled_chunk_kmajor",
                    scale_dtype="ue8m0",
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _make_packed_fp6(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randint(0, 256, (rows, cols * 3 // 4), device="cuda", dtype=torch.uint8, generator=generator)


def _make_fp8(rows: int, cols: int, dtype_name: str, *, seed: int) -> torch.Tensor:
    # Small integers, exact in both fp8 formats, keeping the bitwise band exact.
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    values = torch.randint(-4, 5, (rows, cols), device="cuda", dtype=torch.int64, generator=generator)
    return values.to(torch.float32).to(_TORCH_FP8[dtype_name])


def _pack_scale_words(scale_bytes: torch.Tensor) -> torch.Tensor:
    scale_i64 = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return words.to(torch.uint32).contiguous()


def _make_pow2_scale_words(rows: int, k: int, *, seed: int) -> torch.Tensor:
    # Scales from {0.5, 1, 2} (exact powers of two) so verification is bitwise.
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    choices = torch.tensor([0x7E, 0x7F, 0x80], device="cuda", dtype=torch.int64)
    idx = torch.randint(0, choices.numel(), (rows, k // 32), device="cuda", dtype=torch.int64, generator=generator)
    return _pack_scale_words(choices[idx])


def _decode_fp6_blob(blob: torch.Tensor, rows: int, cols: int, flavor: str) -> torch.Tensor:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from examples.dequantize_gemm.quantize import decode_fp6_values, unpack_fp6_bytes

    codes = unpack_fp6_bytes(blob.cpu(), cols)
    return decode_fp6_values(codes, flavor).to(blob.device)


def _decode_ue8m0_scale_words(words: torch.Tensor, k: int) -> torch.Tensor:
    w = words.to(torch.int64)
    scale_bytes = torch.empty((words.shape[0], k // 32), device=words.device, dtype=torch.int64)
    scale_bytes[:, 0::4] = w & 0xFF
    scale_bytes[:, 1::4] = (w >> 8) & 0xFF
    scale_bytes[:, 2::4] = (w >> 16) & 0xFF
    scale_bytes[:, 3::4] = (w >> 24) & 0xFF
    return torch.pow(2.0, (scale_bytes - 127).to(torch.float32))


def _verify(
    A: torch.Tensor,
    B_blob: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,
    out_dtype: torch.dtype,
    b_flavor: str,
) -> None:
    A_full = A.to(torch.float32)
    B_full = _decode_fp6_blob(B_blob, B_blob.shape[0], B_blob.shape[1] * 4 // 3, b_flavor)
    sfa = _decode_ue8m0_scale_words(SFA, A_full.shape[1])
    sfb = _decode_ue8m0_scale_words(SFB, B_full.shape[1])
    ref = torch.zeros((A_full.shape[0], B_full.shape[0]), device=C.device, dtype=torch.float32)
    for k_sf in range(A_full.shape[1] // 32):
        k0 = k_sf * 32
        k1 = k0 + 32
        ref += (A_full[:, k0:k1] * sfa[:, k_sf].unsqueeze(1)) @ (B_full[:, k0:k1] * sfb[:, k_sf].unsqueeze(1)).T
    torch.testing.assert_close(C, ref.to(out_dtype), rtol=0.0, atol=0.0)


def run_tilelang(args: argparse.Namespace) -> tuple[float, float]:
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32
    out_tilelang_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32

    kernel = sm120_w6a8_mxf8f6f4_gemm(
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        args.a_dtype,
        args.b_flavor,
        out_tilelang_dtype,
    )

    if args.dump_source:
        source_path = Path(args.dump_source)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(kernel.get_kernel_source())
        print(f"TileLang CUDA source: {source_path}")

    sf_words_per_block_k = args.block_k // 128
    if args.from_bf16:
        # bf16 activations -> MXFP8, bf16 weights -> MXFP6; the bitwise
        # engine-vs-engine version of this band is the maint comparison.
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from examples.dequantize_gemm.quantize import (
            quantize_bf16_to_mxfp6_blockscaled,
            quantize_bf16_to_mxfp8_blockscaled,
        )

        x_a = (torch.randn(args.m, args.k, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
        x_b = (torch.randn(args.n, args.k, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
        A, SFA_packed, sfa_bytes = quantize_bf16_to_mxfp8_blockscaled(
            x_a, dtype=args.a_dtype, block_words=sf_words_per_block_k, return_scale_bytes=True
        )
        B_cpu, SFB_packed, sfb_bytes = quantize_bf16_to_mxfp6_blockscaled(
            x_b.cpu().float(), dtype=args.b_flavor, block_words=sf_words_per_block_k, return_scale_bytes=True
        )
        B = B_cpu.cuda()
        sfb_bytes = sfb_bytes.cuda()
        SFA = SFA_packed.reshape(-1, sf_words_per_block_k)
        SFB = SFB_packed.cuda().reshape(-1, sf_words_per_block_k)
    else:
        A = _make_fp8(args.m, args.k, args.a_dtype, seed=args.seed)
        B = _make_packed_fp6(args.n, args.k, seed=args.seed + 1)
        SFA_semantic = _make_pow2_scale_words(args.m, args.k, seed=args.seed + 100)
        SFB_semantic = _make_pow2_scale_words(args.n, args.k, seed=args.seed + 200)

        # Zero-copy tile-rows view of the packed layout (see the kernel docstring).
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic, block_words=sf_words_per_block_k).reshape(-1, sf_words_per_block_k)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic, block_words=sf_words_per_block_k).reshape(-1, sf_words_per_block_k)
    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)

    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.verify:
        if args.from_bf16:
            a_deq = A.to(torch.float32) * torch.pow(2.0, (sfa_bytes.to(torch.int32) - 127).to(torch.float32)).repeat_interleave(32, dim=1)
            b_deq = _decode_fp6_blob(B, args.n, args.k, args.b_flavor) * torch.pow(
                2.0, (sfb_bytes.to(torch.int32) - 127).to(torch.float32)
            ).repeat_interleave(32, dim=1)
            ref = a_deq @ b_deq.T
            # Quantized math is exact in fp32; the only divergence is the
            # bf16 output rounding (2^-8 relative).
            torch.testing.assert_close(C.float(), ref, rtol=8e-3, atol=1e-2)
        else:
            _verify(A, B, SFA_semantic, SFB_semantic, C, out_torch_dtype, args.b_flavor)
        print("TileLang correctness: passed")

    latency_ms = do_bench(
        lambda: kernel(A, B, SFA, SFB, C),
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        _n_warmup=args.n_warmup,
        _n_repeat=args.n_repeat,
        backend=args.backend,
        return_mode=args.return_mode,
    )
    return latency_ms, _tflops(args.m, args.n, args.k, latency_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--a-dtype", choices=["e4m3", "e5m2"], default="e4m3", help="activation (A) fp8 dtype")
    parser.add_argument("--b-flavor", choices=["e2m3", "e3m2"], default="e3m2", help="weight (B) fp6 flavor")
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="mean")
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--n-warmup", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--from-bf16", action="store_true", help="quantize bf16 inputs instead of synthetic data")
    parser.add_argument("--dump-source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability < (12, 0):
        raise RuntimeError(f"SM120 or newer is required, got compute capability {capability}")

    print(f"Shape: M={args.m}, N={args.n}, K={args.k}")
    print(
        f"TileLang tile: {args.block_m}x{args.block_n}x{args.block_k}, "
        f"threads=128, stages={args.num_stages}, activations=mxfp8-{args.a_dtype}, weights=mxfp6-{args.b_flavor}, "
        f"output={args.out_dtype}"
    )
    latency_ms, tflops = run_tilelang(args)
    print(f"TileLang latency: {latency_ms:.4f} ms")
    print(f"TileLang FLOPS: {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
