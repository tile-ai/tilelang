"""SM120 NVFP4 block-scaled GEMM example.

This example keeps the user-facing kernel small: a non-persistent tiled GEMM
that stages FP4 operands and packed scale words into shared memory, then calls
``T.mma_gemm_blockscaled``.  The SM120 MMA package implementation is selected by
the existing TileLang lowering for ``sf_layout="blockscaled_chunk_kmajor"``.

Run from the repository root:

    python examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py --m 2048 --n 2048 --k 2048 --verify
"""

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.quantize import swizzle_blockscaled_chunk_kmajor_scale_words


_SM120_SCALE_LAYOUT = "blockscaled_chunk_kmajor"
_SM120_THREADS = 128


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1.0e-3) / 1.0e12


@tilelang.jit(out_idx=None)
def sm120_nvfp4_blockscaled_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 256,
    num_stages: int = 2,
    out_dtype=T.bfloat16,
):
    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0
    assert block_M == 128
    assert block_N == 128
    assert block_K == 256
    assert num_stages >= 2

    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    sf_words_per_block_k = block_K // 64
    sf_granularity_k = 16

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=_SM120_THREADS) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, sf_words_per_block_k), T.uint32)
            SFB_shared = T.alloc_shared((block_N, sf_words_per_block_k), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(
                    A[by * block_M, ko * block_K],
                    A_shared,
                    annotations={"prefer_instruction": "tma"},
                )
                T.copy(
                    B[bx * block_N, ko * block_K],
                    B_shared,
                    annotations={"prefer_instruction": "tma"},
                )
                T.copy_ue4m3_scale_tile(SFA, SFA_shared, by, ko)
                T.copy_ue4m3_scale_tile(SFB, SFB_shared, bx, ko)
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
                    sf_layout=_SM120_SCALE_LAYOUT,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


_FP4_E2M1_VALUES = (
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
)


def _make_packed_fp4(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randint(-128, 128, (rows, cols // 2), device="cuda", dtype=torch.int8, generator=generator)


def _make_ones_packed_fp4(rows: int, cols: int) -> torch.Tensor:
    return torch.full((rows, cols // 2), 0x22, device="cuda", dtype=torch.int8)


def _make_constant_scale_words(rows: int, k: int, byte: int = 0x38) -> torch.Tensor:
    word = byte | (byte << 8) | (byte << 16) | (byte << 24)
    return torch.full((rows, k // 64), word, device="cuda", dtype=torch.uint32)


def _pack_scale_words(scale_bytes: torch.Tensor) -> torch.Tensor:
    scale_i64 = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return words.to(torch.uint32).contiguous()


def _make_binary_scale_words(rows: int, k: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    scale_bytes = (
        torch.randint(
            0,
            2,
            (rows, k // 16),
            device="cuda",
            dtype=torch.int64,
            generator=generator,
        )
        * 0x38
    )
    return _pack_scale_words(scale_bytes)


def _decode_rowmajor_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _decode_binary_scale_words(words: torch.Tensor, k: int) -> torch.Tensor:
    w = words.to(torch.int64)
    scale_bytes = torch.empty((words.shape[0], k // 16), device=words.device, dtype=torch.int64)
    scale_bytes[:, 0::4] = w & 0xFF
    scale_bytes[:, 1::4] = (w >> 8) & 0xFF
    scale_bytes[:, 2::4] = (w >> 16) & 0xFF
    scale_bytes[:, 3::4] = (w >> 24) & 0xFF
    return (scale_bytes != 0).to(torch.float32)


def _verify(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,
    scale_mode: str,
    out_dtype: torch.dtype,
) -> None:
    A_full = _decode_rowmajor_fp4(A, A.shape[0], A.shape[1] * 2)
    B_full = _decode_rowmajor_fp4(B, B.shape[0], B.shape[1] * 2)
    if scale_mode == "constant":
        ref = A_full @ B_full.T
    elif scale_mode == "random_binary":
        sfa = _decode_binary_scale_words(SFA, A_full.shape[1])
        sfb = _decode_binary_scale_words(SFB, B_full.shape[1])
        ref = torch.zeros((A_full.shape[0], B_full.shape[0]), device=C.device, dtype=torch.float32)
        for k_sf in range(A_full.shape[1] // 16):
            k0 = k_sf * 16
            k1 = k0 + 16
            ref += (A_full[:, k0:k1] * sfa[:, k_sf].unsqueeze(1)) @ (B_full[:, k0:k1] * sfb[:, k_sf].unsqueeze(1)).T
    else:
        raise ValueError(f"Unsupported scale_mode={scale_mode!r}")
    torch.testing.assert_close(C, ref.to(out_dtype), rtol=0.0, atol=0.0)


def run_tilelang(args: argparse.Namespace) -> tuple[float, float]:
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32
    out_tilelang_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32

    kernel = sm120_nvfp4_blockscaled_gemm(
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        out_tilelang_dtype,
    )

    if args.dump_source:
        source_path = Path(args.dump_source)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(kernel.get_kernel_source())
        print(f"TileLang CUDA source: {source_path}")

    if args.input_mode == "ones":
        A = _make_ones_packed_fp4(args.m, args.k)
        B = _make_ones_packed_fp4(args.n, args.k)
    else:
        A = _make_packed_fp4(args.m, args.k, seed=args.seed)
        B = _make_packed_fp4(args.n, args.k, seed=args.seed + 1)

    if args.scale_mode == "constant":
        SFA_semantic = _make_constant_scale_words(args.m, args.k)
        SFB_semantic = _make_constant_scale_words(args.n, args.k)
    else:
        SFA_semantic = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
        SFB_semantic = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)

    SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic)
    SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic)
    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)

    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.verify:
        _verify(A, B, SFA_semantic, SFB_semantic, C, args.scale_mode, out_torch_dtype)
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
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--input-mode", choices=["random", "ones"], default="random")
    parser.add_argument("--scale-mode", choices=["constant", "random_binary"], default="constant")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="mean")
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--n-warmup", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
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
        f"threads={_SM120_THREADS}, stages={args.num_stages}, output={args.out_dtype}, "
        f"input_mode={args.input_mode}, scale_mode={args.scale_mode}"
    )
    latency_ms, tflops = run_tilelang(args)
    print(f"TileLang latency: {latency_ms:.4f} ms")
    print(f"TileLang FLOPS: {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
