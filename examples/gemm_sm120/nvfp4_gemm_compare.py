"""SM120 NVFP4 GEMM benchmark.

This example demonstrates TileLang's SM120 block-scaled NVFP4 MMA tile op and
optionally compares it with the official CUTLASS GeForce NVFP4 example 79a.

Run from the repository root:

    python examples/gemm_sm120/nvfp4_gemm_compare.py --m 2048 --n 2048 --k 2048 --run-cutlass
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepend_env_path(name: str, value: str) -> bool:
    old = os.environ.get(name, "")
    parts = [p for p in old.split(os.pathsep) if p]
    if value in parts:
        return False
    os.environ[name] = value if not old else value + os.pathsep + old
    return True


def _bootstrap_runtime_env() -> None:
    """Re-exec once with the source-build libraries first in the loader path."""

    if os.environ.get("TILELANG_SM120_NVFP4_BENCH_BOOTSTRAPPED") == "1":
        return

    changed = False
    build_lib = REPO_ROOT / "build" / "lib"
    if build_lib.is_dir():
        changed |= _prepend_env_path("LD_LIBRARY_PATH", str(build_lib))

    system_libstdcpp = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if system_libstdcpp.exists():
        changed |= _prepend_env_path("LD_PRELOAD", str(system_libstdcpp))

    cuda_bin = Path("/usr/local/cuda-12.8/bin")
    if cuda_bin.is_dir():
        changed |= _prepend_env_path("PATH", str(cuda_bin))

    if changed:
        os.environ["TILELANG_SM120_NVFP4_BENCH_BOOTSTRAPPED"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


_bootstrap_runtime_env()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1.0e-3) / 1.0e12


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def tilelang_nvfp4_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    out_dtype,
):
    """C[M, N] = A[M, K] @ B[N, K].T for SM120 NVFP4 block-scaled inputs."""

    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0
    assert block_K % 64 == 0

    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    sf_granularity_k = 16
    sf_words_per_block_k = block_K // 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, sf_words_per_block_k), T.uint32)
            SFB_shared = T.alloc_shared((block_N, sf_words_per_block_k), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for i, k in T.Parallel(block_M, sf_words_per_block_k):
                    SFA_shared[i, k] = SFA[by * block_M + i, ko * sf_words_per_block_k + k]
                for j, k in T.Parallel(block_N, sf_words_per_block_k):
                    SFB_shared[j, k] = SFB[bx * block_N + j, ko * sf_words_per_block_k + k]

                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=False,
                    # Scale buffers are staged per K tile, so scale indexing is local.
                    k_start=0,
                    sf_a_granularity_k=sf_granularity_k,
                    sf_b_granularity_k=sf_granularity_k,
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


def _make_constant_scale_words(rows: int, k: int, byte: int = 0x38) -> torch.Tensor:
    word = byte | (byte << 8) | (byte << 16) | (byte << 24)
    return torch.full((rows, k // 64), word, device="cuda", dtype=torch.uint32)


def _decode_rowmajor_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _verify_tilelang_output(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, out_dtype: torch.dtype) -> None:
    ref = (_decode_rowmajor_fp4(A, A.shape[0], A.shape[1] * 2) @ _decode_rowmajor_fp4(B, B.shape[0], B.shape[1] * 2).T).to(out_dtype)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


def run_tilelang(args: argparse.Namespace) -> tuple[float, float]:
    out_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32

    kernel = tilelang_nvfp4_gemm(
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        out_dtype,
    )

    source = kernel.get_kernel_source()
    if "sm120_mma_sync_blockscaled" not in source:
        raise RuntimeError("TileLang source did not lower to sm120_mma_sync_blockscaled")

    A = _make_packed_fp4(args.m, args.k, seed=args.seed)
    B = _make_packed_fp4(args.n, args.k, seed=args.seed + 1)
    SFA = _make_constant_scale_words(args.m, args.k)
    SFB = _make_constant_scale_words(args.n, args.k)
    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)

    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.verify:
        _verify_tilelang_output(A, B, C, out_torch_dtype)
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


def _find_cmake(args: argparse.Namespace) -> str:
    if args.cmake:
        return args.cmake
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    env_cmake = Path(sys.executable).resolve().parent / "cmake"
    if env_cmake.exists():
        return str(env_cmake)
    raise FileNotFoundError("cmake was not found; pass --cmake")


def _find_nvcc(args: argparse.Namespace) -> str:
    if args.nvcc:
        return args.nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    default_nvcc = Path("/usr/local/cuda-12.8/bin/nvcc")
    if default_nvcc.exists():
        return str(default_nvcc)
    raise FileNotFoundError("nvcc was not found; pass --nvcc")


def _run_command(cmd: list[str], *, cwd: Path) -> str:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.stdout:
        print(completed.stdout, end="")
    completed.check_returncode()
    return completed.stdout


def _cutlass_binary_path(args: argparse.Namespace) -> Path:
    if args.cutlass_binary:
        return Path(args.cutlass_binary)
    return Path(args.cutlass_build_dir) / "examples" / "79_blackwell_geforce_gemm" / "79a_blackwell_geforce_nvfp4_bf16_gemm"


def build_cutlass_79a(args: argparse.Namespace) -> Path:
    binary = _cutlass_binary_path(args)
    if binary.exists() and not args.rebuild_cutlass:
        return binary

    cmake = _find_cmake(args)
    nvcc = _find_nvcc(args)
    build_dir = Path(args.cutlass_build_dir)
    util_include = REPO_ROOT / "3rdparty" / "cutlass" / "tools" / "util" / "include"

    configure_cmd = [
        cmake,
        "-S",
        "3rdparty/cutlass",
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCUTLASS_NVCC_ARCHS=120a",
        "-DCUTLASS_ENABLE_EXAMPLES=ON",
        "-DCUTLASS_ENABLE_TOOLS=ON",
        "-DCUTLASS_ENABLE_LIBRARY=OFF",
        "-DCUTLASS_ENABLE_TESTS=OFF",
        "-DCUTLASS_ENABLE_PROFILER=OFF",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_FLAGS=-I{util_include}",
        f"-DCMAKE_CXX_FLAGS=-I{util_include}",
    ]
    _run_command(configure_cmd, cwd=REPO_ROOT)

    build_cmd = [
        cmake,
        "--build",
        str(build_dir),
        "--target",
        "79a_blackwell_geforce_nvfp4_bf16_gemm",
        "-j",
        str(args.cutlass_build_jobs),
    ]
    _run_command(build_cmd, cwd=REPO_ROOT)

    if not binary.exists():
        raise FileNotFoundError(f"CUTLASS 79a binary was not produced at {binary}")
    return binary


def run_cutlass(args: argparse.Namespace) -> tuple[float, float]:
    binary = build_cutlass_79a(args)
    output = _run_command(
        [
            str(binary),
            f"--m={args.m}",
            f"--n={args.n}",
            f"--k={args.k}",
            f"--iterations={args.cutlass_iterations}",
        ],
        cwd=REPO_ROOT,
    )

    latency_match = re.search(r"Avg runtime:\s*([0-9.eE+-]+)\s*ms", output)
    gflops_match = re.search(r"GFLOPS:\s*([0-9.eE+-]+)", output)
    if latency_match is None or gflops_match is None:
        raise RuntimeError("Could not parse CUTLASS latency/GFLOPS output")
    latency_ms = float(latency_match.group(1))
    return latency_ms, float(gflops_match.group(1)) / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="mean")
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--n-warmup", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--run-cutlass", action="store_true")
    parser.add_argument("--cutlass-iterations", type=int, default=20)
    parser.add_argument("--cutlass-build-dir", default="build-cutlass-sm120")
    parser.add_argument("--cutlass-build-jobs", type=int, default=8)
    parser.add_argument("--cutlass-binary")
    parser.add_argument("--rebuild-cutlass", action="store_true")
    parser.add_argument("--cmake")
    parser.add_argument("--nvcc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability < (12, 0):
        raise RuntimeError(f"SM120 or newer is required, got compute capability {capability}")

    print(f"Shape: M={args.m}, N={args.n}, K={args.k}")
    print(f"TileLang tile: {args.block_m}x{args.block_n}x{args.block_k}, output={args.out_dtype}")

    tilelang_latency_ms, tilelang_tflops = run_tilelang(args)
    print(f"TileLang latency: {tilelang_latency_ms:.4f} ms")
    print(f"TileLang FLOPS: {tilelang_tflops:.2f} TFLOPS")

    if args.run_cutlass:
        cutlass_latency_ms, cutlass_tflops = run_cutlass(args)
        print(f"CUTLASS 79a latency: {cutlass_latency_ms:.4f} ms")
        print(f"CUTLASS 79a FLOPS: {cutlass_tflops:.2f} TFLOPS")
        print(f"TileLang / CUTLASS: {tilelang_tflops / cutlass_tflops:.3f}x")


if __name__ == "__main__":
    main()
