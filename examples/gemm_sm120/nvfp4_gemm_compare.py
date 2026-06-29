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
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1.0e-3) / 1.0e12


def _early_bench_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--enable-tma", action="store_true")
    parser.add_argument("--enable-warp-specialized", action="store_true")
    parser.add_argument(
        "--manual-ws2-ab-copy-view",
        choices=[
            "legacy_b16",
            "legacy_b16_single_omma",
            "fp4_ldsm",
            "fp4_ldsm_single_omma",
            "fp4_ldsm_noshift",
            "fp4_ldsm_single_omma_noshift",
            "fp4_ldsm_cute_rowstart",
            "fp4_ldsm_cute_rowstart_noshift",
            "fp4_ldsm_cute_rowstart_aonly",
            "fp4_ldsm_cute_rowstart_bonly",
            "fp4_ldsm_cute_rowstart_caccum",
            "fp4_ldsm_cute_rowstart_single_omma",
            "fp4_ldsm_cute_rowstart_single_omma_noshift",
            "fp4_ldsm_cute_rowstart_single_omma_aonly",
            "fp4_ldsm_cute_rowstart_single_omma_bonly",
            "fp4_ldsm_single_omma_real_scratch",
            "fp4_ldsm_single_omma_scratch_ones",
            "fp4_ldsm_single_omma_scratch_shifted_ones",
        ],
        default="legacy_b16",
    )
    parser.add_argument("--maxrregcount", type=int)
    parser.add_argument("--ptxas-verbose", action="store_true")
    parser.add_argument("--manual-ws2-scratch-byte", type=lambda x: int(x, 0))
    parser.add_argument(
        "--manual-ws2-sf-layout",
        choices=["rowmajor", "cutlass_128x4"],
        default="rowmajor",
    )
    parser.add_argument("--micro-pipeline", default="none")
    parser.add_argument("--manual-ws2-reg-debug", action="store_true")
    parser.add_argument("--manual-ws2-reg-debug-mode", choices=["value", "tag"], default="value")
    args, _ = parser.parse_known_args()
    return args


_EARLY_BENCH_OPTIONS = _early_bench_options()


_P64_LDSM_COPY_VIEWS = {
    "fp4_ldsm",
    "fp4_ldsm_single_omma",
    "fp4_ldsm_noshift",
    "fp4_ldsm_single_omma_noshift",
    "fp4_ldsm_cute_rowstart",
    "fp4_ldsm_cute_rowstart_noshift",
    "fp4_ldsm_cute_rowstart_aonly",
    "fp4_ldsm_cute_rowstart_bonly",
    "fp4_ldsm_cute_rowstart_caccum",
    "fp4_ldsm_cute_rowstart_single_omma",
    "fp4_ldsm_cute_rowstart_single_omma_noshift",
    "fp4_ldsm_cute_rowstart_single_omma_aonly",
    "fp4_ldsm_cute_rowstart_single_omma_bonly",
    "fp4_ldsm_single_omma_real_scratch",
    "fp4_ldsm_single_omma_scratch_ones",
    "fp4_ldsm_single_omma_scratch_shifted_ones",
}


def _device_compile_flags() -> list[str]:
    flags = []
    if os.environ.get("TL_SM120_COMPACT_UNPACKED_FP4_SHARED") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_COMPACT_UNPACKED_FP4_SHARED=1")
    if os.environ.get("TL_SM120_MMA_RAW_UNPACKED_FP4_ACCESS_PTR") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_RAW_UNPACKED_FP4_ACCESS_PTR=1")
    if os.environ.get("TL_SM120_FULLTILE_AFULL_B_PANEL_STREAM") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_AFULL_B_PANEL_STREAM=1")
    if os.environ.get("TL_SM120_FULLTILE_AFULL_B_PANEL_ADDRS") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_AFULL_B_PANEL_ADDRS=1")
    if os.environ.get("TL_SM120_FULLTILE_AFULL_SCALE_REGS") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_AFULL_SCALE_REGS=1")
    if _EARLY_BENCH_OPTIONS.manual_ws2_sf_layout == "cutlass_128x4":
        flags.append("-DTL_SM120_FULLTILE_CUTLASS_SF_BASEPTR=1")
    scale_vector_package = os.environ.get("TL_SM120_FULLTILE_SCALE_VECTOR_PACKAGE") not in (None, "", "0")
    scale_slot_package = os.environ.get("TL_SM120_FULLTILE_SCALE_SLOT_PACKAGE") not in (None, "", "0")
    auto_scale_slot_package = (
        _EARLY_BENCH_OPTIONS.micro_pipeline == "sm120_backend_kblock_fulltile_package_pingpong"
        and _EARLY_BENCH_OPTIONS.manual_ws2_sf_layout == "cutlass_128x4"
        and not scale_vector_package
    )
    if scale_vector_package:
        flags.append("-DTL_SM120_FULLTILE_SCALE_VECTOR_PACKAGE=1")
    if scale_slot_package or auto_scale_slot_package:
        flags.append("-DTL_SM120_FULLTILE_SCALE_SLOT_PACKAGE=1")
    if os.environ.get("TL_SM120_FULLTILE_PACKAGE_ROWMAJOR_VIEW") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_PACKAGE_ROWMAJOR_VIEW=1")
    if os.environ.get("TL_SM120_FULLTILE_PAIR_ASM_B") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_PAIR_ASM_B=1")
    if os.environ.get("TL_SM120_FULLTILE_BOUNDED_2X2_PACKAGE") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_BOUNDED_2X2_PACKAGE=1")
    if os.environ.get("TL_SM120_FULLTILE_A2_BPAIR_ORDER") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_A2_BPAIR_ORDER=1")
    if os.environ.get("TL_SM120_FULLTILE_ROLLING_A_ORDER") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_ROLLING_A_ORDER=1")
    if os.environ.get("TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT") not in (None, "", "0"):
        flags.append("-DTL_SM120_FULLTILE_CUTE_ACCUM_DIRECT=1")
    if _EARLY_BENCH_OPTIONS.maxrregcount is not None:
        if _EARLY_BENCH_OPTIONS.maxrregcount <= 0:
            raise ValueError("--maxrregcount must be positive")
        flags.append(f"--maxrregcount={_EARLY_BENCH_OPTIONS.maxrregcount}")
    if _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view in _P64_LDSM_COPY_VIEWS and os.environ.get("TL_SM120_ALLOW_P64_LDSM_FOR_NVFP4") in (
        None,
        "",
        "0",
    ):
        raise ValueError(
            "--manual-ws2-ab-copy-view="
            f"{_EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view!r} belongs to "
            "the legacy fp4_ldsm diagnostic family, which uses or assumes "
            "ldmatrix.b8x16.b4x16_p64. That is the padded F8F6F4/MXF-style "
            "consumer path, not dense packed NVFP4. Use 'legacy_b16' for the "
            "NVFP4 path, or set TL_SM120_ALLOW_P64_LDSM_FOR_NVFP4=1 only to "
            "reproduce old diagnostic experiments."
        )
    if _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_single_omma":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "legacy_b16_single_omma":
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_noshift":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM_NO_SHIFT=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_single_omma_noshift":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM_NO_SHIFT=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_noshift":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM_NO_SHIFT=1")
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_aonly":
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART_A=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_bonly":
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART_B=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_caccum":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART=1")
        flags.append("-DTL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT=1")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_single_omma":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_single_omma_noshift":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM_NO_SHIFT=1")
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_single_omma_aonly":
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART_A=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_cute_rowstart_single_omma_bonly":
        flags.append("-DTL_SM120_FULLTILE_CUTE_ROWSTART_B=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_single_omma_real_scratch":
        flags.append("-DTL_SM120_FULLTILE_FP4_LDSM=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_REAL_SCRATCH=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_single_omma_scratch_ones":
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_SCRATCH_ONES=1")
        if _EARLY_BENCH_OPTIONS.manual_ws2_scratch_byte is not None:
            flags.append(f"-DTL_SM120_FULLTILE_SINGLE_SCRATCH_BYTE={_EARLY_BENCH_OPTIONS.manual_ws2_scratch_byte:#x}")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    elif _EARLY_BENCH_OPTIONS.manual_ws2_ab_copy_view == "fp4_ldsm_single_omma_scratch_shifted_ones":
        flags.append("-DTL_SM120_FULLTILE_SINGLE_OMMA=1")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_SCRATCH_ONES=1")
        scratch_byte = 0x88 if _EARLY_BENCH_OPTIONS.manual_ws2_scratch_byte is None else _EARLY_BENCH_OPTIONS.manual_ws2_scratch_byte
        flags.append(f"-DTL_SM120_FULLTILE_SINGLE_SCRATCH_BYTE={scratch_byte:#x}")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_KI=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_I=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_J=0")
        flags.append("-DTL_SM120_FULLTILE_SINGLE_HALF=0")
    if _EARLY_BENCH_OPTIONS.manual_ws2_reg_debug:
        flags.append("-DTL_SM120_FULLTILE_REG_DEBUG=1")
        if _EARLY_BENCH_OPTIONS.manual_ws2_reg_debug_mode == "tag":
            flags.append("-DTL_SM120_FULLTILE_REG_TAG_DEBUG=1")
    return flags


def _tilelang_jit_pass_configs() -> dict:
    pass_configs = {}
    if not _EARLY_BENCH_OPTIONS.enable_tma:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = True
    if not _EARLY_BENCH_OPTIONS.enable_warp_specialized:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
    device_compile_flags = _device_compile_flags()
    if device_compile_flags:
        pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = device_compile_flags
    if _EARLY_BENCH_OPTIONS.ptxas_verbose:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT] = True
    return pass_configs


def _manual_ws2_jit_pass_configs() -> dict:
    pass_configs = {}
    device_compile_flags = _device_compile_flags()
    if device_compile_flags:
        pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = device_compile_flags
    if _EARLY_BENCH_OPTIONS.ptxas_verbose:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT] = True
    return pass_configs


@tilelang.jit(
    out_idx=None,
    pass_configs=_tilelang_jit_pass_configs(),
)
def tilelang_nvfp4_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    warp_policy,
    out_dtype,
    micro_pipeline,
    load_mode,
    producer_regs: int,
    consumer_regs: int,
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
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, sf_words_per_block_k), T.uint32)
            SFB_shared = T.alloc_shared((block_N, sf_words_per_block_k), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if producer_regs > 0:
                T.annotate_producer_reg_dealloc(producer_regs)
            if consumer_regs > 0:
                T.annotate_consumer_reg_alloc(consumer_regs)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                if load_mode in ("tcopy", "tcopy_ab"):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                    if load_mode == "tcopy":
                        T.copy(SFA[by * block_M, ko * sf_words_per_block_k], SFA_shared)
                        T.copy(SFB[bx * block_N, ko * sf_words_per_block_k], SFB_shared)
                    else:
                        for i, k in T.Parallel(block_M, sf_words_per_block_k):
                            SFA_shared[i, k] = SFA[by * block_M + i, ko * sf_words_per_block_k + k]
                        for j, k in T.Parallel(block_N, sf_words_per_block_k):
                            SFB_shared[j, k] = SFB[bx * block_N + j, ko * sf_words_per_block_k + k]
                else:
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
                    policy=warp_policy,
                    clear_accum=False,
                    # Scale buffers are staged per K tile, so scale indexing is local.
                    k_start=0,
                    sf_a_granularity_k=sf_granularity_k,
                    sf_b_granularity_k=sf_granularity_k,
                    micro_pipeline=micro_pipeline,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(
    out_idx=None,
    pass_configs=_manual_ws2_jit_pass_configs(),
)
def tilelang_nvfp4_gemm_manual_ws2(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    warp_policy,
    out_dtype,
    micro_pipeline,
    producer_regs: int,
    consumer_regs: int,
    sf_load_mode: str,
    manual_ws2_split: str,
    sf_layout: str,
    ab_shared_storage: str,
):
    """Manual 1-producer/2-consumer SM120 NVFP4 path for pipeline study."""

    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0
    assert block_M % 2 == 0
    assert block_K % 64 == 0
    assert num_stages >= 2
    if sf_layout == "cutlass_128x4" and manual_ws2_split not in (
        "full",
        "full128",
        "pp",
        "pp_one",
        "pp_wait",
        "pp_stream",
        "pp_stream_output_tma_panel32",
        "pp_stream_output_tma_panel32_pipe2",
        "pp_stream_output_tma_epi64x32",
        "pp_stream_tma_store_literal",
        "pp_stream_tma_store_literal_copy",
        "pp_stream_panel32_tma_store",
        "pp_stream_panel64_tma_store",
    ):
        raise ValueError(
            "manual_ws2_sf_layout='cutlass_128x4' requires a full 128x4 scale tile; "
            "split m/n paths pass half-tile scale slices and need an explicit logical offset."
        )
    if sf_load_mode == "reorder" and sf_layout != "cutlass_128x4":
        raise ValueError("manual_ws2_sf_load='reorder' is only defined for cutlass_128x4 scale layout")

    in_dtype = T.float4_e2m1fn
    ab_shared_dtype = T.float4_e2m1_unpacked if ab_shared_storage == "unpacked" else in_dtype
    ab_swizzle_layout = os.environ.get("TL_SM120_WS2_AB_SWIZZLE_LAYOUT", "").strip()
    if not ab_swizzle_layout and os.environ.get("TL_SM120_WS2_AB_TCGEN05_SWIZZLE") not in (None, "", "0"):
        ab_swizzle_layout = "tcgen05"
    if ab_swizzle_layout not in ("", "tcgen05", "wgmma", "full_bank", "generic"):
        raise ValueError("TL_SM120_WS2_AB_SWIZZLE_LAYOUT must be one of: tcgen05, wgmma, full_bank, generic")
    accum_dtype = T.float32
    half_block_M = block_M // 2
    half_block_N = block_N // 2
    sf_granularity_k = 16
    sf_words_per_block_k = block_K // 64

    if manual_ws2_split in (
        "pp_stream",
        "pp_stream_output_tma_panel32",
        "pp_stream_output_tma_panel32_pipe2",
        "pp_stream_output_tma_epi64x32",
        "pp_stream_tma_store_literal",
        "pp_stream_tma_store_literal_copy",
        "pp_stream_panel32_tma_store",
        "pp_stream_panel64_tma_store",
    ):
        if sf_load_mode != "tma":
            raise ValueError("manual_ws2_split='pp_stream*' currently supports only --manual-ws2-sf-load tma")
        if sf_words_per_block_k < 4:
            raise ValueError("manual_ws2_split='pp_stream*' requires block_K >= 256 for scale TMA")
        output_tma_panel32 = manual_ws2_split in ("pp_stream_output_tma_panel32", "pp_stream_tma_store_literal")
        output_tma_panel32_pipe2 = manual_ws2_split == "pp_stream_output_tma_panel32_pipe2"
        output_tma_epi64x32 = manual_ws2_split == "pp_stream_output_tma_epi64x32"
        tma_store_literal = output_tma_panel32
        tma_store_literal_copy = manual_ws2_split == "pp_stream_tma_store_literal_copy"
        panel32_tma_store = manual_ws2_split == "pp_stream_panel32_tma_store"
        panel64_tma_store = manual_ws2_split == "pp_stream_panel64_tma_store"
        store_block_N = 32
        literal_panel_store = output_tma_panel32_pipe2 or output_tma_epi64x32 or tma_store_literal or tma_store_literal_copy
        epilogue_store_block_M = 64 if output_tma_epi64x32 else block_M
        epilogue_store_block_N = half_block_N if (tma_store_literal_copy or panel64_tma_store) else store_block_N
        epilogue_store_slots = (
            4 if output_tma_epi64x32 else (2 if output_tma_panel32_pipe2 else (1 if (literal_panel_store or panel64_tma_store) else 2))
        )
        if (literal_panel_store or panel32_tma_store or panel64_tma_store) and block_N != 128:
            raise ValueError("manual_ws2_split='pp_stream*_tma_store*' currently requires block_N=128")

        sm_num = driver.get_num_sms()
        n_blocks = T.ceildiv(N, block_N)
        m_blocks = T.ceildiv(M, block_M)
        total_tiles = n_blocks * m_blocks
        k_tiles = K // block_K
        stream_iters = T.ceildiv(total_tiles, sm_num)
        owner_iters = T.ceildiv(stream_iters, 2)

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K), in_dtype),
            SFA: T.Tensor((M, K // 64), T.uint32),
            SFB: T.Tensor((N, K // 64), T.uint32),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(sm_num, threads=384) as block_id:
                A_shared = T.alloc_shared((num_stages, block_M, block_K), ab_shared_dtype)
                B_shared = T.alloc_shared((num_stages, block_N, block_K), ab_shared_dtype)
                SFA_shared = T.alloc_shared((num_stages, block_M, sf_words_per_block_k), T.uint32)
                SFB_shared = T.alloc_shared((num_stages, block_N, sf_words_per_block_k), T.uint32)
                if ab_swizzle_layout == "tcgen05":
                    T.annotate_layout(
                        {
                            A_shared: tilelang.layout.make_tcgen05mma_swizzled_layout(A_shared),
                            B_shared: tilelang.layout.make_tcgen05mma_swizzled_layout(B_shared),
                        }
                    )
                elif ab_swizzle_layout == "wgmma":
                    T.annotate_layout(
                        {
                            A_shared: tilelang.layout.make_wgmma_swizzled_layout(A_shared),
                            B_shared: tilelang.layout.make_wgmma_swizzled_layout(B_shared),
                        }
                    )
                elif ab_swizzle_layout == "full_bank":
                    T.annotate_layout(
                        {
                            A_shared: tilelang.layout.make_full_bank_swizzled_layout(A_shared),
                            B_shared: tilelang.layout.make_full_bank_swizzled_layout(B_shared),
                        }
                    )
                elif ab_swizzle_layout == "generic":
                    T.annotate_layout(
                        {
                            A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                            B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                        }
                    )
                if literal_panel_store or panel32_tma_store or panel64_tma_store:
                    C_shared = T.alloc_shared((epilogue_store_slots, epilogue_store_block_M, epilogue_store_block_N), out_dtype)

                loaded = T.alloc_barrier([128] * (2 * num_stages))
                consumed = T.alloc_barrier([128] * (2 * num_stages))
                wg_order = T.alloc_barrier([128] * 2)
                store_order = T.alloc_barrier([128] * 2)

                tx = T.get_thread_binding()

                if tx >= 256:
                    if producer_regs > 0:
                        T.set_max_nreg(producer_regs, 0)
                    for stream in T.serial(stream_iters):
                        tile_id = block_id + stream * sm_num
                        if tile_id < total_tiles:
                            tile_n = tile_id % n_blocks
                            tile_m = tile_id // n_blocks
                            owner = stream & 1
                            owner_iter = stream // 2
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                stage = ko % num_stages
                                phase = ((owner_iter * k_tiles + ko) // num_stages) & 1
                                barrier_slot = owner * num_stages + stage
                                if stream > 0 and ko < num_stages:
                                    prev_owner = (stream - 1) & 1
                                    prev_owner_iter = (stream - 1) // 2
                                    prev_barrier_slot = prev_owner * num_stages + stage
                                    prev_last_ko = ((k_tiles - 1 - stage) // num_stages) * num_stages + stage
                                    prev_phase = ((prev_owner_iter * k_tiles + prev_last_ko) // num_stages) & 1
                                    T.barrier_wait(consumed[prev_barrier_slot], prev_phase)
                                else:
                                    T.barrier_wait(consumed[barrier_slot], phase ^ 1)
                                T.tma_copy(
                                    A[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    A_shared[stage, :, :],
                                    barrier=loaded[barrier_slot],
                                    leader_scope_threads=128,
                                )
                                T.tma_copy(
                                    B[
                                        tile_n * block_N : (tile_n + 1) * block_N,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    B_shared[stage, :, :],
                                    barrier=loaded[barrier_slot],
                                    leader_scope_threads=128,
                                )
                                T.tma_copy(
                                    SFA[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                    ],
                                    SFA_shared[stage, :, :],
                                    barrier=loaded[barrier_slot],
                                    leader_scope_threads=128,
                                )
                                T.tma_copy(
                                    SFB[
                                        tile_n * block_N : (tile_n + 1) * block_N,
                                        ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                    ],
                                    SFB_shared[stage, :, :],
                                    barrier=loaded[barrier_slot],
                                    leader_scope_threads=128,
                                )
                                T.barrier_arrive(loaded[barrier_slot])

                elif tx < 128:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    if panel32_tma_store:
                        C0_local_0 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C0_local_1 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C0_local_2 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C0_local_3 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                    elif panel64_tma_store:
                        C0_local_0 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                        C0_local_1 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                    else:
                        C0_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for owner_iter in T.serial(owner_iters):
                        stream = owner_iter * 2
                        tile_id = block_id + stream * sm_num
                        if tile_id < total_tiles:
                            tile_n = tile_id % n_blocks
                            tile_m = tile_id // n_blocks

                            if owner_iter > 0:
                                T.barrier_wait(wg_order[0], (owner_iter - 1) & 1)

                            if panel32_tma_store:
                                T.clear(C0_local_0)
                                T.clear(C0_local_1)
                                T.clear(C0_local_2)
                                T.clear(C0_local_3)
                            elif panel64_tma_store:
                                T.clear(C0_local_0)
                                T.clear(C0_local_1)
                            else:
                                T.clear(C0_local)
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                stage = ko % num_stages
                                phase = ((owner_iter * k_tiles + ko) // num_stages) & 1
                                barrier_slot = stage
                                T.barrier_wait(loaded[barrier_slot], phase)
                                if panel32_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:store_block_N, :],
                                        C0_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, store_block_N : 2 * store_block_N, :],
                                        C0_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, store_block_N : 2 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        C0_local_2,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        C0_local_3,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                elif panel64_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:half_block_N, :],
                                        C0_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:half_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, half_block_N:block_N, :],
                                        C0_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, half_block_N:block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                else:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C0_local,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, :, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                T.barrier_arrive(consumed[barrier_slot])

                            T.barrier_arrive(wg_order[1])

                            if panel32_tma_store:
                                base_n0 = tile_n * block_N
                                T.copy(C0_local_0, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n1 = tile_n * block_N + store_block_N
                                T.copy(C0_local_1, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n1 : base_n1 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + 2 * store_block_N
                                T.copy(C0_local_2, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n3 = tile_n * block_N + 3 * store_block_N
                                T.copy(C0_local_3, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n3 : base_n3 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                            elif panel64_tma_store:
                                if owner_iter > 0:
                                    T.barrier_wait(store_order[0], (owner_iter - 1) & 1)
                                base_n0 = tile_n * block_N
                                T.copy(C0_local_0, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + half_block_N
                                T.copy(C0_local_1, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                                T.barrier_arrive(store_order[1])
                            elif literal_panel_store:
                                if owner_iter > 0:
                                    T.barrier_wait(store_order[0], (owner_iter - 1) & 1)
                                if output_tma_panel32_pipe2:
                                    for panel32 in T.serial(4):
                                        epi_slot = panel32 & 1
                                        if panel32 >= 2:
                                            T.tma_store_wait(1, True)
                                            T.sync_threads(7, 128)
                                        T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                            C0_local.data,
                                            0,
                                            C_shared.data,
                                            epi_slot * epilogue_store_block_M * epilogue_store_block_N,
                                            panel32,
                                        )
                                        T.fence_proxy_async()
                                        T.sync_threads(7, 128)
                                        T.tma_copy(
                                            C_shared[epi_slot, :, :],
                                            C[
                                                tile_m * block_M : (tile_m + 1) * block_M,
                                                tile_n * block_N + panel32 * epilogue_store_block_N : tile_n * block_N
                                                + (panel32 + 1) * epilogue_store_block_N,
                                            ],
                                            leader_scope_threads=128,
                                        )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(7, 128)
                                elif output_tma_epi64x32:
                                    for epi in T.serial(8):
                                        epi_slot = epi % 4
                                        epi_m = epi // 4
                                        epi_n = epi % 4
                                        if epi >= 4:
                                            T.tma_store_wait(3, True)
                                            T.sync_threads(7, 128)
                                        T.sm120_store_full_c_fragment_epi64x32_tma_bf16(
                                            C0_local.data,
                                            0,
                                            C_shared.data,
                                            epi_slot * epilogue_store_block_M * epilogue_store_block_N,
                                            epi_m,
                                            epi_n,
                                        )
                                        T.fence_proxy_async()
                                        T.sync_threads(7, 128)
                                        T.tma_copy(
                                            C_shared[epi_slot, :, :],
                                            C[
                                                tile_m * block_M + epi_m * epilogue_store_block_M : tile_m * block_M
                                                + (epi_m + 1) * epilogue_store_block_M,
                                                tile_n * block_N + epi_n * epilogue_store_block_N : tile_n * block_N
                                                + (epi_n + 1) * epilogue_store_block_N,
                                            ],
                                            leader_scope_threads=128,
                                        )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(7, 128)
                                elif tma_store_literal_copy:
                                    base_n0 = tile_n * block_N
                                    T.sm120_store_full_c_fragment_panel64_bf16(
                                        C0_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        0,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(7, 128)
                                    for linear in T.Parallel(block_M * epilogue_store_block_N):
                                        store_i = linear // epilogue_store_block_N
                                        store_j = linear % epilogue_store_block_N
                                        C[tile_m * block_M + store_i, base_n0 + store_j] = C_shared[0, store_i, store_j]

                                    base_n2 = tile_n * block_N + half_block_N
                                    T.sm120_store_full_c_fragment_panel64_bf16(
                                        C0_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        1,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(7, 128)
                                    for linear in T.Parallel(block_M * epilogue_store_block_N):
                                        store_i = linear // epilogue_store_block_N
                                        store_j = linear % epilogue_store_block_N
                                        C[tile_m * block_M + store_i, base_n2 + store_j] = C_shared[0, store_i, store_j]
                                else:
                                    base_n0 = tile_n * block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C0_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        0,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(7, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n0 : base_n0 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                    # All consumer threads must wait before the reused C_shared panel is overwritten.
                                    T.sync_threads(7, 128)

                                    base_n1 = tile_n * block_N + store_block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C0_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        1,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(7, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n1 : base_n1 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(7, 128)

                                    base_n2 = tile_n * block_N + half_block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C0_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        2,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(7, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n2 : base_n2 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(7, 128)

                                    base_n3 = tile_n * block_N + half_block_N + store_block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C0_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        3,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(7, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n3 : base_n3 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                T.barrier_arrive(store_order[1])
                            else:
                                T.copy(C0_local, C[tile_m * block_M, tile_n * block_N])

                else:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    if panel32_tma_store:
                        C1_local_0 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C1_local_1 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C1_local_2 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C1_local_3 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                    elif panel64_tma_store:
                        C1_local_0 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                        C1_local_1 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                    else:
                        C1_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for owner_iter in T.serial(owner_iters):
                        stream = owner_iter * 2 + 1
                        tile_id = block_id + stream * sm_num
                        if tile_id < total_tiles:
                            tile_n = tile_id % n_blocks
                            tile_m = tile_id // n_blocks

                            T.barrier_wait(wg_order[1], owner_iter & 1)

                            if panel32_tma_store:
                                T.clear(C1_local_0)
                                T.clear(C1_local_1)
                                T.clear(C1_local_2)
                                T.clear(C1_local_3)
                            elif panel64_tma_store:
                                T.clear(C1_local_0)
                                T.clear(C1_local_1)
                            else:
                                T.clear(C1_local)
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                stage = ko % num_stages
                                phase = ((owner_iter * k_tiles + ko) // num_stages) & 1
                                barrier_slot = num_stages + stage
                                T.barrier_wait(loaded[barrier_slot], phase)
                                if panel32_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:store_block_N, :],
                                        C1_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, store_block_N : 2 * store_block_N, :],
                                        C1_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, store_block_N : 2 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        C1_local_2,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        C1_local_3,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                elif panel64_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:half_block_N, :],
                                        C1_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:half_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, half_block_N:block_N, :],
                                        C1_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, half_block_N:block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                else:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C1_local,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, :, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                T.barrier_arrive(consumed[barrier_slot])

                            T.barrier_arrive(wg_order[0])

                            if panel32_tma_store:
                                base_n0 = tile_n * block_N
                                T.copy(C1_local_0, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n1 = tile_n * block_N + store_block_N
                                T.copy(C1_local_1, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n1 : base_n1 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + 2 * store_block_N
                                T.copy(C1_local_2, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n3 = tile_n * block_N + 3 * store_block_N
                                T.copy(C1_local_3, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n3 : base_n3 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                            elif panel64_tma_store:
                                T.barrier_wait(store_order[1], owner_iter & 1)
                                base_n0 = tile_n * block_N
                                T.copy(C1_local_0, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + half_block_N
                                T.copy(C1_local_1, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                                T.barrier_arrive(store_order[0])
                            elif literal_panel_store:
                                T.barrier_wait(store_order[1], owner_iter & 1)
                                if output_tma_panel32_pipe2:
                                    for panel32 in T.serial(4):
                                        epi_slot = panel32 & 1
                                        if panel32 >= 2:
                                            T.tma_store_wait(1, True)
                                            T.sync_threads(8, 128)
                                        T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                            C1_local.data,
                                            0,
                                            C_shared.data,
                                            epi_slot * epilogue_store_block_M * epilogue_store_block_N,
                                            panel32,
                                        )
                                        T.fence_proxy_async()
                                        T.sync_threads(8, 128)
                                        T.tma_copy(
                                            C_shared[epi_slot, :, :],
                                            C[
                                                tile_m * block_M : (tile_m + 1) * block_M,
                                                tile_n * block_N + panel32 * epilogue_store_block_N : tile_n * block_N
                                                + (panel32 + 1) * epilogue_store_block_N,
                                            ],
                                            leader_scope_threads=128,
                                        )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(8, 128)
                                elif output_tma_epi64x32:
                                    for epi in T.serial(8):
                                        epi_slot = epi % 4
                                        epi_m = epi // 4
                                        epi_n = epi % 4
                                        if epi >= 4:
                                            T.tma_store_wait(3, True)
                                            T.sync_threads(8, 128)
                                        T.sm120_store_full_c_fragment_epi64x32_tma_bf16(
                                            C1_local.data,
                                            0,
                                            C_shared.data,
                                            epi_slot * epilogue_store_block_M * epilogue_store_block_N,
                                            epi_m,
                                            epi_n,
                                        )
                                        T.fence_proxy_async()
                                        T.sync_threads(8, 128)
                                        T.tma_copy(
                                            C_shared[epi_slot, :, :],
                                            C[
                                                tile_m * block_M + epi_m * epilogue_store_block_M : tile_m * block_M
                                                + (epi_m + 1) * epilogue_store_block_M,
                                                tile_n * block_N + epi_n * epilogue_store_block_N : tile_n * block_N
                                                + (epi_n + 1) * epilogue_store_block_N,
                                            ],
                                            leader_scope_threads=128,
                                        )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(8, 128)
                                elif tma_store_literal_copy:
                                    base_n0 = tile_n * block_N
                                    T.sm120_store_full_c_fragment_panel64_bf16(
                                        C1_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        0,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(8, 128)
                                    for linear in T.Parallel(block_M * epilogue_store_block_N):
                                        store_i = linear // epilogue_store_block_N
                                        store_j = linear % epilogue_store_block_N
                                        C[tile_m * block_M + store_i, base_n0 + store_j] = C_shared[0, store_i, store_j]

                                    base_n2 = tile_n * block_N + half_block_N
                                    T.sm120_store_full_c_fragment_panel64_bf16(
                                        C1_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        1,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(8, 128)
                                    for linear in T.Parallel(block_M * epilogue_store_block_N):
                                        store_i = linear // epilogue_store_block_N
                                        store_j = linear % epilogue_store_block_N
                                        C[tile_m * block_M + store_i, base_n2 + store_j] = C_shared[0, store_i, store_j]
                                else:
                                    base_n0 = tile_n * block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C1_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        0,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(8, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n0 : base_n0 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                    # All consumer threads must wait before the reused C_shared panel is overwritten.
                                    T.sync_threads(8, 128)

                                    base_n1 = tile_n * block_N + store_block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C1_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        1,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(8, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n1 : base_n1 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(8, 128)

                                    base_n2 = tile_n * block_N + half_block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C1_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        2,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(8, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n2 : base_n2 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                    T.sync_threads(8, 128)

                                    base_n3 = tile_n * block_N + half_block_N + store_block_N
                                    T.sm120_store_full_c_fragment_panel32_tma_bf16(
                                        C1_local.data,
                                        0,
                                        C_shared.data,
                                        0,
                                        3,
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(8, 128)
                                    T.tma_copy(
                                        C_shared[0, :, :],
                                        C[
                                            tile_m * block_M : (tile_m + 1) * block_M,
                                            base_n3 : base_n3 + epilogue_store_block_N,
                                        ],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_store_wait(0, True)
                                T.barrier_arrive(store_order[0])
                            else:
                                T.copy(C1_local, C[tile_m * block_M, tile_n * block_N])

        return main

    if manual_ws2_split in ("pp", "pp_one", "pp_wait"):
        if sf_load_mode not in ("tma", "parallel", "serial", "none", "direct"):
            raise ValueError("manual_ws2_split='pp*' currently supports --manual-ws2-sf-load tma, parallel, serial, none, or direct")
        if sf_words_per_block_k < 4:
            raise ValueError("manual_ws2_split='pp' requires block_K >= 256 for scale TMA")
        if manual_ws2_split in ("pp", "pp_one", "pp_wait") and (N // block_N) % 2 != 0:
            raise ValueError("manual_ws2_split='pp*' currently requires an even number of N tiles")

        sm_num = driver.get_num_sms()
        n_blocks = T.ceildiv(N, block_N)
        m_blocks = T.ceildiv(M, block_M)
        pair_cols = n_blocks // 2
        total_tiles = n_blocks * m_blocks
        total_pairs = pair_cols * m_blocks
        k_tiles = K // block_K
        waves = T.ceildiv(total_pairs, sm_num)
        grid_blocks = 1 if manual_ws2_split == "pp_one" else sm_num

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K), in_dtype),
            SFA: T.Tensor((M, K // 64), T.uint32),
            SFB: T.Tensor((N, K // 64), T.uint32),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(grid_blocks, threads=384) as block_id:
                A_shared = T.alloc_shared((num_stages, block_M, block_K), ab_shared_dtype)
                B_shared = T.alloc_shared((num_stages, block_N, block_K), ab_shared_dtype)
                SFA_shared = T.alloc_shared((num_stages, block_M, sf_words_per_block_k), T.uint32)
                SFB_shared = T.alloc_shared((num_stages, block_N, sf_words_per_block_k), T.uint32)

                loaded0 = T.alloc_barrier([128] * num_stages)
                loaded1 = T.alloc_barrier([128] * num_stages)
                consumed0 = T.alloc_barrier([128] * num_stages)
                consumed1 = T.alloc_barrier([128] * num_stages)

                tx = T.get_thread_binding()

                if tx >= 256:
                    if producer_regs > 0:
                        T.set_max_nreg(producer_regs, 0)
                    for wave in T.serial(waves):
                        pair_id = block_id + wave * sm_num
                        pair_n = pair_id % pair_cols
                        pair_m = pair_id // pair_cols
                        tile0 = pair_m * n_blocks + pair_n * 2
                        tile1 = tile0 + 1
                        tile0_n = pair_n * 2
                        tile0_m = pair_m
                        tile1_n = tile0_n + 1
                        tile1_m = pair_m
                        for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                            iter_k = wave * k_tiles + ko
                            stage = ko % num_stages
                            phase = (iter_k // num_stages) & 1
                            if tile0 < total_tiles:
                                if ko < num_stages:
                                    T.barrier_wait(consumed1[stage], phase ^ 1)
                                else:
                                    T.barrier_wait(consumed0[stage], phase ^ 1)
                                T.tma_copy(
                                    A[
                                        tile0_m * block_M : (tile0_m + 1) * block_M,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    A_shared[stage, :, :],
                                    barrier=loaded0[stage],
                                    leader_scope_threads=128,
                                )
                                T.tma_copy(
                                    B[
                                        tile0_n * block_N : (tile0_n + 1) * block_N,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    B_shared[stage, :, :],
                                    barrier=loaded0[stage],
                                    leader_scope_threads=128,
                                )
                                if sf_load_mode == "tma":
                                    T.tma_copy(
                                        SFA[
                                            tile0_m * block_M : (tile0_m + 1) * block_M,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        SFA_shared[stage, :, :],
                                        barrier=loaded0[stage],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_copy(
                                        SFB[
                                            tile0_n * block_N : (tile0_n + 1) * block_N,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        SFB_shared[stage, :, :],
                                        barrier=loaded0[stage],
                                        leader_scope_threads=128,
                                    )
                                elif sf_load_mode == "none":
                                    pass
                                elif sf_load_mode != "direct":
                                    if sf_load_mode == "serial":
                                        for i in T.serial(block_M):
                                            for k in T.serial(sf_words_per_block_k):
                                                SFA_shared[stage, i, k] = SFA[tile0_m * block_M + i, ko * sf_words_per_block_k + k]
                                        for j in T.serial(block_N):
                                            for k in T.serial(sf_words_per_block_k):
                                                SFB_shared[stage, j, k] = SFB[tile0_n * block_N + j, ko * sf_words_per_block_k + k]
                                    else:
                                        for i, k in T.Parallel(block_M, sf_words_per_block_k):
                                            SFA_shared[stage, i, k] = SFA[tile0_m * block_M + i, ko * sf_words_per_block_k + k]
                                        for j, k in T.Parallel(block_N, sf_words_per_block_k):
                                            SFB_shared[stage, j, k] = SFB[tile0_n * block_N + j, ko * sf_words_per_block_k + k]
                                T.barrier_arrive(loaded0[stage])
                        for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                            iter_k = wave * k_tiles + ko
                            stage = ko % num_stages
                            phase = (iter_k // num_stages) & 1
                            if tile1 < total_tiles:
                                if ko < num_stages:
                                    T.barrier_wait(consumed0[stage], phase)
                                else:
                                    T.barrier_wait(consumed1[stage], phase ^ 1)
                                T.tma_copy(
                                    A[
                                        tile1_m * block_M : (tile1_m + 1) * block_M,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    A_shared[stage, :, :],
                                    barrier=loaded1[stage],
                                    leader_scope_threads=128,
                                )
                                T.tma_copy(
                                    B[
                                        tile1_n * block_N : (tile1_n + 1) * block_N,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    B_shared[stage, :, :],
                                    barrier=loaded1[stage],
                                    leader_scope_threads=128,
                                )
                                if sf_load_mode == "tma":
                                    T.tma_copy(
                                        SFA[
                                            tile1_m * block_M : (tile1_m + 1) * block_M,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        SFA_shared[stage, :, :],
                                        barrier=loaded1[stage],
                                        leader_scope_threads=128,
                                    )
                                    T.tma_copy(
                                        SFB[
                                            tile1_n * block_N : (tile1_n + 1) * block_N,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        SFB_shared[stage, :, :],
                                        barrier=loaded1[stage],
                                        leader_scope_threads=128,
                                    )
                                elif sf_load_mode == "none":
                                    pass
                                elif sf_load_mode != "direct":
                                    if sf_load_mode == "serial":
                                        for i in T.serial(block_M):
                                            for k in T.serial(sf_words_per_block_k):
                                                SFA_shared[stage, i, k] = SFA[tile1_m * block_M + i, ko * sf_words_per_block_k + k]
                                        for j in T.serial(block_N):
                                            for k in T.serial(sf_words_per_block_k):
                                                SFB_shared[stage, j, k] = SFB[tile1_n * block_N + j, ko * sf_words_per_block_k + k]
                                    else:
                                        for i, k in T.Parallel(block_M, sf_words_per_block_k):
                                            SFA_shared[stage, i, k] = SFA[tile1_m * block_M + i, ko * sf_words_per_block_k + k]
                                        for j, k in T.Parallel(block_N, sf_words_per_block_k):
                                            SFB_shared[stage, j, k] = SFB[tile1_n * block_N + j, ko * sf_words_per_block_k + k]
                                T.barrier_arrive(loaded1[stage])

                elif tx < 128:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    C0_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for wave in T.serial(waves):
                        pair_id = block_id + wave * sm_num
                        pair_n = pair_id % pair_cols
                        pair_m = pair_id // pair_cols
                        tile0 = pair_m * n_blocks + pair_n * 2
                        tile0_n = pair_n * 2
                        tile0_m = pair_m
                        if tile0 < total_tiles:
                            if manual_ws2_split != "pp_wait":
                                T.clear(C0_local)
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                iter_k = wave * k_tiles + ko
                                stage = ko % num_stages
                                phase = (iter_k // num_stages) & 1
                                T.barrier_wait(loaded0[stage], phase)
                                if manual_ws2_split == "pp_wait":
                                    pass
                                elif sf_load_mode == "direct":
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C0_local,
                                        SFA[
                                            tile0_m * block_M : (tile0_m + 1) * block_M,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        SFB[
                                            tile0_n * block_N : (tile0_n + 1) * block_N,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                else:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C0_local,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, :, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                T.barrier_arrive(consumed0[stage])
                            if manual_ws2_split != "pp_wait":
                                T.copy(C0_local, C[tile0_m * block_M, tile0_n * block_N])

                else:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    C1_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for wave in T.serial(waves):
                        pair_id = block_id + wave * sm_num
                        pair_n = pair_id % pair_cols
                        pair_m = pair_id // pair_cols
                        tile1 = pair_m * n_blocks + pair_n * 2 + 1
                        tile1_n = pair_n * 2 + 1
                        tile1_m = pair_m
                        if tile1 < total_tiles:
                            if manual_ws2_split != "pp_wait":
                                T.clear(C1_local)
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                iter_k = wave * k_tiles + ko
                                stage = ko % num_stages
                                phase = (iter_k // num_stages) & 1
                                T.barrier_wait(loaded1[stage], phase)
                                if manual_ws2_split == "pp_wait":
                                    pass
                                elif sf_load_mode == "direct":
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C1_local,
                                        SFA[
                                            tile1_m * block_M : (tile1_m + 1) * block_M,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        SFB[
                                            tile1_n * block_N : (tile1_n + 1) * block_N,
                                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                        ],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                else:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C1_local,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, :, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        micro_pipeline=micro_pipeline,
                                        sf_layout=sf_layout,
                                    )
                                T.barrier_arrive(consumed1[stage])
                            if manual_ws2_split != "pp_wait":
                                T.copy(C1_local, C[tile1_m * block_M, tile1_n * block_N])

        return main

    if manual_ws2_split in ("full", "full128"):
        full_consumer_threads = 128 if manual_ws2_split == "full128" else 256

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K), in_dtype),
            SFA: T.Tensor((M, K // 64), T.uint32),
            SFB: T.Tensor((N, K // 64), T.uint32),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=384) as (bx, by):
                A_shared = T.alloc_shared((num_stages, block_M, block_K), ab_shared_dtype)
                B_shared = T.alloc_shared((num_stages, block_N, block_K), ab_shared_dtype)
                SFA_shared = T.alloc_shared((num_stages, block_M, sf_words_per_block_k), T.uint32)
                SFB_shared = T.alloc_shared((num_stages, block_N, sf_words_per_block_k), T.uint32)

                loaded = T.alloc_barrier([128] * num_stages)
                consumed = T.alloc_barrier([full_consumer_threads] * num_stages)

                tx = T.get_thread_binding()
                T.use_swizzle(panel_size=10)

                if tx >= 256:
                    if producer_regs > 0:
                        T.set_max_nreg(producer_regs, 0)
                    for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                        stage = ko % num_stages
                        phase = (ko // num_stages) & 1
                        T.barrier_wait(consumed[stage], phase ^ 1)
                        T.tma_copy(
                            A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K],
                            A_shared[stage, :, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                        T.tma_copy(
                            B[bx * block_N : (bx + 1) * block_N, ko * block_K : (ko + 1) * block_K],
                            B_shared[stage, :, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                        if sf_load_mode == "reorder":
                            for linear in T.Parallel(block_M * sf_words_per_block_k):
                                i = linear // sf_words_per_block_k
                                k = linear % sf_words_per_block_k
                                dst_i = k * 32 + (i % 32)
                                dst_k = i // 32
                                SFA_shared[stage, dst_i, dst_k] = SFA[by * block_M + i, ko * sf_words_per_block_k + k]
                            for linear in T.Parallel(block_N * sf_words_per_block_k):
                                j = linear // sf_words_per_block_k
                                k = linear % sf_words_per_block_k
                                dst_j = k * 32 + (j % 32)
                                dst_k = j // 32
                                SFB_shared[stage, dst_j, dst_k] = SFB[bx * block_N + j, ko * sf_words_per_block_k + k]
                        elif sf_load_mode == "tma" and sf_words_per_block_k >= 4:
                            T.tma_copy(
                                SFA[
                                    by * block_M : (by + 1) * block_M,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFA_shared[stage, :, :],
                                barrier=loaded[stage],
                                leader_scope_threads=128,
                            )
                            T.tma_copy(
                                SFB[
                                    bx * block_N : (bx + 1) * block_N,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFB_shared[stage, :, :],
                                barrier=loaded[stage],
                                leader_scope_threads=128,
                            )
                        elif sf_load_mode != "direct":
                            for i, k in T.Parallel(block_M, sf_words_per_block_k):
                                SFA_shared[stage, i, k] = SFA[by * block_M + i, ko * sf_words_per_block_k + k]
                            for j, k in T.Parallel(block_N, sf_words_per_block_k):
                                SFB_shared[stage, j, k] = SFB[bx * block_N + j, ko * sf_words_per_block_k + k]
                        T.barrier_arrive(loaded[stage])

                elif tx < full_consumer_threads:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    T.clear(C_local)
                    for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                        stage = ko % num_stages
                        phase = (ko // num_stages) & 1
                        T.barrier_wait(loaded[stage], phase)
                        if sf_load_mode == "direct":
                            T.mma_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, :, :],
                                C_local,
                                SFA[
                                    by * block_M : (by + 1) * block_M,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFB[
                                    bx * block_N : (bx + 1) * block_N,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                transpose_B=True,
                                policy=warp_policy,
                                clear_accum=False,
                                k_start=0,
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                micro_pipeline=micro_pipeline,
                                sf_layout=sf_layout,
                            )
                        else:
                            T.mma_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, :, :],
                                C_local,
                                SFA_shared[stage, :, :],
                                SFB_shared[stage, :, :],
                                transpose_B=True,
                                policy=warp_policy,
                                clear_accum=False,
                                k_start=0,
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                micro_pipeline=micro_pipeline,
                                sf_layout=sf_layout,
                            )
                        T.barrier_arrive(consumed[stage])
                    T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    if manual_ws2_split == "n":
        assert block_N % 2 == 0

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K), in_dtype),
            SFA: T.Tensor((M, K // 64), T.uint32),
            SFB: T.Tensor((N, K // 64), T.uint32),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=384) as (bx, by):
                A_shared = T.alloc_shared((num_stages, block_M, block_K), ab_shared_dtype)
                B_shared = T.alloc_shared((num_stages, block_N, block_K), ab_shared_dtype)
                SFA_shared = T.alloc_shared((num_stages, block_M, sf_words_per_block_k), T.uint32)
                SFB_shared = T.alloc_shared((num_stages, block_N, sf_words_per_block_k), T.uint32)

                loaded = T.alloc_barrier([128] * num_stages)
                consumed = T.alloc_barrier([256] * num_stages)

                tx = T.get_thread_binding()
                T.use_swizzle(panel_size=10)

                if tx >= 256:
                    if producer_regs > 0:
                        T.set_max_nreg(producer_regs, 0)
                    for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                        stage = ko % num_stages
                        phase = (ko // num_stages) & 1
                        T.barrier_wait(consumed[stage], phase ^ 1)
                        T.tma_copy(
                            A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K],
                            A_shared[stage, :, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                        T.tma_copy(
                            B[bx * block_N : (bx + 1) * block_N, ko * block_K : (ko + 1) * block_K],
                            B_shared[stage, :, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                        if sf_load_mode == "tma" and sf_words_per_block_k >= 4:
                            T.tma_copy(
                                SFA[
                                    by * block_M : (by + 1) * block_M,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFA_shared[stage, :, :],
                                barrier=loaded[stage],
                                leader_scope_threads=128,
                            )
                            T.tma_copy(
                                SFB[
                                    bx * block_N : (bx + 1) * block_N,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFB_shared[stage, :, :],
                                barrier=loaded[stage],
                                leader_scope_threads=128,
                            )
                        elif sf_load_mode != "direct":
                            for i, k in T.Parallel(block_M, sf_words_per_block_k):
                                SFA_shared[stage, i, k] = SFA[by * block_M + i, ko * sf_words_per_block_k + k]
                            for j, k in T.Parallel(block_N, sf_words_per_block_k):
                                SFB_shared[stage, j, k] = SFB[bx * block_N + j, ko * sf_words_per_block_k + k]
                        T.barrier_arrive(loaded[stage])

                elif tx < 128:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    C0_local = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                    T.clear(C0_local)
                    for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                        stage = ko % num_stages
                        phase = (ko // num_stages) & 1
                        T.barrier_wait(loaded[stage], phase)
                        if sf_load_mode == "direct":
                            T.mma_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, 0:half_block_N, :],
                                C0_local,
                                SFA[
                                    by * block_M : (by + 1) * block_M,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFB[
                                    bx * block_N : bx * block_N + half_block_N,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                transpose_B=True,
                                policy=warp_policy,
                                clear_accum=False,
                                k_start=0,
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                micro_pipeline=micro_pipeline,
                                sf_layout=sf_layout,
                            )
                        else:
                            T.mma_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, 0:half_block_N, :],
                                C0_local,
                                SFA_shared[stage, :, :],
                                SFB_shared[stage, 0:half_block_N, :],
                                transpose_B=True,
                                policy=warp_policy,
                                clear_accum=False,
                                k_start=0,
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                micro_pipeline=micro_pipeline,
                                sf_layout=sf_layout,
                            )
                        T.barrier_arrive(consumed[stage])
                    T.copy(C0_local, C[by * block_M, bx * block_N])

                else:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    C1_local = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                    T.clear(C1_local)
                    for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                        stage = ko % num_stages
                        phase = (ko // num_stages) & 1
                        T.barrier_wait(loaded[stage], phase)
                        if sf_load_mode == "direct":
                            T.mma_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, half_block_N:block_N, :],
                                C1_local,
                                SFA[
                                    by * block_M : (by + 1) * block_M,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                SFB[
                                    bx * block_N + half_block_N : (bx + 1) * block_N,
                                    ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                                ],
                                transpose_B=True,
                                policy=warp_policy,
                                clear_accum=False,
                                k_start=0,
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                micro_pipeline=micro_pipeline,
                                sf_layout=sf_layout,
                            )
                        else:
                            T.mma_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, half_block_N:block_N, :],
                                C1_local,
                                SFA_shared[stage, :, :],
                                SFB_shared[stage, half_block_N:block_N, :],
                                transpose_B=True,
                                policy=warp_policy,
                                clear_accum=False,
                                k_start=0,
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                micro_pipeline=micro_pipeline,
                                sf_layout=sf_layout,
                            )
                        T.barrier_arrive(consumed[stage])
                    T.copy(C1_local, C[by * block_M, bx * block_N + half_block_N])

        return main

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=384) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), ab_shared_dtype)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), ab_shared_dtype)
            SFA_shared = T.alloc_shared((num_stages, block_M, sf_words_per_block_k), T.uint32)
            SFB_shared = T.alloc_shared((num_stages, block_N, sf_words_per_block_k), T.uint32)

            loaded = T.alloc_barrier([128] * num_stages)
            consumed = T.alloc_barrier([256] * num_stages)

            tx = T.get_thread_binding()
            T.use_swizzle(panel_size=10)

            if tx >= 256:
                if producer_regs > 0:
                    T.set_max_nreg(producer_regs, 0)
                for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.barrier_wait(consumed[stage], phase ^ 1)
                    T.tma_copy(
                        A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K],
                        A_shared[stage, :, :],
                        barrier=loaded[stage],
                        leader_scope_threads=128,
                    )
                    T.tma_copy(
                        B[bx * block_N : (bx + 1) * block_N, ko * block_K : (ko + 1) * block_K],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                        leader_scope_threads=128,
                    )
                    if sf_load_mode == "tma" and sf_words_per_block_k >= 4:
                        T.tma_copy(
                            SFA[
                                by * block_M : (by + 1) * block_M,
                                ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                            ],
                            SFA_shared[stage, :, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                        T.tma_copy(
                            SFB[
                                bx * block_N : (bx + 1) * block_N,
                                ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                            ],
                            SFB_shared[stage, :, :],
                            barrier=loaded[stage],
                            leader_scope_threads=128,
                        )
                    elif sf_load_mode != "direct":
                        for i, k in T.Parallel(block_M, sf_words_per_block_k):
                            SFA_shared[stage, i, k] = SFA[by * block_M + i, ko * sf_words_per_block_k + k]
                        for j, k in T.Parallel(block_N, sf_words_per_block_k):
                            SFB_shared[stage, j, k] = SFB[bx * block_N + j, ko * sf_words_per_block_k + k]
                    T.barrier_arrive(loaded[stage])

            elif tx < 128:
                if consumer_regs > 0:
                    T.set_max_nreg(consumer_regs, 1)
                C0_local = T.alloc_fragment((half_block_M, block_N), accum_dtype)
                T.clear(C0_local)
                for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.barrier_wait(loaded[stage], phase)
                    if sf_load_mode == "direct":
                        T.mma_gemm_blockscaled(
                            A_shared[stage, 0:half_block_M, :],
                            B_shared[stage, :, :],
                            C0_local,
                            SFA[
                                by * block_M : by * block_M + half_block_M,
                                ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                            ],
                            SFB[
                                bx * block_N : (bx + 1) * block_N,
                                ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                            ],
                            transpose_B=True,
                            policy=warp_policy,
                            clear_accum=False,
                            k_start=0,
                            sf_a_granularity_k=sf_granularity_k,
                            sf_b_granularity_k=sf_granularity_k,
                            micro_pipeline=micro_pipeline,
                            sf_layout=sf_layout,
                        )
                    else:
                        T.mma_gemm_blockscaled(
                            A_shared[stage, 0:half_block_M, :],
                            B_shared[stage, :, :],
                            C0_local,
                            SFA_shared[stage, 0:half_block_M, :],
                            SFB_shared[stage, :, :],
                            transpose_B=True,
                            policy=warp_policy,
                            clear_accum=False,
                            k_start=0,
                            sf_a_granularity_k=sf_granularity_k,
                            sf_b_granularity_k=sf_granularity_k,
                            micro_pipeline=micro_pipeline,
                            sf_layout=sf_layout,
                        )
                    T.barrier_arrive(consumed[stage])
                T.copy(C0_local, C[by * block_M, bx * block_N])

            else:
                if consumer_regs > 0:
                    T.set_max_nreg(consumer_regs, 1)
                C1_local = T.alloc_fragment((half_block_M, block_N), accum_dtype)
                T.clear(C1_local)
                for ko in T.unroll(K // block_K, explicit=False, unroll_factor=1):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.barrier_wait(loaded[stage], phase)
                    if sf_load_mode == "direct":
                        T.mma_gemm_blockscaled(
                            A_shared[stage, half_block_M:block_M, :],
                            B_shared[stage, :, :],
                            C1_local,
                            SFA[
                                by * block_M + half_block_M : (by + 1) * block_M,
                                ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                            ],
                            SFB[
                                bx * block_N : (bx + 1) * block_N,
                                ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                            ],
                            transpose_B=True,
                            policy=warp_policy,
                            clear_accum=False,
                            k_start=0,
                            sf_a_granularity_k=sf_granularity_k,
                            sf_b_granularity_k=sf_granularity_k,
                            micro_pipeline=micro_pipeline,
                            sf_layout=sf_layout,
                        )
                    else:
                        T.mma_gemm_blockscaled(
                            A_shared[stage, half_block_M:block_M, :],
                            B_shared[stage, :, :],
                            C1_local,
                            SFA_shared[stage, half_block_M:block_M, :],
                            SFB_shared[stage, :, :],
                            transpose_B=True,
                            policy=warp_policy,
                            clear_accum=False,
                            k_start=0,
                            sf_a_granularity_k=sf_granularity_k,
                            sf_b_granularity_k=sf_granularity_k,
                            micro_pipeline=micro_pipeline,
                            sf_layout=sf_layout,
                        )
                    T.barrier_arrive(consumed[stage])
                T.copy(C1_local, C[by * block_M + half_block_M, bx * block_N])

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
    # Two FP4 e2m1 values with raw code 0x2, packed into one byte.
    return torch.full((rows, cols // 2), 0x22, device="cuda", dtype=torch.int8)


def _make_constant_scale_words(rows: int, k: int, byte: int = 0x38) -> torch.Tensor:
    word = byte | (byte << 8) | (byte << 16) | (byte << 24)
    return torch.full((rows, k // 64), word, device="cuda", dtype=torch.uint32)


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
    words = scale_bytes[:, 0::4] | (scale_bytes[:, 1::4] << 8) | (scale_bytes[:, 2::4] << 16) | (scale_bytes[:, 3::4] << 24)
    return words.to(torch.uint32)


def _swizzle_scale_words_cutlass_128x4(words: torch.Tensor, block_rows: int = 128, block_words: int = 4) -> torch.Tensor:
    rows, cols = words.shape
    if rows % block_rows != 0 or cols % block_words != 0:
        raise ValueError(
            f"cutlass_128x4 scale storage requires rows multiple of {block_rows} "
            f"and cols multiple of {block_words}, got {tuple(words.shape)}"
        )
    src = words.view(rows // block_rows, block_rows, cols // block_words, block_words)
    dst = torch.empty_like(src)
    for i in range(block_rows):
        for k in range(block_words):
            dst[:, k * 32 + (i % 32), :, i // 32] = src[:, i, :, k]
    return dst.view(rows, cols).contiguous()


def _decode_binary_scale_words(words: torch.Tensor, k: int) -> torch.Tensor:
    w = words.to(torch.int64)
    scale_bytes = torch.empty((words.shape[0], k // 16), device=words.device, dtype=torch.int64)
    scale_bytes[:, 0::4] = w & 0xFF
    scale_bytes[:, 1::4] = (w >> 8) & 0xFF
    scale_bytes[:, 2::4] = (w >> 16) & 0xFF
    scale_bytes[:, 3::4] = (w >> 24) & 0xFF
    return (scale_bytes != 0).to(torch.float32)


def _decode_rowmajor_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _verify_tilelang_output(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,
    out_dtype: torch.dtype,
    scale_mode: str,
    block_m: int = 128,
    block_n: int = 128,
    sm_num: int | None = None,
) -> None:
    A_full = _decode_rowmajor_fp4(A, A.shape[0], A.shape[1] * 2)
    B_full = _decode_rowmajor_fp4(B, B.shape[0], B.shape[1] * 2)
    if scale_mode == "constant":
        ref = (A_full @ B_full.T).to(out_dtype)
    elif scale_mode in ("random_binary", "random_sfa", "random_sfb"):
        sfa = _decode_binary_scale_words(SFA, A_full.shape[1])
        sfb = _decode_binary_scale_words(SFB, B_full.shape[1])
        ref_f32 = torch.zeros((A_full.shape[0], B_full.shape[0]), device=C.device, dtype=torch.float32)
        for k_sf in range(A_full.shape[1] // 16):
            k0 = k_sf * 16
            k1 = k0 + 16
            a_chunk = A_full[:, k0:k1] * sfa[:, k_sf].unsqueeze(1)
            b_chunk = B_full[:, k0:k1] * sfb[:, k_sf].unsqueeze(1)
            ref_f32 += a_chunk @ b_chunk.T
        ref = ref_f32.to(out_dtype)
    else:
        raise ValueError(f"Unsupported scale_mode={scale_mode!r}")
    try:
        torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)
    except AssertionError:
        diff = ref != C
        mismatch = int(diff.sum().item())
        total = diff.numel()
        print(f"TileLang mismatch summary: {mismatch}/{total} ({mismatch / total:.2%})")
        if C.shape[0] % block_m == 0 and C.shape[1] % block_n == 0:
            m_blocks = C.shape[0] // block_m
            n_blocks = C.shape[1] // block_n
            tile_counts = diff.reshape(m_blocks, block_m, n_blocks, block_n).sum(dim=(1, 3))
            flat = tile_counts.flatten()
            top = torch.topk(flat, k=min(12, flat.numel()))
            for value, index in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                if int(value) == 0:
                    continue
                tile_m = index // n_blocks
                tile_n = index % n_blocks
                tile_id = tile_m * n_blocks + tile_n
                owner = "unknown"
                if sm_num is not None and sm_num > 0:
                    owner = f"WG{(tile_id // sm_num) & 1}"
                print(f"  tile m={tile_m} n={tile_n} tile_id={tile_id} owner={owner} mismatch={int(value)}/{block_m * block_n}")
            if block_n % 64 == 0:
                panel_counts = diff.reshape(m_blocks, block_m, n_blocks, block_n // 64, 64).sum(dim=(1, 4))
                panel_flat = panel_counts.flatten()
                panel_top = torch.topk(panel_flat, k=min(12, panel_flat.numel()))
                for value, index in zip(panel_top.values.cpu().tolist(), panel_top.indices.cpu().tolist()):
                    if int(value) == 0:
                        continue
                    panel = index % (block_n // 64)
                    tile_index = index // (block_n // 64)
                    tile_m = tile_index // n_blocks
                    tile_n = tile_index % n_blocks
                    tile_id = tile_m * n_blocks + tile_n
                    owner = "unknown"
                    if sm_num is not None and sm_num > 0:
                        owner = f"WG{(tile_id // sm_num) & 1}"
                    print(
                        "  panel64 "
                        f"m={tile_m} n={tile_n} panel={panel} tile_id={tile_id} owner={owner} "
                        f"mismatch={int(value)}/{block_m * 64}"
                    )
            if block_n % 32 == 0:
                panel32_counts = diff.reshape(m_blocks, block_m, n_blocks, block_n // 32, 32).sum(dim=(1, 4))
                panel32_flat = panel32_counts.flatten()
                panel32_top = torch.topk(panel32_flat, k=min(16, panel32_flat.numel()))
                for value, index in zip(panel32_top.values.cpu().tolist(), panel32_top.indices.cpu().tolist()):
                    if int(value) == 0:
                        continue
                    panel = index % (block_n // 32)
                    tile_index = index // (block_n // 32)
                    tile_m = tile_index // n_blocks
                    tile_n = tile_index % n_blocks
                    tile_id = tile_m * n_blocks + tile_n
                    owner = "unknown"
                    if sm_num is not None and sm_num > 0:
                        owner = f"WG{(tile_id // sm_num) & 1}"
                    print(
                        "  panel32 "
                        f"m={tile_m} n={tile_n} panel={panel} tile_id={tile_id} owner={owner} "
                        f"mismatch={int(value)}/{block_m * 32}"
                    )
                tile0 = diff[:block_m, :block_n]
                for panel in range(block_n // 32):
                    panel_diff = tile0[:, panel * 32 : (panel + 1) * 32]
                    row32 = panel_diff.reshape(4, 32, 32).sum(dim=(1, 2)).cpu().tolist()
                    col8 = panel_diff.reshape(block_m, 4, 8).sum(dim=(0, 2)).cpu().tolist()
                    print(f"  tile0-panel32-buckets panel={panel} row32={[int(x) for x in row32]} col8={[int(x) for x in col8]}")
        raise


def _print_kblock_debug(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    A_full = _decode_rowmajor_fp4(A, A.shape[0], A.shape[1] * 2)
    B_full = _decode_rowmajor_fp4(B, B.shape[0], B.shape[1] * 2)
    refs = []
    for k0 in range(0, A_full.shape[1], block_k):
        k1 = k0 + block_k
        refs.append((A_full[:, k0:k1] @ B_full[:, k0:k1].T).to(out_dtype))
    full_ref = (A_full @ B_full.T).to(out_dtype)

    named_refs = [(f"kblock{idx}", ref) for idx, ref in enumerate(refs)] + [("full", full_ref)]
    for name, ref in named_refs:
        diff = ref != C
        print(
            f"kblock-debug {name}: mismatch={int(diff.sum().item())}/{C.numel()}, "
            f"maxabs={float((C.float() - ref.float()).abs().max().item())}"
        )

    print("kblock-debug tile mismatches vs full:")
    for tm in range(0, C.shape[0], block_m):
        row = []
        for tn in range(0, C.shape[1], block_n):
            row.append(int((C[tm : tm + block_m, tn : tn + block_n] != full_ref[tm : tm + block_m, tn : tn + block_n]).sum().item()))
        print(f"kblock-debug tile-row {tm // block_m}: {row}")

    coords = torch.nonzero(full_ref != C)[:16].detach().cpu().tolist()
    samples = [
        (
            int(i),
            int(j),
            float(C[i, j].float().item()),
            float(full_ref[i, j].float().item()),
            *[float(ref[i, j].float().item()) for ref in refs[:2]],
        )
        for i, j in coords
    ]
    print(f"kblock-debug samples: {samples}")


def _verify_tilelang_single_omma_site(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    scratch_ones: bool = False,
    scratch_expected: float = 64.0,
) -> None:
    """Verify the private fixed site: ki=0, m_atom=0, n8=0, half=0."""
    A_full = _decode_rowmajor_fp4(A, A.shape[0], A.shape[1] * 2)
    B_full = _decode_rowmajor_fp4(B, B.shape[0], B.shape[1] * 2)
    ref_f32 = torch.zeros(C.shape, device=C.device, dtype=torch.float32)
    for tile_m in range(0, A_full.shape[0], block_m):
        for tile_n in range(0, B_full.shape[0], block_n):
            for k_base in range(0, A_full.shape[1], block_k):
                k_range = slice(k_base, k_base + 64)
                for warp_m_base in (0, 64):
                    for warp_n_base in (0, 64):
                        rows = slice(tile_m + warp_m_base, tile_m + warp_m_base + 16)
                        cols = slice(tile_n + warp_n_base, tile_n + warp_n_base + 8)
                        if scratch_ones:
                            ref_f32[rows, cols] += scratch_expected
                        else:
                            ref_f32[rows, cols] += A_full[rows, k_range] @ B_full[cols, k_range].T
    ref = ref_f32.to(out_dtype)
    diff_mask = ref != C
    if torch.any(diff_mask):
        actual_mask = C != 0
        ref_mask = ref != 0
        overlap_mask = actual_mask & ref_mask
        print(
            "single-OMMA diagnostic: "
            f"actual_nonzero={int(actual_mask.sum().item())}, "
            f"ref_nonzero={int(ref_mask.sum().item())}, "
            f"overlap_nonzero={int(overlap_mask.sum().item())}, "
            f"mismatch={int(diff_mask.sum().item())}"
        )
        coords = torch.nonzero(diff_mask)[:16].cpu().tolist()
        samples = [(int(i), int(j), float(C[i, j].float().item()), float(ref[i, j].float().item())) for i, j in coords]
        print(f"single-OMMA diagnostic samples: {samples}")
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


def _print_single_omma_active_stats(C: torch.Tensor, block_m: int, block_n: int) -> None:
    active = []
    for tile_m in range(0, C.shape[0], block_m):
        for tile_n in range(0, C.shape[1], block_n):
            for warp_m_base in (0, 64):
                for warp_n_base in (0, 64):
                    active.append(
                        C[
                            tile_m + warp_m_base : tile_m + warp_m_base + 16,
                            tile_n + warp_n_base : tile_n + warp_n_base + 8,
                        ]
                        .float()
                        .reshape(-1)
                    )
    values = torch.cat(active)
    unique = torch.unique(values)
    preview = unique[:16].detach().cpu().tolist()
    print(
        "single-OMMA active stats: "
        f"count={values.numel()}, nonzero={int((values != 0).sum().item())}, "
        f"unique_count={unique.numel()}, unique_preview={preview}"
    )


def _print_reg_debug(C: torch.Tensor, dump_path: str | None = None, limit: int = 160) -> None:
    if C.dtype != torch.float32:
        print(f"reg-debug requires --out-dtype float32, got {C.dtype}")
        return
    raw = C.detach().cpu().contiguous().view(torch.int32)
    if dump_path:
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("row,col,word_hex\n")
            for i in range(raw.shape[0]):
                for j in range(raw.shape[1]):
                    value = int(raw[i, j].item()) & 0xFFFFFFFF
                    f.write(f"{i},{j},0x{value:08x}\n")
        print(f"reg-debug csv={path} entries={raw.numel()}")
    nz = torch.nonzero(raw, as_tuple=False)
    print(f"reg-debug nonzero_words={int(nz.shape[0])}")
    rows = []
    for coord in nz[:limit].tolist():
        i, j = int(coord[0]), int(coord[1])
        value = int(raw[i, j].item()) & 0xFFFFFFFF
        rows.append((i, j, f"0x{value:08x}"))
    print(f"reg-debug first_words={rows}")


def run_tilelang(args: argparse.Namespace) -> tuple[float, float]:
    out_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32
    warp_policy = getattr(T.GemmWarpPolicy, args.warp_policy)

    if args.manual_ws2:
        if args.threads != 384:
            print("manual_ws2 uses a fixed 384-thread CTA; ignoring --threads")
        if args.load_mode != "manual":
            print("manual_ws2 uses explicit T.tma_copy; ignoring --load-mode")
        producer_regs = args.producer_regs if args.producer_regs > 0 else 80
        consumer_regs = args.consumer_regs if args.consumer_regs > 0 else 168
        kernel = tilelang_nvfp4_gemm_manual_ws2(
            args.m,
            args.n,
            args.k,
            args.block_m,
            args.block_n,
            args.block_k,
            args.num_stages,
            warp_policy,
            out_dtype,
            None if args.micro_pipeline == "none" else args.micro_pipeline,
            producer_regs,
            consumer_regs,
            args.manual_ws2_sf_load,
            args.manual_ws2_split,
            args.manual_ws2_sf_layout,
            args.manual_ws2_ab_shared_storage,
        )
    else:
        kernel = tilelang_nvfp4_gemm(
            args.m,
            args.n,
            args.k,
            args.block_m,
            args.block_n,
            args.block_k,
            args.num_stages,
            args.threads,
            warp_policy,
            out_dtype,
            None if args.micro_pipeline == "none" else args.micro_pipeline,
            args.load_mode,
            args.producer_regs,
            args.consumer_regs,
        )

    source = kernel.get_kernel_source()
    if args.dump_source:
        dump_path = Path(args.dump_source)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(source)
        print(f"TileLang CUDA source: {dump_path}")
    if args.dump_sass:
        sass_path = Path(args.dump_sass)
        sass_path.parent.mkdir(parents=True, exist_ok=True)
        saved_compile_flags = kernel.compile_flags
        device_compile_flags = _device_compile_flags()
        if device_compile_flags:
            merged_compile_flags = list(saved_compile_flags or [])
            for flag in device_compile_flags:
                if flag not in merged_compile_flags:
                    merged_compile_flags.append(flag)
            kernel.compile_flags = merged_compile_flags
        kernel.export_sass(str(sass_path))
        kernel.compile_flags = saved_compile_flags
        print(f"TileLang SASS: {sass_path}")
    if (
        args.manual_ws2_split != "pp_wait"
        and "sm120_mma_sync_blockscaled" not in source
        and "sm120_mma_blockscaled_kblock_fulltile" not in source
        and "sm120_mma_blockscaled_cute_consumer_bridge" not in source
    ):
        raise RuntimeError("TileLang source did not lower to an SM120 blockscaled MMA helper")

    if args.input_mode == "ones":
        A = _make_ones_packed_fp4(args.m, args.k)
        B = _make_ones_packed_fp4(args.n, args.k)
    else:
        A = _make_packed_fp4(args.m, args.k, seed=args.seed)
        B = _make_packed_fp4(args.n, args.k, seed=args.seed + 1)
    if args.scale_mode == "constant":
        SFA_semantic = _make_constant_scale_words(args.m, args.k)
        SFB_semantic = _make_constant_scale_words(args.n, args.k)
    elif args.scale_mode == "random_binary":
        SFA_semantic = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
        SFB_semantic = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)
    elif args.scale_mode == "random_sfa":
        SFA_semantic = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
        SFB_semantic = _make_constant_scale_words(args.n, args.k)
    elif args.scale_mode == "random_sfb":
        SFA_semantic = _make_constant_scale_words(args.m, args.k)
        SFB_semantic = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)
    else:
        raise ValueError(f"Unsupported scale_mode={args.scale_mode!r}")
    if args.scale_storage_layout == "rowmajor":
        SFA = SFA_semantic
        SFB = SFB_semantic
    elif args.scale_storage_layout == "cutlass_128x4":
        SFA = _swizzle_scale_words_cutlass_128x4(SFA_semantic)
        SFB = _swizzle_scale_words_cutlass_128x4(SFB_semantic)
    else:
        raise ValueError(f"Unsupported scale_storage_layout={args.scale_storage_layout!r}")
    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)

    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.profile_only and args.profile_warmup_launches < 0:
        raise ValueError("--profile-warmup-launches must be non-negative")
    if args.profile_only:
        for _ in range(args.profile_warmup_launches):
            kernel(A, B, SFA, SFB, C)
        kernel(A, B, SFA, SFB, C)
        torch.cuda.synchronize()

    if args.manual_ws2_reg_debug:
        _print_reg_debug(C, args.manual_ws2_reg_debug_dump)
        return float("nan"), float("nan")

    if args.manual_ws2_kblock_debug:
        if args.scale_mode != "constant":
            print("kblock-debug currently assumes --scale-mode constant; skipping")
        else:
            _print_kblock_debug(A, B, C, out_torch_dtype, args.block_m, args.block_n, args.block_k)

    if args.manual_ws2_ab_copy_view.startswith("fp4_ldsm_single_omma_scratch"):
        _print_single_omma_active_stats(C, args.block_m, args.block_n)

    if args.verify:
        if "single_omma" in args.manual_ws2_ab_copy_view:
            _verify_tilelang_single_omma_site(
                A,
                B,
                C,
                out_torch_dtype,
                args.block_m,
                args.block_n,
                args.block_k,
                scratch_ones=args.manual_ws2_ab_copy_view.startswith("fp4_ldsm_single_omma_scratch"),
                scratch_expected=32.0 if args.manual_ws2_ab_copy_view == "fp4_ldsm_single_omma_scratch_shifted_ones" else 64.0,
            )
        else:
            _verify_tilelang_output(
                A,
                B,
                SFA_semantic,
                SFB_semantic,
                C,
                out_torch_dtype,
                args.scale_mode,
                args.block_m,
                args.block_n,
                driver.get_num_sms(),
            )
        print("TileLang correctness: passed")

    if args.profile_only:
        launches = 2 + args.profile_warmup_launches
        print(f"TileLang profile-only: ran {launches} kernel launch(es)")
        return float("nan"), float("nan")

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
            "--skip-verification",
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
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warp-policy", choices=["Square", "FullRow", "FullCol"], default="Square")
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--input-mode", choices=["random", "ones"], default="random")
    parser.add_argument(
        "--scale-mode",
        choices=["constant", "random_binary", "random_sfa", "random_sfb"],
        default="constant",
    )
    parser.add_argument("--scale-storage-layout", choices=["rowmajor", "cutlass_128x4"], default="rowmajor")
    parser.add_argument(
        "--micro-pipeline",
        choices=[
            "none",
            "b_pingpong",
            "b_pingpong_scale_prefetch",
            "k_static_b_pingpong_scale_stream",
            "k_static_b_atom_n8_stream",
            "k_static_b_atom_n8_stream_vecsf",
            "k_static_full_b_atom_scale_stream",
            "sm120_kblock_fulltile",
            "sm120_kblock_fulltile_selector_probe",
            "sm120_kblock_fulltile_b_owner_probe",
            "sm120_kblock_fulltile_ab_owner_probe",
            "sm120_kblock_fulltile_ab_owner_wide_probe",
            "sm120_backend_kblock_fulltile",
            "sm120_backend_kblock_fulltile_ab_owner_wide",
            "sm120_backend_kblock_fulltile_afull_bpanel_owner_wide",
            "sm120_backend_kblock_fulltile_package_pingpong",
            "sm120_cute_consumer_bridge_skeleton",
            "sm120_pkg_atom_neutral",
            "k_static",
            "k_static_scale_prefetch",
            "k_static_scale_stream",
            "k_static_b_all_prefetch_scale_stream",
            "k_static_scale_stream_cutlass_order",
        ],
        default="none",
    )
    parser.add_argument("--load-mode", choices=["manual", "tcopy", "tcopy_ab"], default="manual")
    parser.add_argument("--manual-ws2", action="store_true")
    parser.add_argument(
        "--manual-ws2-sf-load",
        choices=["tma", "parallel", "serial", "none", "direct", "reorder"],
        default="tma",
    )
    parser.add_argument(
        "--manual-ws2-split",
        choices=[
            "m",
            "n",
            "full",
            "full128",
            "pp",
            "pp_one",
            "pp_wait",
            "pp_stream",
            "pp_stream_output_tma_panel32",
            "pp_stream_output_tma_panel32_pipe2",
            "pp_stream_output_tma_epi64x32",
            "pp_stream_tma_store_literal",
            "pp_stream_tma_store_literal_copy",
            "pp_stream_panel32_tma_store",
            "pp_stream_panel64_tma_store",
        ],
        default="m",
    )
    parser.add_argument("--manual-ws2-sf-layout", choices=["rowmajor", "cutlass_128x4"], default="rowmajor")
    parser.add_argument("--manual-ws2-ab-shared-storage", choices=["packed", "unpacked"], default="packed")
    parser.add_argument("--manual-ws2-scratch-byte", type=lambda x: int(x, 0))
    parser.add_argument("--manual-ws2-reg-debug", action="store_true")
    parser.add_argument("--manual-ws2-reg-debug-mode", choices=["value", "tag"], default="value")
    parser.add_argument("--manual-ws2-reg-debug-dump")
    parser.add_argument("--manual-ws2-kblock-debug", action="store_true")
    parser.add_argument(
        "--manual-ws2-ab-copy-view",
        choices=[
            "legacy_b16",
            "legacy_b16_single_omma",
            "fp4_ldsm",
            "fp4_ldsm_single_omma",
            "fp4_ldsm_noshift",
            "fp4_ldsm_single_omma_noshift",
            "fp4_ldsm_cute_rowstart",
            "fp4_ldsm_cute_rowstart_noshift",
            "fp4_ldsm_cute_rowstart_aonly",
            "fp4_ldsm_cute_rowstart_bonly",
            "fp4_ldsm_cute_rowstart_caccum",
            "fp4_ldsm_cute_rowstart_single_omma",
            "fp4_ldsm_cute_rowstart_single_omma_noshift",
            "fp4_ldsm_cute_rowstart_single_omma_aonly",
            "fp4_ldsm_cute_rowstart_single_omma_bonly",
            "fp4_ldsm_single_omma_real_scratch",
            "fp4_ldsm_single_omma_scratch_ones",
            "fp4_ldsm_single_omma_scratch_shifted_ones",
        ],
        default="legacy_b16",
    )
    parser.add_argument("--enable-tma", action="store_true")
    parser.add_argument("--enable-warp-specialized", action="store_true")
    parser.add_argument("--producer-regs", type=int, default=0)
    parser.add_argument("--consumer-regs", type=int, default=0)
    parser.add_argument("--maxrregcount", type=int)
    parser.add_argument("--ptxas-verbose", action="store_true")
    parser.add_argument("--dump-source")
    parser.add_argument("--dump-sass")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="mean")
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--n-warmup", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-warmup-launches", type=int, default=0)
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
    print(
        f"TileLang tile: {args.block_m}x{args.block_n}x{args.block_k}, "
        f"threads={args.threads}, policy={args.warp_policy}, output={args.out_dtype}, "
        f"micro_pipeline={args.micro_pipeline}, load_mode={args.load_mode}, "
        f"input_mode={args.input_mode}, scale_mode={args.scale_mode}, "
        f"scale_storage_layout={args.scale_storage_layout}, "
        f"manual_ws2={args.manual_ws2}, manual_ws2_sf_load={args.manual_ws2_sf_load}, "
        f"manual_ws2_split={args.manual_ws2_split}, manual_ws2_sf_layout={args.manual_ws2_sf_layout}, "
        f"manual_ws2_ab_shared_storage={args.manual_ws2_ab_shared_storage}, "
        f"manual_ws2_ab_copy_view={args.manual_ws2_ab_copy_view}"
    )
    print(
        f"TileLang pipeline: tma={args.enable_tma}, ws={args.enable_warp_specialized}, "
        f"producer_regs={args.producer_regs}, consumer_regs={args.consumer_regs}, "
        f"maxrregcount={args.maxrregcount}"
    )

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
