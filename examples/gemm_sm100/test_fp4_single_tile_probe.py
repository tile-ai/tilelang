"""Single-tile FP4 GEMM probe for SM100/SM110.

Purpose:
  1. Remove the second K tile so we only exercise the first
     TMA -> tcgen05.mma -> mbarrier wait sequence.
  2. Allow quick TMA vs non-TMA comparison via TL_FP4_DISABLE_TMA=1.
  3. Allow isolating the epilogue path via TL_FP4_LINEAR_EPILOGUE=1,
     so we can compare "TMA input + linear global store" against the
     default "TMA input + TMA store" path.
  4. Allow forcing an explicit fence between copy/wait and tcgen05.mma via
     TL_FP4_FORCE_FENCE=1 to test whether the TMA->UMMA handoff is missing a
     proxy fence.
  5. Allow stopping immediately after input load via TL_FP4_TMA_ONLY=1, so
     TMA wait and tcgen05 MMA wait can be isolated.
  6. Allow forcing the same descriptor/gap-aware input TMA path without MMA via
     TL_FP4_TMA_ONLY_DESCRIPTOR=1.

Run:
  python examples/gemm_sm100/test_fp4_single_tile_probe.py
  TL_FP4_DISABLE_TMA=1 python examples/gemm_sm100/test_fp4_single_tile_probe.py
  TL_FP4_COMPILE_ONLY=1 python examples/gemm_sm100/test_fp4_single_tile_probe.py
  TL_FP4_LINEAR_EPILOGUE=1 python examples/gemm_sm100/test_fp4_single_tile_probe.py
  TL_FP4_FORCE_FENCE=1 python examples/gemm_sm100/test_fp4_single_tile_probe.py
  TL_FP4_TMA_ONLY=1 python examples/gemm_sm100/test_fp4_single_tile_probe.py
  TL_FP4_TMA_ONLY=1 TL_FP4_TMA_ONLY_DESCRIPTOR=1 python examples/gemm_sm100/test_fp4_single_tile_probe.py
"""

import os

import torch
import tilelang
import tilelang.language as T

M, N, K = 128, 64, 128
block_M, block_N, block_K = 128, 64, 128
in_dtype = T.float4_e2m1fn
out_dtype = T.float32
accum_dtype = T.float32

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


def unpack_fp4_to_float(packed_int8, rows, cols):
    lut = torch.tensor(
        FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device
    )
    flat = packed_int8.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def make_fp4_diagonal(rows, cols):
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device="cuda")
    fp4_one = 2  # FP4 encoding for 1.0
    diag = torch.arange(min(rows, cols), device="cuda", dtype=torch.int64)
    byte_idx = diag // 2
    nibble_shift = (diag % 2).to(torch.uint8) * 4
    packed[diag, byte_idx] = torch.bitwise_left_shift(
        torch.full_like(nibble_shift, fp4_one, dtype=torch.uint8), nibble_shift
    )
    return packed.to(torch.int8)


def make_fp4_row_pattern(rows, cols):
    vals = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device="cuda"
    )
    row_vals = vals[torch.arange(rows, device="cuda", dtype=torch.int64) % 8]
    packed_row_vals = row_vals | torch.bitwise_left_shift(row_vals, 4)
    packed = packed_row_vals[:, None].expand(rows, cols // 2).contiguous()
    return packed.to(torch.int8)


def print_generated_kernel_debug_summary(kernel_src_path: str):
    keywords = (
        "arrive_and_expect_tx",
        "expect_transaction",
        "tma_load(",
        "tma_store(",
        "tma_store_arrive",
        "tma_store_wait",
        "initialize_tcgen05_descriptor",
        "fence_proxy_async",
        "tcgen05mma_ss",
        ".wait(",
    )

    with open(kernel_src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Generated kernel debug summary:")
    for line_no, line in enumerate(lines, start=1):
        if any(keyword in line for keyword in keywords):
            print(f"{line_no:4d}: {line.rstrip()}")


def fp4_single_tile_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    linear_epilogue,
    force_fence,
    tma_only,
    tma_only_descriptor,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            if not disable_tma:
                load_mbar_a = T.alloc_barrier(128)
                load_mbar_b = T.alloc_barrier(128)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            if not linear_epilogue:
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            if tma_only and tma_only_descriptor:
                T.annotate_layout(
                    {
                        A_shared: tilelang.layout.make_tcgen05mma_swizzled_layout(
                            A_shared, continuity=block_K, k_major=True
                        ),
                        B_shared: tilelang.layout.make_tcgen05mma_swizzled_layout(
                            B_shared, continuity=block_K, k_major=True
                        ),
                    }
                )

            if disable_tma:
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
            else:
                T.tma_copy(A[0, 0], A_shared, barrier=load_mbar_a)
                T.tma_copy(B[0, 0], B_shared, barrier=load_mbar_b)
                T.barrier_arrive(load_mbar_a)
                T.barrier_arrive(load_mbar_b)
                T.mbarrier_wait_parity(load_mbar_a, 0)
                T.mbarrier_wait_parity(load_mbar_b, 0)
                T.fence_proxy_async()
            if tma_only:
                return
            if force_fence:
                T.fence_proxy_async()
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_A=False,
                transpose_B=True,
                mbar=mbar,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar, 0)

            T.copy(C_tmem, C_local)
            if linear_epilogue:
                T.copy(C_local, C[0, 0])
            else:
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[0, 0])

    return main


disable_tma = os.environ.get("TL_FP4_DISABLE_TMA", "0") == "1"
linear_epilogue = os.environ.get("TL_FP4_LINEAR_EPILOGUE", "0") == "1"
force_fence = os.environ.get("TL_FP4_FORCE_FENCE", "0") == "1"
tma_only = os.environ.get("TL_FP4_TMA_ONLY", "0") == "1"
tma_only_descriptor = os.environ.get("TL_FP4_TMA_ONLY_DESCRIPTOR", "0") == "1"
mode_parts = ["non_tma" if disable_tma else "tma"]
if linear_epilogue:
    mode_parts.append("linear_epilogue")
if force_fence:
    mode_parts.append("force_fence")
if tma_only:
    mode_parts.append("tma_only")
if tma_only_descriptor:
    mode_parts.append("descriptor")
mode = ".".join(mode_parts)
print(
    f"FP4 single-tile probe: mode={mode}, M={M}, N={N}, K={K}, "
    f"block=({block_M},{block_N},{block_K})"
)

func = fp4_single_tile_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    linear_epilogue,
    force_fence,
    tma_only,
    tma_only_descriptor,
)
kernel = tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: disable_tma,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
print("Compiled OK")

kernel_src_path = os.path.join(
    os.path.dirname(__file__), f"test_fp4_single_tile_probe.{mode}.generated.cu"
)
with open(kernel_src_path, "w", encoding="utf-8") as f:
    f.write(kernel.get_kernel_source())
print(f"Kernel source written to: {kernel_src_path}")
print_generated_kernel_debug_summary(kernel_src_path)

if os.environ.get("TL_FP4_COMPILE_ONLY", "0") == "1":
    print("TL_FP4_COMPILE_ONLY=1, skip kernel launch.")
    raise SystemExit(0)

print("Building FP4 test inputs...")
a_packed = make_fp4_row_pattern(M, K)
b_packed = make_fp4_diagonal(N, K)
torch.cuda.synchronize()

print("Launching kernel...")
c = kernel(a_packed, b_packed)
print("Kernel launch returned, synchronizing...")
torch.cuda.synchronize()
print("Kernel finished.")

if tma_only:
    print("TL_FP4_TMA_ONLY=1, skipped GEMM and output comparison.")
    raise SystemExit(0)

a_float = unpack_fp4_to_float(a_packed, M, K)
b_float = unpack_fp4_to_float(b_packed, N, K)
ref_c = a_float @ b_float.T

diff = (c.float() - ref_c).abs()
max_diff = diff.max().item()
rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
print(f"max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
