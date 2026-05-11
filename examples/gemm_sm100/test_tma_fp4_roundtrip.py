"""Diagnostic: force TMA load for FP4 on SM100/SM110.

This test isolates the GMEM -> SMEM TMA load path for FP4 ALIGN16B:
  1. input tile is loaded with explicit ``T.tma_copy(...)`` so it cannot
     silently fall back to a normal copy;
  2. shared memory uses the ALIGN16B gap-aware layout that GEMM also relies on;
  3. the write-back to GMEM stays as a normal copy so the result reflects only
     whether the TMA load populated SMEM correctly.

Run: TILELANG_DISABLE_CACHE=1 python examples/gemm_sm100/test_tma_fp4_roundtrip.py
"""

import os

import torch
import tilelang
import tilelang.language as T

M, K = 256, 256
block_M, block_K = 128, 128
in_dtype = T.float4_e2m1fn


def tma_roundtrip_kernel(
    M,
    K,
    block_M,
    block_K,
    in_dtype,
    force_tma_load: bool,
    skip_store: bool,
    use_explicit_tma_copy: bool,
):
    """Copy A[block_M, block_K] to SMEM, then write back to Out."""
    A_shape = (M, K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        Out: T.Tensor(A_shape, in_dtype),
    ):
        with T.Kernel(1, T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            if force_tma_load and use_explicit_tma_copy:
                mbar = T.alloc_barrier(128)

            T.annotate_layout(
                {
                    A_shared: tilelang.layout.make_align16b_swizzled_layout(A_shared),
                }
            )

            if force_tma_load and use_explicit_tma_copy:
                T.tma_copy(A[by * block_M, 0], A_shared, barrier=mbar)
                T.barrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, 0)
                T.fence_proxy_async()
            elif force_tma_load:
                T.copy(A[by * block_M, 0], A_shared)
            else:
                T.copy(A[by * block_M, 0], A_shared)
            if not skip_store:
                T.copy(A_shared, Out[by * block_M, 0])

    return main


def dump_and_check_kernel_source(kernel, mode: str, expect_tma_load: bool):
    kernel_src_path = os.path.join(
        os.path.dirname(__file__),
        f"test_tma_fp4_roundtrip.{mode}.generated.cu",
    )
    src = kernel.get_kernel_source()
    with open(kernel_src_path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Kernel source written to: {kernel_src_path}")

    keywords = (
        "arrive_and_expect_tx",
        "tma_load(",
        "tma_store(",
        "fence_proxy_async",
    )
    print(f"{mode} kernel debug summary:")
    matched = []
    for line_no, line in enumerate(src.splitlines(), start=1):
        if any(keyword in line for keyword in keywords):
            matched.append((line_no, line.rstrip()))
            print(f"{line_no:4d}: {line.rstrip()}")

    has_tma_load = any("tma_load(" in line for _, line in matched)
    if expect_tma_load and not has_tma_load:
        raise RuntimeError(
            f"{mode} kernel was expected to use TMA load, but generated source "
            "contains no tl::tma_load(...) call."
        )
    if not expect_tma_load and has_tma_load:
        raise RuntimeError(
            f"{mode} kernel was expected to disable TMA load, but generated source "
            "still contains tl::tma_load(...)."
        )


print(f"TMA FP4 load diagnostic: M={M}, K={K}, block=({block_M},{block_K})")
run_ref = os.environ.get("TL_FP4_SKIP_REF", "0") != "1"
skip_store = os.environ.get("TL_FP4_SKIP_STORE", "0") == "1"
use_explicit_tma_copy = os.environ.get("TL_FP4_USE_TMA_COPY", "1") == "1"
mode = "explicit_tma_copy" if use_explicit_tma_copy else "auto_tma_lower"
print(f"Load mode: {mode}")

# --- Build with explicit TMA load ---
func_tma = tma_roundtrip_kernel(
    M,
    K,
    block_M,
    block_K,
    in_dtype,
    force_tma_load=True,
    skip_store=skip_store,
    use_explicit_tma_copy=use_explicit_tma_copy,
)
kernel_tma = tilelang.compile(
    func_tma,
    out_idx=[1],
    target="cuda",
    pass_configs={
        # Keep explicit T.tma_copy(), but disable implicit TMA selection on the
        # store path so this test only validates the load side.
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
print("TMA kernel compiled OK")
dump_and_check_kernel_source(kernel_tma, "tma", expect_tma_load=True)

kernel_ref = None
if run_ref:
    # --- Build with TMA disabled (reference) ---
    func_ref = tma_roundtrip_kernel(
        M,
        K,
        block_M,
        block_K,
        in_dtype,
        force_tma_load=False,
        skip_store=skip_store,
        use_explicit_tma_copy=False,
    )
    kernel_ref = tilelang.compile(
        func_ref,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Reference kernel (no TMA) compiled OK")
    dump_and_check_kernel_source(kernel_ref, "non_tma", expect_tma_load=False)

torch.manual_seed(42)
a_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8).to(torch.int8)

# --- Run TMA kernel first and synchronize immediately so failures are attributed correctly ---
print("Launching TMA kernel...")
out_tma = kernel_tma(a_packed)
torch.cuda.synchronize()
print("TMA kernel finished.")

out_ref = None
if kernel_ref is not None:
    print("Launching reference kernel...")
    out_ref = kernel_ref(a_packed)
    torch.cuda.synchronize()
    print("Reference kernel finished.")

if skip_store:
    print("\nStore path skipped; this run only validates kernel completion after TMA load.")
    raise SystemExit(0)

# --- Compare byte-by-byte ---
tma_bytes = out_tma.view(torch.uint8).cpu()
inp_bytes = a_packed.view(torch.uint8).cpu()

match_tma_inp = (tma_bytes == inp_bytes).sum().item()
total = tma_bytes.numel()

print(f"\nTotal bytes: {total}")
print(f"TMA vs Input:  {match_tma_inp}/{total} match ({100*match_tma_inp/total:.1f}%)")
if out_ref is not None:
    ref_bytes = out_ref.view(torch.uint8).cpu()
    match_ref_inp = (ref_bytes == inp_bytes).sum().item()
    match_tma_ref = (tma_bytes == ref_bytes).sum().item()
    print(f"Ref vs Input:  {match_ref_inp}/{total} match ({100*match_ref_inp/total:.1f}%)")
    print(f"TMA vs Ref:    {match_tma_ref}/{total} match ({100*match_tma_ref/total:.1f}%)")

if match_tma_inp == total:
    print("\n[PASS] TMA round-trip is byte-exact!")
else:
    mismatch_idx = (tma_bytes != inp_bytes).nonzero(as_tuple=True)[0]
    print(f"\nFirst 20 mismatched byte indices: {mismatch_idx[:20].tolist()}")
    print(f"  TMA bytes: {tma_bytes[mismatch_idx[:20]].tolist()}")
    print(f"  Inp bytes: {inp_bytes[mismatch_idx[:20]].tolist()}")

    row_size = K // 2
    mismatch_rows = (mismatch_idx // row_size).unique()
    print(f"Mismatched rows ({len(mismatch_rows)} total): {mismatch_rows[:20].tolist()}")
