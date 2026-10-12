"""L0C fp32 -> UB f16/bf16 ``dual_copy`` on-path cast.

Dual + ``quant_pre`` is illegal, so lowering emits two non-dual FixPipes.
Each AIV gets a half-tile already in the destination dtype.

Run on an Ascend NPU; checks torch accuracy and benches M- and N-split.
"""

import torch

import tilelang
from tilelang.ascend import language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, dtype="bfloat16", dst_dtype="float16", split="M"):
    """GEMM with L0C fp32 -> UB f16/bf16 FixPipe quant.

    Args:
        dtype: input A/B dtype ("bfloat16" or "float16").
        dst_dtype: output/UB dtype of the on-path cast ("float16" or "bfloat16").
        split: "M" (rows split across AIVs) or "N" (columns split across AIVs).
    """
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    if M_DIM <= 0 or N_DIM <= 0 or K_DIM <= 0 or M_DIM % TILE_M or N_DIM % TILE_N or K_DIM % TILE_K:
        raise ValueError("M_DIM, N_DIM and K_DIM must be positive multiples of 256")
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    half_m = TILE_M // 2
    half_n = TILE_N // 2
    dst_shape = (half_m, TILE_N) if split == "M" else (TILE_M, half_n)

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), dtype),
        W: T.Buffer((N_DIM, K_DIM), dtype),
        C: T.Buffer((M_DIM, N_DIM), dst_dtype),
    ):
        with T.MixedKernel(NUM_BLOCKS) as (bx, _):
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            temp = T.alloc_shared(dst_shape, dst_dtype)

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))

                # fp32 L0C -> f16/bf16 UB, fanned out across the two AIVs, then
                # each AIV stores its half back into the matching half of C.
                T.dual_copy(res, temp)
                T.dual_copy(
                    temp,
                    C[
                        m_tile * TILE_M : (m_tile + 1) * TILE_M,
                        n_tile * TILE_N : (n_tile + 1) * TILE_N,
                    ],
                )

    return main


def gemm_subregion(dtype="float16", dst_dtype="float16", split="M"):
    """Compact GEMM into ``acc[:, :n]`` on a wider L0C, then on-path dual_copy.

    M-split pipe-1 offset uses the L0C row pitch (64), not the copied N (32).
    """
    TILE_M = TILE_N = TILE_K = 64
    REGION_M, REGION_N, REGION_K = 64, 32, 32
    dst_shape = (REGION_M // 2, REGION_N) if split == "M" else (REGION_M, REGION_N // 2)

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), dtype),
        B: T.Buffer((TILE_N, TILE_K), dtype),
        C: T.Buffer((REGION_M, REGION_N), dst_dtype),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            b_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            a_l0 = T.alloc_l0a((TILE_M, TILE_K), dtype)
            b_l0 = T.alloc_l0b((TILE_N, TILE_K), dtype)
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared(dst_shape, dst_dtype)

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1[0:REGION_M, 0:REGION_K], a_l0[0:REGION_M, 0:REGION_K])
            T.copy(b_l1[0:REGION_N, 0:REGION_K], b_l0[0:REGION_N, 0:REGION_K])
            T.gemm(
                a_l0[0:REGION_M, 0:REGION_K],
                b_l0[0:REGION_N, 0:REGION_K],
                acc[0:REGION_M, 0:REGION_N],
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(acc[0:REGION_M, 0:REGION_N], tmp)
            T.dual_copy(tmp, C)

    return main


def gemm_dynamic_tail(dtype="float16", dst_dtype="float16", split="M"):
    """Full 256x256 GEMM; on-path dual_copy only a dynamic prefix of L0C.

    UB is the full half-tile. The copied extent is
    ``min(sizes[0], tile_half / unit) * unit`` so rewrite sees a symbolic
    region and does not demand a static 2:1. Host passes 192.
    """
    TILE_M = TILE_N = TILE_K = 256
    dst_shape = (TILE_M // 2, TILE_N) if split == "M" else (TILE_M, TILE_N // 2)

    if split == "M":

        @T.prim_func
        def main(
            A: T.Buffer((TILE_M, TILE_K), dtype),
            B: T.Buffer((TILE_N, TILE_K), dtype),
            sizes: T.Buffer((1,), "int32"),
            C: T.Buffer((TILE_M, TILE_N), dst_dtype),
        ):
            with T.MixedKernel(1) as (_bx, _sid):
                a_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
                b_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
                acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
                tmp = T.alloc_shared(dst_shape, dst_dtype)

                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
                T.dual_copy(
                    acc[0 : T.max(T.min(T.int32(sizes[0]), TILE_M // 2), 0) * 2, 0:TILE_N],
                    tmp,
                )
                T.dual_copy(tmp, C)

        return main

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), dtype),
        B: T.Buffer((TILE_N, TILE_K), dtype),
        sizes: T.Buffer((1,), "int32"),
        C: T.Buffer((TILE_M, TILE_N), dst_dtype),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            b_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared(dst_shape, dst_dtype)

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(
                acc[0:TILE_M, 0 : T.max(T.min(T.int32(sizes[0]), TILE_N // 32), 0) * 32],
                tmp,
            )
            T.dual_copy(tmp, C)

    return main


def gemm_compact_full_n_copy():
    """64x64 L0C, MAD writes ``acc[0:32, 0:64]``, then a plain fp32 copy.

    A non-dual FixPipe only writes sub-block 0's UB, so only that AIV stores.
    """
    m, n, k = 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.MixedKernel(1) as (_bx, sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 64), "float32")
            ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            # Non-dual FixPipe writes only subblock 0. The other AIV's UB is
            # empty; letting it store the same GM range clobbers the result.
            T.copy(acc[0:m, 0:n], ub)
            if sid == 0:
                T.copy(ub, C)

    return main


def gemm_compact_full_n_dual(dst_dtype="float16"):
    """Same compact MAD, then a dual_copy into a half-row UB."""
    m, n, k = 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), dst_dtype),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 64), "float32")
            ub = T.alloc_shared((m // 2, n), dst_dtype)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    return main


def gemm_compact_n_split(dst_dtype="float16"):
    """64x128 L0C, compact 32x64 MAD, N-split into two 32x32 UB halves."""
    m, n, k = 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), dst_dtype),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 128), "float32")
            ub = T.alloc_shared((m, n // 2), dst_dtype)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    return main


def gemm_unit_flag_reuse():
    """Two GEMMs reuse one 32x64 L0C with ``unit_flag_ctrl=3``."""
    m, n, k = 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((2, m, k), "float16"),
        B: T.Buffer((2, n, k), "float16"),
        C: T.Buffer((2 * m, n), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            acc = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((m, n // 2), "float16")
            for step in T.serial(2):
                T.copy(A[step, :, :], a_l1)
                T.copy(B[step, :, :], b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
                T.dual_copy(acc, ub, unit_flag_ctrl=3)
                T.dual_copy(ub, C[step * m : (step + 1) * m, 0:n])

    return main


def gemm_destination_window(row_start, row_end):
    """Full 64x64 GEMM; dual_copy into a 32-row window of a 64x64 UB."""
    m, n, k = 64, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            acc = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((m, n), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, ub[row_start:row_end, 0:n])
            T.dual_copy(ub[row_start:row_end, 0:n], C)

    return main


def gemm_region_prefix():
    """256x256 GEMM; dual_copy only ``acc[0:128, 0:128]``. Pitch stays 256."""
    alloc, region = 256, 128

    @T.prim_func
    def main(
        A: T.Buffer((alloc, alloc), "float16"),
        B: T.Buffer((alloc, alloc), "float16"),
        C: T.Buffer((region, region), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((alloc, alloc), "float16")
            b_l1 = T.alloc_l1((alloc, alloc), "float16")
            acc = T.alloc_l0c((alloc, alloc), "float32")
            ub = T.alloc_shared((region // 2, region), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:region, 0:region], ub)
            T.dual_copy(ub, C)

    return main


def ref_program(x, w, dst_dtype):
    out = x.float() @ w.float().T
    return out.to(getattr(torch, dst_dtype))


def check_onpath_accuracy(program, x, w, expected, thresh, label):
    """Compile and run ``program``; raise if relative L-inf error exceeds ``thresh``."""
    kernel = tilelang.compile(program, out_idx=-1)
    c = kernel(x, w)
    torch.npu.synchronize()
    return _assert_onpath_rel(c, expected, thresh, label)


def _assert_onpath_rel(c, expected, thresh, label):
    diff = (c.float() - expected.float()).abs()
    if not torch.isfinite(c).all() or not torch.isfinite(expected).all():
        raise AssertionError(
            f"{label} produced non-finite values: "
            f"out_finite={torch.isfinite(c).all().item()} "
            f"ref_finite={torch.isfinite(expected).all().item()} "
            f"out_max={c.float().abs().max().item() if c.numel() else 'empty'}"
        )
    denom = expected.float().abs().max().clamp_min(1e-8)
    rel = diff.max() / denom
    print(f"  {label}: {'PASS' if rel < thresh else 'FAIL'}  rel_max={rel:.2e}")
    if not rel < thresh:
        raise AssertionError(f"{label} mismatch! rel_max={rel:.2e}")
    return rel


def run_subregion(dtype="float16", dst_dtype="float16", split="M", thresh=2e-2):
    device = torch.device("npu")
    x = torch.randn(64, 64, dtype=getattr(torch, dtype), device=device) * 0.25
    w = torch.randn(64, 64, dtype=getattr(torch, dtype), device=device) * 0.25
    expected = ref_program(
        x[:, :32].contiguous(),
        w[:32, :32].contiguous(),
        dst_dtype,
    )
    return check_onpath_accuracy(
        gemm_subregion(dtype=dtype, dst_dtype=dst_dtype, split=split),
        x,
        w,
        expected,
        thresh,
        f"column-slice {split}-split fp32->{dst_dtype}",
    )


def run_dynamic_tail(dtype="float16", dst_dtype="float16", split="M", thresh=2e-2):
    TILE, TAIL = 256, 192
    device = torch.device("npu")
    x = torch.randn(TILE, TILE, dtype=getattr(torch, dtype), device=device) * 0.25
    w = torch.randn(TILE, TILE, dtype=getattr(torch, dtype), device=device) * 0.25
    sizes = torch.tensor(
        [TAIL // 2 if split == "M" else TAIL // 32],
        dtype=torch.int32,
        device=device,
    )
    kernel = tilelang.compile(
        gemm_dynamic_tail(dtype=dtype, dst_dtype=dst_dtype, split=split),
        out_idx=-1,
    )
    c = kernel(x, w, sizes)
    torch.npu.synchronize()
    expected = ref_program(x, w, dst_dtype)
    if split == "M":
        c, expected = c[:TAIL], expected[:TAIL]
    else:
        c, expected = c[:, :TAIL], expected[:, :TAIL]
    return _assert_onpath_rel(
        c,
        expected,
        thresh,
        f"dynamic-tail {TAIL} {split}-split fp32->{dst_dtype}",
    )


def _randn(*shape, dtype):
    return torch.randn(*shape, dtype=dtype, device=torch.device("npu")) * 0.25


def run_compact_full_n_copy_and_dual():
    """Compact 32x64 MAD inside a 64x64 L0C, plain fp32 copy + fp32/fp16 dual_copy."""
    m, n, k = 32, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    ref_fp32 = a.float() @ b.float().T
    ref_fp16 = ref_fp32.to(torch.float16)
    check_onpath_accuracy(gemm_compact_full_n_copy(), a, b, ref_fp32, 1e-3, "compact fp32 copy")
    check_onpath_accuracy(gemm_compact_full_n_dual("float32"), a, b, ref_fp32, 1e-3, "compact fp32 dual_copy")
    check_onpath_accuracy(gemm_compact_full_n_dual("float16"), a, b, ref_fp16, 2e-2, "compact fp16 dual_copy")


def run_compact_n_split(dst_dtype="float16", thresh=2e-2):
    """Compact 32x64 MAD in a 64x128 L0C, N-split into two 32x32 UB halves."""
    m, n, k = 32, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(getattr(torch, dst_dtype))
    return check_onpath_accuracy(
        gemm_compact_n_split(dst_dtype),
        a,
        b,
        expected,
        thresh,
        f"compact n-split {dst_dtype}",
    )


def run_unit_flag_reuse(thresh=2e-2):
    """Two GEMMs reuse one 32x64 L0C with ``unit_flag_ctrl=3`` then dual_copy."""
    m, n, k = 32, 64, 32
    a = _randn(2, m, k, dtype=torch.float16)
    b = _randn(2, n, k, dtype=torch.float16)
    expected = torch.cat(
        [(a[i].float() @ b[i].float().T).to(torch.float16) for i in range(2)],
        dim=0,
    )
    return check_onpath_accuracy(gemm_unit_flag_reuse(), a, b, expected, thresh, "unit-flag reuse")


def run_destination_window(row_start=0, row_end=32, thresh=2e-2):
    """Full 64x64 GEMM; dual_copy into a window of a 64x64 UB allocation."""
    m, n, k = 64, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(torch.float16)
    return check_onpath_accuracy(
        gemm_destination_window(row_start, row_end),
        a,
        b,
        expected,
        thresh,
        f"dst window [{row_start}:{row_end}]",
    )


def run_region_prefix(thresh=2e-2):
    """256x256 GEMM; dual_copy only ``acc[0:128, 0:128]`` (pitch stays 256)."""
    alloc, region = 256, 128
    a = _randn(alloc, alloc, dtype=torch.float16)
    b = _randn(alloc, alloc, dtype=torch.float16)
    expected = (a[:region].float() @ b[:region].float().T).to(torch.float16)
    return check_onpath_accuracy(gemm_region_prefix(), a, b, expected, thresh, "prefix 128 of 256")


def run_regression_perf(M=8192, K=8192, N=8192, dtype="bfloat16", dst_dtype="float16", split="M"):
    """Compile + benchmark a single on-path-cast GEMM; returns latency in ms."""
    device = torch.device("npu")
    program = gemm(M, K, N, dtype=dtype, dst_dtype=dst_dtype, split=split)
    kernel = tilelang.compile(program, out_idx=-1)

    x = torch.randn(M, K, dtype=getattr(torch, dtype), device=device)
    w = torch.randn(N, K, dtype=getattr(torch, dtype), device=device)

    kernel(x, w)
    torch.npu.synchronize()

    NUM_REPEATS = 100 if dtype == "bfloat16" else 50

    def run_kernel(kernel=kernel, x=x, w=w):
        return kernel(x, w)

    latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M * N * K
    print(f"  {latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
    return latency_ms


if __name__ == "__main__":
    device = torch.device("npu")
    M, K, N = 8192, 8192, 8192
    dtype = "bfloat16"

    for dst_dtype, thresh in [("float16", 2e-2), ("bfloat16", 5e-2)]:
        for split in ["M", "N"]:
            print(f"\n=== compact column-slice acc[:, :32] fp32 -> {dst_dtype}, {split}-split ===")
            run_subregion(dtype="float16", dst_dtype=dst_dtype, split=split, thresh=thresh)

    for dst_dtype, thresh in [("float16", 2e-2), ("bfloat16", 5e-2)]:
        for split in ["M", "N"]:
            print(f"\n=== dynamic tail 192/256 fp32 -> {dst_dtype}, {split}-split ===")
            run_dynamic_tail(dtype="float16", dst_dtype=dst_dtype, split=split, thresh=thresh)

    print("\n=== compact 32x64 MAD inside a 64x64 L0C (plain copy + dual_copy) ===")
    run_compact_full_n_copy_and_dual()

    for dst_dtype, thresh in [("float16", 2e-2), ("bfloat16", 5e-2)]:
        print(f"\n=== compact N-split 32x64 fp32 -> {dst_dtype} ===")
        run_compact_n_split(dst_dtype, thresh)

    print("\n=== N-split UnitFlag reuses one L0C ===")
    run_unit_flag_reuse()

    print("\n=== destination window of a 64x64 UB ===")
    run_destination_window(0, 32)
    run_destination_window(16, 48)

    print("\n=== 128 prefix of a 256 GEMM ===")
    run_region_prefix()

    for dst_dtype, thresh in [("float16", 2e-2), ("bfloat16", 5e-2)]:
        for split in ["M", "N"]:
            print(f"\n=== fp32 -> {dst_dtype} on-path cast, {split}-split ===")
            program = gemm(M, K, N, dtype=dtype, dst_dtype=dst_dtype, split=split)
            kernel = tilelang.compile(program, out_idx=-1)
            print("Compilation succeeded!")

            x = torch.randn(M, K, dtype=getattr(torch, dtype), device=device)
            w = torch.randn(N, K, dtype=getattr(torch, dtype), device=device)

            c = kernel(x, w)
            torch.npu.synchronize()

            expected = ref_program(x, w, dst_dtype)
            # Relative L-inf error, robust against the ~sqrt(K) output magnitude.
            rel = (c.float() - expected.float()).abs().max() / expected.float().abs().max()
            print(f"  {'PASS' if rel < thresh else 'FAIL'}  rel_max={rel:.2e}")
            if not rel < thresh:
                raise AssertionError(f"Results mismatch! rel_max={rel:.2e}")

            run_regression_perf(M, K, N, dtype=dtype, dst_dtype=dst_dtype, split=split)
