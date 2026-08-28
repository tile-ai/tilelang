"""Compare on-path FixPipe quant vs VF cast under fused GEMM + bias add.

Fused ``Y = X @ W.T + bias``: on-path FixPipe quant vs SimdVF ``vcvt``.

Both paths M-split so each AIV can ``vadd`` bias. They differ only in the
L0C fp32 -> UB f16/bf16 cast:

- ``onpath``: two non-dual FixPipes with ``quant_pre`` (dual + quant is illegal).
- ``vf_cast``: one hardware dual (fp32) plus SimdVF ``vcvt`` / ``PK_B32``.

Ref: ``(X @ W.T).to(dst_dtype) + bias``.  Run with plain ``python``; do not
wrap with the ``msprof`` CLI (nested profilers only capture cache-flush).
"""

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

EPILOGUES = ("onpath", "vf_cast")


def gemm(
    M_DIM=8192,
    K_DIM=256,
    N_DIM=8192,
    dtype="bfloat16",
    dst_dtype="float16",
    epilogue="onpath",
):
    """Mixed GEMM + bias add; epilogue is on-path FixPipe quant or SimdVF vcvt.

    Args:
        dtype: input A/B dtype ("bfloat16" or "float16").
        dst_dtype: output / bias dtype after the cast ("float16" or "bfloat16").
        epilogue: "onpath" (two FixPipes + quant_pre) or "vf_cast"
            (one hardware dual + SimdVF vcvt).
    """
    if epilogue not in EPILOGUES:
        raise ValueError(f"epilogue must be one of {EPILOGUES}, got {epilogue!r}")

    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    NUM_STAGES = min(2, K_TILES)
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    half_m = TILE_M // 2
    dst_shape = (half_m, TILE_N)

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

    @T.macro
    def vf_bias_add(buf, bias):
        with T.SimdVF():
            for i, j in T.Parallel(half_m, TILE_N):
                buf[i, j] = buf[i, j] + bias[j]

    @T.macro
    def vf_cast(src, dst):
        with T.SimdVF():
            for i in range(half_m):
                for j in range(0, TILE_N, 64):
                    x = T.simd.vld(src[i, j])
                    y = T.simd.vcvt(x, dst_dtype)
                    T.simd.vsts(dst[i, j], y, dist="PK_B32")

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), dtype),
        W: T.Buffer((N_DIM, K_DIM), dtype),
        Bias: T.Buffer((N_DIM,), dst_dtype),
        C: T.Buffer((M_DIM, N_DIM), dst_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            temp = T.alloc_shared(dst_shape, dst_dtype)
            bias_ub = T.alloc_shared((TILE_N,), dst_dtype)
            if epilogue == "vf_cast":
                temp_f32 = T.alloc_shared(dst_shape, "float32")

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))

                T.copy(Bias[n_tile * TILE_N : (n_tile + 1) * TILE_N], bias_ub)
                if epilogue == "onpath":
                    # Two non-dual FixPipes with quant_pre; temp is already dst_dtype.
                    T.dual_copy(res, temp)
                else:
                    # One hardware dual (fp32 -> fp32), then SimdVF vcvt into temp.
                    T.dual_copy(res, temp_f32)
                    vf_cast(temp_f32, temp)

                vf_bias_add(temp, bias_ub)
                T.dual_copy(
                    temp,
                    C[
                        m_tile * TILE_M : (m_tile + 1) * TILE_M,
                        n_tile * TILE_N : (n_tile + 1) * TILE_N,
                    ],
                )

    return main


def ref_program(x, w, bias, dst_dtype):
    """Bias add after the GEMM is in ``dst_dtype``, matching both kernels."""
    out = x.float() @ w.float().T
    return out.to(getattr(torch, dst_dtype)) + bias


def _label(epilogue, dst_dtype):
    if epilogue == "onpath":
        how = "two FixPipes + on-path quant"
    else:
        how = "one hardware dual + SimdVF vcvt"
    return f"{how} + VF bias add  (M-split, dst={dst_dtype})"


def run_one(M, K, N, dtype, dst_dtype, epilogue, *, bench=True, thresh=None):
    """Compile, check against torch, optionally bench.  Returns latency_ms or None."""
    device = torch.device("npu")
    program = gemm(M, K, N, dtype=dtype, dst_dtype=dst_dtype, epilogue=epilogue)
    kernel = tilelang.compile(program, out_idx=-1)
    print(f"  compiled {_label(epilogue, dst_dtype)}")

    x = torch.randn(M, K, dtype=getattr(torch, dtype), device=device)
    w = torch.randn(N, K, dtype=getattr(torch, dtype), device=device)
    bias = torch.randn(N, dtype=getattr(torch, dst_dtype), device=device)
    c = kernel(x, w, bias)
    torch.npu.synchronize()

    expected = ref_program(x, w, bias, dst_dtype)
    rel = (c.float() - expected.float()).abs().max() / expected.float().abs().max()
    if thresh is None:
        thresh = 2e-2 if dst_dtype == "float16" else 5e-2
    print(f"  {'PASS' if rel < thresh else 'FAIL'}  rel_max={rel:.2e}")
    if not rel < thresh:
        raise AssertionError(f"{epilogue} mismatch: rel_max={rel:.2e}")

    if not bench:
        return None

    NUM_REPEATS = 100 if dtype == "bfloat16" else 50

    def run_kernel(kernel=kernel, x=x, w=w, bias=bias):
        return kernel(x, w, bias)

    latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M * N * K
    print(f"  {latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS  |  shape {M}x{K}x{N}")
    return latency_ms


def compare_epilogues(M=8192, K=256, N=8192, dtype="bfloat16", dst_dtype="float16"):
    """Bench both epilogues on the same shape and print a side-by-side table."""
    print(f"\n=== on-path FixPipe quant vs SimdVF vcvt, M-split GEMM + bias, shape {M}x{K}x{N}, dst={dst_dtype} ===")
    onpath_ms = run_one(M, K, N, dtype, dst_dtype, "onpath")
    vf_ms = run_one(M, K, N, dtype, dst_dtype, "vf_cast")
    delta = vf_ms - onpath_ms
    ratio = vf_ms / onpath_ms
    faster = "onpath" if onpath_ms < vf_ms else "vf_cast"
    print(f"  delta (vf_cast - onpath) = {delta:+.3f} ms  |  vf_cast/onpath = {ratio:.3f}x  |  faster: {faster}")
    return onpath_ms, vf_ms


if __name__ == "__main__":
    for dst_dtype in ("float16", "bfloat16"):
        compare_epilogues(8192, 256, 8192, dtype="bfloat16", dst_dtype=dst_dtype)
