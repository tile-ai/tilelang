"""TopK-gate (MoE routing) kernel on Ascend NPU with SimdVF selection loop.

Selects the top-k largest scores per token and writes back their expert
indices, sorted descending by value with ties broken by smaller index.

Key techniques demonstrated:
  1. T.Pipelined over rows with software pipelining (NUM_STAGES=6) + UB
     double/multi-buffering via T.annotate_buffer_versions
  2. SimdVF selection loop: vmax/vcmax reduce across 4 lanes → mask compare
     to find smallest-index tie-break → vsts the winner → mask out and repeat

The inner loop is k unrolled iterations of:
  - max-reduce 4 partial max vregs → broadcast scalar max  (vmax/vcmax/vdupv)
  - per-lane (value==max) ? index : INT_MAX, then min-reduce → winner index
  - store winner index, mask its lane to -inf in registers
This is a per-token "k selection-sort steps" approach; works well when k is
small (e.g. 6-8) and the inner-axis is short enough to stay in registers.

Alignment: the SIMD loop always scans TILE_E lanes, so a prologue T.fill(-inf)
primes the whole window. When num_experts is not 8-aligned (e.g. 161), the
per-row GM→UB copy has a sub-32B tail; T.copy(..., pad_value=-inf) right-pads
that tail so it is defined too. See the copy site for details.

Performance (float32 scores, 4096 tokens, NPU):
  experts=256 topk=8:  ~22.4 us
  experts=128 topk=6:  ~17.6 us
  experts=160 topk=8:  ~22.3 us
"""

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.profiler import do_bench


def topk_gate(num_experts: int, num_topk: int, backend="asc"):
    N_CORES = 64
    TILE_E = 256  # max experts per token supported by this kernel
    NUM_STAGES = 6
    VL = 64  # SIMD vector length (float32 lanes per vreg)
    assert num_experts <= TILE_E, f"num_experts {num_experts} > TILE_E {TILE_E}"

    NUM_TOPK_8 = (num_topk + 7) // 8 * 8  # round up to multiple of 8 for UB alignment
    M = T.dynamic("num_tokens")

    @T.prim_func
    def main(
        scores: T.Tensor((M, num_experts), T.float32),
        topk_idx: T.Tensor((M, num_topk), T.int32),
    ):
        with T.Kernel(N_CORES) as core_id:
            s_ub = T.alloc_shared((TILE_E,), T.float32)
            out_ub = T.alloc_shared((NUM_TOPK_8,), T.int32)
            T.annotate_buffer_versions({s_ub: NUM_STAGES, out_ub: NUM_STAGES})

            if num_experts < TILE_E:
                with T.SimdVF():
                    T.fill(s_ub, -T.infinity(T.float32))
            for w in T.Pipelined(
                T.ceildiv(M, N_CORES),
                num_stages=NUM_STAGES,
            ):
                row = w * N_CORES + core_id
                if row < M:
                    # When num_experts * 4B is not 32B-aligned, the per-row
                    # copy has a sub-32B tail. pad_value=-inf right-pads that
                    # tail so the copy's own [num_experts..align32] lanes are
                    # defined as sentinel too (the prologue fill covers the
                    # rest of the window).
                    T.copy(
                        scores[row, :],
                        s_ub[:num_experts],
                        pad_value=-T.infinity(T.float32),
                    )

                    with T.SimdVF():
                        v = T.simd.alloc_local((4), "float32")
                        r = T.simd.alloc_local((4), "int32")

                        full = S.pset(32, "PAT_ALL")
                        one = S.pset(32, "PAT_VL1")

                        neg_inf = S.vdup(-T.infinity(T.float32), "float32", full)
                        int_max = S.vdup(T.max_value(T.int32), "int32", full)

                        for i in T.Unroll(4, explicit=True):
                            v[i] = S.vld(s_ub[i * VL])
                            r[i] = S.vci(T.int32(i * VL), "int32")

                        for k in range(num_topk):
                            # Max value across the 4 partial vregs, broadcast.
                            mx = S.vdupv(
                                S.vcmax(
                                    S.vmax(S.vmax(v[0], v[1], full), S.vmax(v[2], v[3], full), full),
                                    full,
                                ),
                                full,
                            )
                            # Per-lane: keep index where value == max, else INT_MAX.
                            idx0 = S.vsel(r[0], int_max, S.vcmp(v[0], mx, full, "eq"))
                            idx1 = S.vsel(r[1], int_max, S.vcmp(v[1], mx, full, "eq"))
                            idx2 = S.vsel(r[2], int_max, S.vcmp(v[2], mx, full, "eq"))
                            idx3 = S.vsel(r[3], int_max, S.vcmp(v[3], mx, full, "eq"))
                            # Min-reduce gives the smallest index (tie-break).
                            idx = S.vdupv(
                                S.vcmin(S.vmin(S.vmin(idx0, idx1, full), S.vmin(idx2, idx3, full), full), full),
                                full,
                            )
                            # Store winner index.
                            S.vsts(out_ub[k], idx, one, "ONEPT_B32")
                            # Mask out the winner lane in each vreg so the next k picks the next best.
                            for i in T.Unroll(4, explicit=True):
                                v[i] = S.vsel(neg_inf, v[i], S.vcmp(r[i], idx, full, "eq"))

                    T.copy(out_ub[:num_topk], topk_idx[row, :])

    return main


def ref_program(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Reference: torch.topk (descending, smaller-index tie-break)."""
    return torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True).indices.to(torch.int32)


N_ITERS = 20


def run_regression_perf(num_experts=256, num_topk=8, num_tokens=4096):
    device = torch.device("npu")
    program = topk_gate(num_experts, num_topk)
    kernel = tilelang.compile(program, out_idx=-1)

    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)

    kernel(scores)
    torch.npu.synchronize()

    def run_kernel(kernel=kernel, scores=scores):
        return kernel(scores)

    latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=N_ITERS)
    elapsed_us = latency_ms * 1e3
    total_bytes = num_tokens * (num_experts + num_topk) * 4
    bw_gbs = total_bytes / (elapsed_us * 1e-6) / 1e9
    print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s")
    return latency_ms


if __name__ == "__main__":
    device = torch.device("npu")
    num_tokens = 4096

    for num_experts, num_topk in [(256, 8), (128, 6), (160, 8), (161, 8)]:
        print(f"\n--- experts={num_experts}, topk={num_topk}, tokens={num_tokens} ---")
        program = topk_gate(num_experts, num_topk)
        kernel = tilelang.compile(program, out_idx=-1)

        scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)

        out = kernel(scores)
        torch.npu.synchronize()

        expected = ref_program(scores, num_topk)
        ok = torch.equal(out, expected)
        verdict = "PASS" if ok else "FAIL"

        def run_kernel(kernel=kernel, scores=scores):
            return kernel(scores)

        latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=N_ITERS)
        elapsed_us = latency_ms * 1e3
        total_bytes = num_tokens * (num_experts + num_topk) * 4
        bw_gbs = total_bytes / (elapsed_us * 1e-6) / 1e9
        print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s  {verdict}")
