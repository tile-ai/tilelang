"""While-loop auto-schedule example for Ascend (nested for/while/for/while).

Structure: an outer serial `for`, a `while` traversing tile groups, an inner
`T.Pipelined` loop that multi-buffers `b_ub` (MTE `T.copy` + SimdVF add + store),
and an innermost small pure-scalar `while` computing a bias. AutoSchedule cannot
pipeline a `while` directly, so `NormalizeControlFlowForSchedule` temporarily
rewrites every while into a bounded serial for (tagged `synthetic_while`),
AutoSchedule pipelines the inner loop, and `RestoreWhileLoops` turns each back
into `while(true)` with the original termination test preserved as an
`if (...) T.loop_break()` guard.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

NUM_CORES = 64
TILE_ELEMS = 8192
OUTER = 2
MID = 7
SUB = 7
BIAS_REPS = 3
NUM_STAGES = 4
N = NUM_CORES * TILE_ELEMS * OUTER * MID * SUB
NUM_REPEATS = 50


def while_pipelined():
    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        C: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(NUM_CORES) as bx:
            b_ub = T.alloc_shared((TILE_ELEMS,), T.float32)
            T.annotate_buffer_versions({b_ub: NUM_STAGES})

            for o in range(OUTER):  # outer real for
                i = T.alloc_var("int32")
                i = 0
                while i < MID:  # while #1: traverses tile groups
                    for j in T.Pipelined(SUB, num_stages=NUM_STAGES):  # multi-buffers b_ub
                        tile = ((o * MID + i) * SUB + j) * NUM_CORES + bx
                        begin = tile * TILE_ELEMS
                        end = begin + TILE_ELEMS

                        T.copy(A[begin:end], b_ub)

                        bias = T.alloc_var("float32")
                        bias = 0.0
                        s = T.alloc_var("int32")
                        s = 0
                        while s < BIAS_REPS:  # while #2: small pure-scalar loop
                            bias = bias + 1.0
                            s = s + 1

                        with T.SimtVF(threads=256):
                            for k in T.Parallel(TILE_ELEMS):
                                b_ub[k] = b_ub[k] + bias

                        T.assume_no_conflict(C[begin:end], cross=True)
                        T.copy(b_ub, C[begin:end])

                    i = i + 1

    return main


def ref_program(a):
    return a + float(BIAS_REPS)


def run(target: str = "ascend") -> float:
    device = torch.device("npu")
    compile_kwargs = {"out_idx": -1}
    kernel = tilelang.compile(while_pipelined(), **compile_kwargs)

    print(f"\n--- Generated {target} Source ---")
    print(kernel.get_kernel_source())

    a = torch.randn(N, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()

    expected = ref_program(a)
    max_diff = (out - expected).abs().max().item()
    ok = "PASS" if max_diff < 1e-2 else f"FAIL (diff={max_diff:.1e})"

    latency_ms = do_bench(lambda: kernel(a), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    elapsed_us = latency_ms * 1e3
    bytes_moved = N * 4 * 2
    bw_gbs = bytes_moved / (elapsed_us * 1e-6) / 1e9
    print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s  {ok}")
    return latency_ms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["ascend"], default="ascend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.target)


if __name__ == "__main__":
    main()
