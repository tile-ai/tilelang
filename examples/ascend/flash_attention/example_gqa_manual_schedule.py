"""GQA with a validated frontend-provided task schedule.

The ``kv`` loop in ``core.py`` uses the following fixed stages in source task
order when ``manual_schedule=True``::

    qk:         [0, 1, 1, 1, 3]
    softmax:    [5]
    pack P:     [6]
    pv:         [3, 6, 6, 6, 7]
    accumulate: [7]

The presence of ``T.Stage`` selects manual scheduling for the ``kv`` loop only;
no pass config is required. The maximum stage is 7, so the example requires at
least eight KV blocks (``S2 >= 8 * 128``).
"""

import argparse

import tilelang
import torch
from tilelang.profiler import do_bench

from example_gqa import gqa, ref_program


def manual_schedule_gqa(S1=128, G=4, S2=1024, D_val=128, num_blocks=4):
    return gqa(
        S1=S1,
        G=G,
        S2=S2,
        D_val=D_val,
        num_blocks=num_blocks,
        manual_schedule=True,
    )


def check_correctness():
    S1, G, S2, D_val = 128, 4, 1024, 128
    kernel = tilelang.compile(
        manual_schedule_gqa(S1, G, S2, D_val),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )

    device = torch.device("npu")
    torch.manual_seed(0)
    Q = torch.randn(S1 * G, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    actual = kernel(Q, K, V)
    torch.npu.synchronize()
    expected = ref_program(Q, K, V, S1=S1, G=G)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    rel_diff = max_abs / expected.float().abs().max().item()
    assert torch.isfinite(actual).all()
    assert rel_diff < 0.05, f"rel_diff={rel_diff:.4f}"
    print(f"PASS: manually scheduled GQA, rel_diff={rel_diff:.6f}")


def run_regression_perf(S1=8192, G=32, S2=8192, D_val=128):
    kernel = tilelang.compile(
        manual_schedule_gqa(S1, G, S2, D_val, num_blocks=32),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    device = torch.device("npu")
    torch.manual_seed(42)
    Q = torch.randn(S1 * G, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    kernel(Q, K, V)
    torch.npu.synchronize()
    prof = do_bench(
        lambda: kernel(Q, K, V),
        backend="msprof_detail",
        _n_warmup=10,
        _n_repeat=50,
    )
    flops = 4.0 * (S1 * G) * S2 * D_val
    print(f"manual GQA: {prof.dur_us:.2f} us/iter | {prof.tflops(flops):.1f} TFLOPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    check_correctness()
    if args.benchmark:
        run_regression_perf()
