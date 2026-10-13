"""Grouped-query attention.

Q and O use the flattened ``[S1 * G, D]`` layout. K and V are shared by all
query groups and use ``[S2, D]``.
"""

import math

import tilelang

from tilelang.profiler import do_bench

from core import flash_attention_fwd


def gqa(
    S1=8192,
    G=32,
    S2=8192,
    D_val=128,
    softmax_scale=None,
    out_dtype="bfloat16",
    num_blocks=32,
    manual_schedule=False,
):
    return flash_attention_fwd(
        q_len=S1 * G,
        kv_len=S2,
        head_dim=D_val,
        out_dtype=out_dtype,
        softmax_scale=softmax_scale,
        num_blocks=num_blocks,
        manual_schedule=manual_schedule,
    )


def ref_program(Q_flat, K, V, S1=8192, G=32, softmax_scale=None):
    import torch

    D_ref = Q_flat.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_ref)
    q = Q_flat.view(S1, G, D_ref).permute(1, 0, 2).unsqueeze(0)
    k = K.unsqueeze(0).unsqueeze(0)
    v = V.unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
    return out.squeeze(0).permute(1, 0, 2).contiguous().view(S1 * G, D_ref)


def run_regression_perf(S1=8192, G=32, S2=8192, D_val=128):
    import torch

    device = torch.device("npu")
    kernel = tilelang.compile(gqa(S1, G, S2, D_val), out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
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
    print(f"    [S1={S1} G={G} S2={S2} D={D_val}] {prof.dur_us:.2f} us/iter | {prof.tflops(flops):.1f} TFLOPS")
    return prof.dur_ns / 1e6


def bench_torch_gqa(S1=8192, G=32, S2=8192, D_val=128):
    """Benchmark PyTorch SDPA (GQA path) for comparison."""
    import torch

    softmax_scale = 1.0 / math.sqrt(D_val)
    device = torch.device("npu")
    torch.manual_seed(42)
    Q_flat = torch.randn(S1 * G, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    # Pre-reshape to SDPA format so the timed lambda only contains the op
    q_sdpa = Q_flat.view(S1, G, D_val).permute(1, 0, 2).unsqueeze(0).contiguous()  # (1, G, S1, D)
    k_sdpa = K.unsqueeze(0).unsqueeze(0)  # (1, 1, S2, D) — broadcast by SDPA
    v_sdpa = V.unsqueeze(0).unsqueeze(0)  # (1, 1, S2, D)

    torch.npu.synchronize()
    prof = do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=softmax_scale),
        backend="msprof_detail",
        _n_warmup=10,
        _n_repeat=50,
    )
    flops = 4.0 * (S1 * G) * S2 * D_val
    print(f"  Torch SDPA: {prof.dur_us:.2f} us/iter | {prof.tflops(flops):.1f} TFLOPS")
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    import torch

    S1, G, S2, D_val = 8192, 32, 8192, 128
    device = torch.device("npu")
    print("Compiling GQA attention kernel...")
    kernel = tilelang.compile(gqa(S1, G, S2, D_val), out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    print("Compilation succeeded!")
    torch.manual_seed(42)
    Q = torch.randn(S1 * G, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(S2, D_val, dtype=torch.bfloat16, device=device)
    O = kernel(Q, K, V)
    torch.npu.synchronize()
    expected = ref_program(Q, K, V, S1=S1, G=G)
    diff = (O.cpu().float() - expected.cpu().float()).abs().max().item()
    rel = diff / expected.cpu().float().abs().max().item()
    print(f"  rel={rel:.4f} {'PASS' if rel < 0.05 else 'FAIL'}")

    print("\n--- Performance comparison ---")
    tilelang_ms = run_regression_perf(S1, G, S2, D_val)
    torch_ms = bench_torch_gqa(S1, G, S2, D_val)
    ratio = torch_ms / tilelang_ms
    print(
        f"\n  TileLang: {tilelang_ms:.2f} ms | Torch: {torch_ms:.2f} ms | Speedup: {ratio:.2f}x"
        if ratio >= 1
        else f"\n  TileLang: {tilelang_ms:.2f} ms | Torch: {torch_ms:.2f} ms | Torch is {1 / ratio:.2f}x faster"
    )
