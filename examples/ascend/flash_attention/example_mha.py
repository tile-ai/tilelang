"""Multi-Head Attention forward kernel on Ascend NPU using Cube + Vector cores.

Online-softmax MHA: O = softmax(Q @ K^T / sqrt(d)) @ V

The softmax writes probabilities directly in NZ layout so that the second
GEMM can consume them without an intermediate ND-to-NZ conversion.
"""

import tilelang

from tilelang.profiler import do_bench

from core import flash_attention_fwd


def mha(D_val=128, SEQ_LEN=4096):
    return flash_attention_fwd(
        q_len=SEQ_LEN,
        kv_len=SEQ_LEN,
        head_dim=D_val,
        out_dtype="float32",
    )


def ref_program(Q, K, V):
    import torch

    return torch.nn.functional.scaled_dot_product_attention(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)).squeeze(0).to(torch.float32)


def run_regression_perf(D_val=128, SEQ_LEN=4096):
    """Compile + benchmark the FlashAttention kernel, returning latency in ms."""
    import torch

    device = torch.device("npu")
    program = mha(D_val, SEQ_LEN)
    kernel = tilelang.compile(program, out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})

    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    kernel(Q, K, V)
    torch.npu.synchronize()

    prof = do_bench(lambda: kernel(Q, K, V), backend="msprof_detail", _n_warmup=30, _n_repeat=500)
    flops = 4.0 * SEQ_LEN * SEQ_LEN * D_val
    print(f"    [D={D_val} SEQ_LEN={SEQ_LEN}] {prof.dur_us:.2f} us/iter  |  {prof.tflops(flops):.1f} TFLOPS")
    return prof.dur_ns / 1e6


def bench_torch_mha(D_val=128, SEQ_LEN=4096):
    """Benchmark PyTorch SDPA (MHA path) for comparison."""
    import torch
    import math

    softmax_scale = 1.0 / math.sqrt(D_val)
    device = torch.device("npu")
    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    # Pre-reshape to SDPA format so the timed lambda only contains the op
    q_sdpa = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, SEQ_LEN, D)
    k_sdpa = K.unsqueeze(0).unsqueeze(0)  # (1, 1, SEQ_LEN, D)
    v_sdpa = V.unsqueeze(0).unsqueeze(0)  # (1, 1, SEQ_LEN, D)

    torch.npu.synchronize()
    prof = do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=softmax_scale),
        backend="msprof_detail",
        _n_warmup=30,
        _n_repeat=500,
    )
    flops = 4.0 * SEQ_LEN * SEQ_LEN * D_val
    print(f"  Torch SDPA: {prof.dur_us:.2f} us/iter | {prof.tflops(flops):.1f} TFLOPS")
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    import torch

    D_val = 128
    SEQ_LEN = 4096

    device = torch.device("npu")
    print(f"Compiling mha (D={D_val}, SEQ_LEN={SEQ_LEN})...")
    program = mha(D_val, SEQ_LEN)
    kernel = tilelang.compile(program, out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    print("Compilation succeeded!")

    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    K = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)
    V = torch.randn(SEQ_LEN, D_val, dtype=torch.bfloat16, device=device)

    O = kernel(Q, K, V)
    torch.npu.synchronize()

    expected = ref_program(Q, K, V)
    diff = (O.cpu().float() - expected.cpu()).abs().max().item()
    rel = diff / expected.cpu().abs().max().item()
    print(f"  rel={rel:.4f} {'PASS' if rel < 0.05 else 'FAIL'}")

    prof = do_bench(lambda: kernel(Q, K, V), backend="msprof_detail", _n_warmup=30, _n_repeat=500)
    flops = 4.0 * SEQ_LEN * SEQ_LEN * D_val
    print(f"  {prof.dur_us:6.1f} us  |  {prof.tflops(flops):5.1f} TFLOPS")

    print("\n--- Performance comparison ---")
    tilelang_ms = prof.dur_ns / 1e6
    torch_ms = bench_torch_mha(D_val, SEQ_LEN)
    ratio = torch_ms / tilelang_ms
    print(
        f"\n  TileLang: {tilelang_ms:.2f} ms | Torch: {torch_ms:.2f} ms | Speedup: {ratio:.2f}x"
        if ratio >= 1
        else f"\n  TileLang: {tilelang_ms:.2f} ms | Torch: {torch_ms:.2f} ms | Torch is {1 / ratio:.2f}x faster"
    )
