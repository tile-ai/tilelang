"""Grouped-query FlashAttention backward example for Ascend NPU.

Q/O/dO use the flattened ``[S1 * G, D]`` layout while K/V and their
gradients use ``[S2, D]``.  The backward path consumes the optional LSE output
from ``flash_attention_fwd`` and runs a frontend-staged AutoSchedule fused
schedule with atomic dQ.
"""

import math

import tilelang

from tilelang.profiler import do_bench

from core import FwdTiling, flash_attention_fwd
from core_bwd import (
    flash_attention_bwd_fused_dq_atomic_staged,
    flash_attention_bwd_preprocess,
)


def gqa_fwd_with_lse(
    S1,
    G,
    S2,
    D=128,
    *,
    softmax_scale=None,
    num_blocks=None,
    tiling=None,
):
    return flash_attention_fwd(
        q_len=S1 * G,
        kv_len=S2,
        head_dim=D,
        out_dtype="bfloat16",
        softmax_scale=softmax_scale,
        num_blocks=num_blocks,
        tiling=tiling,
        return_lse=True,
    )


def ref_backward(Q, K, V, dO, softmax_scale=None):
    """Float32 autograd reference in the same flattened layout."""

    import torch

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(Q.shape[-1])
    q_ref = Q.float().detach().requires_grad_()
    k_ref = K.float().detach().requires_grad_()
    v_ref = V.float().detach().requires_grad_()
    scores = q_ref @ k_ref.transpose(0, 1) * scale
    output = torch.softmax(scores, dim=-1) @ v_ref
    output.backward(dO.float())
    return q_ref.grad, k_ref.grad, v_ref.grad


def run_correctness(S1=512, G=2, S2=384, D=128, *, target="ascend"):
    import torch

    q_len = S1 * G
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    fwd = tilelang.compile(
        gqa_fwd_with_lse(
            S1,
            G,
            S2,
            D,
            tiling=FwdTiling(num_stages=min(3, S2 // 128)),
        ),
        target=target,
        out_idx=[3, 4],
        pass_configs=pass_configs,
    )
    delta_kernel = tilelang.compile(
        flash_attention_bwd_preprocess(q_len, D),
        target=target,
        out_idx=[2],
        pass_configs=pass_configs,
    )
    bwd_kernel = tilelang.compile(
        flash_attention_bwd_fused_dq_atomic_staged(q_len, S2, D),
        target=target,
        out_idx=[],
        pass_configs=pass_configs,
    )

    torch.manual_seed(42)
    q = torch.randn(q_len, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(S2, D, dtype=torch.bfloat16, device="npu")
    v = torch.randn(S2, D, dtype=torch.bfloat16, device="npu")
    do = torch.randn(q_len, D, dtype=torch.bfloat16, device="npu")

    o, lse = fwd(q, k, v)
    delta = delta_kernel(o, do)
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    bwd_kernel(q, k, v, do, lse, delta, dq, dk, dv)
    torch.npu.synchronize()

    refs = ref_backward(q, k, v, do)
    for name, actual, expected in zip(("dQ", "dK", "dV"), (dq, dk, dv), refs):
        error = actual.float() - expected.float()
        rel_l2 = (error.norm() / expected.float().norm().clamp_min(1e-12)).item()
        max_abs = error.abs().max().item()
        print(f"{name}: max_abs={max_abs:.6f}, rel_l2={rel_l2:.6f}")
        rel_l2_limit = 0.011 if name == "dQ" else 0.01
        assert max_abs < 0.02 and rel_l2 < rel_l2_limit


def _profile_msprof(fn, n_warmup, n_repeat):
    return do_bench(
        fn,
        backend="msprof_detail",
        _n_warmup=n_warmup,
        _n_repeat=n_repeat,
    )


def _print_pipe_utilization(name, profile):
    """Print every pipe ratio collected by ``msprof_detail``."""

    print(f"  {name}:")
    if profile.aic_total_cycles:
        print(
            "    AIC: "
            f"MAD={profile.aic_mad:7.2%}  "
            f"Scalar={profile.aic_scalar:7.2%}  "
            f"MTE1={profile.aic_mte1:7.2%}  "
            f"MTE2={profile.aic_mte2:7.2%}  "
            f"MTE3={profile.aic_mte3:7.2%}  "
            f"FixPipe={profile.aic_fixpipe:7.2%}  "
            f"cycles={profile.aic_total_cycles}"
        )
    else:
        print("    AIC: n/a")

    if profile.aiv_total_cycles:
        print(
            "    AIV: "
            f"Vector={profile.aiv_vec:7.2%}  "
            f"Scalar={profile.aiv_scalar:7.2%}  "
            f"MTE2={profile.aiv_mte2:7.2%}  "
            f"MTE3={profile.aiv_mte3:7.2%}  "
            f"cycles={profile.aiv_total_cycles}"
        )
    else:
        print("    AIV: n/a")


def run_regression_perf(
    S1=8192,
    G=32,
    S2=8192,
    D=128,
    *,
    n_warmup=5,
    n_repeat=20,
):
    """Benchmark the fused TileLang backward kernel for perf regression."""

    import torch

    q_len = S1 * G
    softmax_scale = 1.0 / math.sqrt(D)
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    fwd = tilelang.compile(
        gqa_fwd_with_lse(
            S1,
            G,
            S2,
            D,
            softmax_scale=softmax_scale,
            tiling=FwdTiling(num_stages=min(3, S2 // 128)),
        ),
        out_idx=[],
        pass_configs=pass_configs,
    )
    delta_kernel = tilelang.compile(
        flash_attention_bwd_preprocess(q_len, D),
        out_idx=[],
        pass_configs=pass_configs,
    )
    bwd_kernel = tilelang.compile(
        flash_attention_bwd_fused_dq_atomic_staged(
            q_len,
            S2,
            D,
            softmax_scale=softmax_scale,
        ),
        out_idx=[],
        pass_configs=pass_configs,
    )

    torch.manual_seed(42)
    q = torch.randn(q_len, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(S2, D, dtype=torch.bfloat16, device="npu")
    v = torch.randn(S2, D, dtype=torch.bfloat16, device="npu")
    do = torch.randn(q_len, D, dtype=torch.bfloat16, device="npu")
    o = torch.empty_like(q)
    lse = torch.empty(q_len, dtype=torch.float32, device="npu")
    delta = torch.empty(q_len, dtype=torch.float32, device="npu")
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    fwd(q, k, v, o, lse)

    def tilelang_delta():
        delta_kernel(o, do, delta)

    def tilelang_fused():
        dq.zero_()
        bwd_kernel(q, k, v, do, lse, delta, dq, dk, dv)

    tilelang_delta()
    tilelang_fused()
    torch.npu.synchronize()
    profile = _profile_msprof(tilelang_fused, n_warmup, n_repeat)
    effective_bwd_flops = 10.0 * q_len * S2 * D
    print(
        f"GQA fused backward [S1={S1} G={G} S2={S2} D={D}]: {profile.dur_us:.2f} us/iter | {profile.tflops(effective_bwd_flops):.1f} TFLOPS"
    )
    return profile.dur_ns / 1e6


def run_performance_comparison(
    S1=8192,
    G=32,
    S2=8192,
    D=128,
    *,
    n_warmup=5,
    n_repeat=20,
):
    """Compare the optimized TileLang backward with Torch NPU SDPA backward."""

    import torch

    q_len = S1 * G
    softmax_scale = 1.0 / math.sqrt(D)
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}

    # Preallocate every output so timed regions contain only kernel work.
    fwd = tilelang.compile(
        gqa_fwd_with_lse(
            S1,
            G,
            S2,
            D,
            softmax_scale=softmax_scale,
            tiling=FwdTiling(num_stages=min(3, S2 // 128)),
        ),
        out_idx=[],
        pass_configs=pass_configs,
    )
    delta_kernel = tilelang.compile(
        flash_attention_bwd_preprocess(q_len, D),
        out_idx=[],
        pass_configs=pass_configs,
    )
    bwd_kernel = tilelang.compile(
        flash_attention_bwd_fused_dq_atomic_staged(
            q_len,
            S2,
            D,
            softmax_scale=softmax_scale,
        ),
        out_idx=[],
        pass_configs=pass_configs,
    )

    torch.manual_seed(42)
    q = torch.randn(q_len, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(S2, D, dtype=torch.bfloat16, device="npu")
    v = torch.randn(S2, D, dtype=torch.bfloat16, device="npu")
    do = torch.randn(q_len, D, dtype=torch.bfloat16, device="npu")
    o = torch.empty_like(q)
    lse = torch.empty(q_len, dtype=torch.float32, device="npu")
    delta = torch.empty(q_len, dtype=torch.float32, device="npu")
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    fwd(q, k, v, o, lse)

    def tilelang_delta():
        delta_kernel(o, do, delta)

    def tilelang_fused():
        # BF16 dQ is accumulated atomically, so clearing it is timed.
        dq.zero_()
        bwd_kernel(q, k, v, do, lse, delta, dq, dk, dv)

    tilelang_delta()
    tilelang_fused()
    torch.npu.synchronize()

    # Leaf tensors avoid layout-gradient operators in the timed Torch region.
    q_sdpa = q.view(S1, G, D).permute(1, 0, 2).unsqueeze(0).contiguous().detach().requires_grad_()
    k_sdpa = k.unsqueeze(0).unsqueeze(0).detach().requires_grad_()
    v_sdpa = v.unsqueeze(0).unsqueeze(0).detach().requires_grad_()
    do_sdpa = do.view(S1, G, D).permute(1, 0, 2).unsqueeze(0).contiguous()
    torch_o = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        scale=softmax_scale,
    )

    def torch_backward():
        return torch.autograd.grad(
            torch_o,
            (q_sdpa, k_sdpa, v_sdpa),
            do_sdpa,
            retain_graph=True,
        )

    torch_grads = torch_backward()
    torch.npu.synchronize()
    refs = (
        torch_grads[0].squeeze(0).permute(1, 0, 2).contiguous().view(q_len, D),
        torch_grads[1].squeeze(0).squeeze(0),
        torch_grads[2].squeeze(0).squeeze(0),
    )

    print(f"TileLang import: {tilelang.__file__}")
    print(f"shape: S1={S1}, G={G}, S2={S2}, D={D}, dtype=bfloat16, noncausal")
    print("backward: Delta kernel + frontend-staged AutoSchedule fused kernel")
    print(f"Torch backward op: {type(torch_o.grad_fn).__name__}")
    print(f"protocol: cold L2, warmup={n_warmup}, repeat={n_repeat}, TileLang/Torch grads=bfloat16")
    for name, actual, expected in zip(("dQ", "dK", "dV"), (dq, dk, dv), refs):
        error = actual.float() - expected.float()
        rel_l2 = (error.norm() / expected.float().norm().clamp_min(1e-12)).item()
        max_abs = error.abs().max().item()
        print(f"  correctness {name}: max_abs={max_abs:.6f}, rel_l2={rel_l2:.6f}")
        rel_l2_limit = 0.011 if name == "dQ" else 0.01
        assert max_abs < 0.02 and rel_l2 < rel_l2_limit

    delta_prof = _profile_msprof(tilelang_delta, n_warmup, n_repeat)
    fused_prof = _profile_msprof(tilelang_fused, n_warmup, n_repeat)
    torch_prof = _profile_msprof(torch_backward, n_warmup, n_repeat)

    effective_bwd_flops = 10.0 * q_len * S2 * D
    tilelang_ms = fused_prof.dur_ns / 1e6
    torch_ms = torch_prof.dur_ns / 1e6
    torch_over_tilelang = torch_ms / tilelang_ms

    print("\nFFTS kernel duration (cold L2 between iterations):")
    delta_bytes = o.numel() * o.element_size() + do.numel() * do.element_size() + delta.numel() * delta.element_size()
    print(f"  TileLang Delta:  {delta_prof.dur_us:10.2f} us | {delta_prof.gbps(delta_bytes):.2f} GB/s")
    print(f"  TileLang fused:  {fused_prof.dur_us:10.2f} us | {fused_prof.tflops(effective_bwd_flops):.1f} TFLOPS")
    print(f"  Torch SDPA bwd: {torch_prof.dur_us:10.2f} us | {torch_prof.tflops(effective_bwd_flops):.1f} TFLOPS")
    if torch_over_tilelang >= 1:
        print(f"  TileLang is {torch_over_tilelang:.3f}x faster than Torch")
    else:
        print(f"  Torch is {1.0 / torch_over_tilelang:.3f}x faster than TileLang")

    print("\nPipe utilization (active cycles / total cycles; pipes can overlap):")
    for name, profile in (
        ("TileLang Delta", delta_prof),
        ("TileLang T.Stage fused + dQ zero", fused_prof),
        ("Torch SDPA backward", torch_prof),
    ):
        _print_pipe_utilization(name, profile)

    return {
        "tilelang_delta_ms": delta_prof.dur_ns / 1e6,
        "tilelang_fused_ms": fused_prof.dur_ns / 1e6,
        "torch_ms": torch_ms,
        "tilelang_speedup": torch_over_tilelang,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action=argparse.BooleanOptionalAction, default=True, help="run the GQA backward performance comparison")
    parser.add_argument("--s1", type=int)
    parser.add_argument("--g", type=int)
    parser.add_argument("--s2", type=int)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if args.perf:
        run_performance_comparison(
            args.s1 or 8192,
            args.g or 32,
            args.s2 or 8192,
            args.d,
            n_warmup=args.warmup,
            n_repeat=args.repeat,
        )
    else:
        run_correctness(args.s1 or 512, args.g or 2, args.s2 or 384, args.d)
