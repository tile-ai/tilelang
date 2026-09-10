"""RMSNorm example with a fragment shared by two SIMT_VF regions.

The weight fragment is allocated in the kernel scope, initialized once by the
first ``SimtVF`` region, and read by the per-token ``SimtVF`` region.  This is
the frontend shape used to test PTO's persistent-fragment lowering; it does
not add a backend-specific API.

``rms_norm_persistent_fwd`` stages the weight through UB, while
``rms_norm_persistent_gm_to_fragment_fwd`` loads it directly from GM in the
initialization region.
"""

import argparse

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.profiler import do_bench


def rms_norm_persistent_fwd(batch: int, d: int, dtype: str = "float32"):
    """Build an RMSNorm kernel with an init and a carry SIMT_VF section."""

    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)

    # Adaptive threads: keep ~32 elements per thread (8 float4)
    threads = 256 if TILE > 4096 else 128

    N = batch * d

    @T.prim_func
    def main(
        X: T.Buffer((N,), dtype),
        Y: T.Buffer((N,), dtype),
        W: T.Buffer((d,), dtype),
        RSTD: T.Buffer((batch,), "float32"),
        eps: T.float32,
    ):
        n_tokens_per_core = batch // N_CORES

        with T.Kernel(N_CORES) as core_id:
            # Double-buffered shared buffers
            w_ub = T.alloc_shared((d,), dtype)
            x_ub = T.alloc_shared((TILE), "float32")
            y_ub = T.alloc_shared((TILE), "float32")
            z_rstd_ub = T.alloc_shared((8), "float32")

            # Defined outside both regions: this is the candidate persistent
            # fragment.  Layout inference maps its logical elements to lanes.
            w_frag = T.alloc_fragment((TILE), "float32")

            # Load weights once
            T.copy(W[:d], w_ub[:d])

            # Init section: each lane loads its weight slice once.
            with T.SimtVF(threads=threads):
                for i in T.Parallel(TILE):
                    if i < d:
                        w_frag[i] = w_ub[i]

            # Carry section: w_frag is read for every token, but is not loaded
            # from UB again.
            for t in T.Pipelined(n_tokens_per_core, num_stages=2):
                base = (t * N_CORES + core_id) * d

                T.copy(X[base : base + d], x_ub[:d])

                with T.SimtVF(threads=threads):
                    # Fragment: vectorized float4 load from UB to registers
                    x_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE):
                        x_frag[i] = x_ub[i]

                    # Reduce: sum(x^2) from registers (no UB load)
                    sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
                    T.clear(sum_sq)
                    for i in T.Parallel(TILE):
                        if i < d:
                            sum_sq[0] += x_frag[i] * x_frag[i]
                    T.finalize_reducer(sum_sq)

                    # Compute rstd
                    var = sum_sq[0] / d + eps
                    rstd_val = T.rsqrt(var)
                    z_rstd_ub[0] = rstd_val

                    # Output: y = x * rstd * w (reuse x_frag, no x reload)
                    for i in T.Parallel(TILE):
                        if i < d:
                            y_ub[i] = x_frag[i] * rstd_val * w_frag[i]

                # MTE3: store y[token] and rstd from UB to GM
                row_id = t * N_CORES + core_id
                T.copy(y_ub[:d], Y[base : base + d])
                T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])

    return main


def rms_norm_persistent_gm_to_fragment_fwd(batch: int, d: int, dtype: str = "float32"):
    """Build RMSNorm with the persistent weight loaded directly from GM."""

    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)

    # Adaptive threads: keep ~32 elements per thread (8 float4)
    threads = 256 if TILE > 4096 else 128

    N = batch * d

    @T.prim_func
    def main(
        X: T.Buffer((N,), dtype),
        Y: T.Buffer((N,), dtype),
        W: T.Buffer((d,), dtype),
        RSTD: T.Buffer((batch,), "float32"),
        eps: T.float32,
    ):
        n_tokens_per_core = batch // N_CORES

        with T.Kernel(N_CORES) as core_id:
            # Double-buffered shared buffers
            x_ub = T.alloc_shared((TILE), "float32")
            y_ub = T.alloc_shared((TILE), "float32")
            z_rstd_ub = T.alloc_shared((8), "float32")

            w_frag = T.alloc_fragment((TILE), "float32")

            # Init section: each lane loads its persistent weight slice
            # directly from GM, without an intermediate UB allocation.
            with T.SimtVF(threads=threads):
                for i in T.Parallel(TILE):
                    if i < d:
                        w_frag[i] = W[i]

            for t in T.Pipelined(n_tokens_per_core, num_stages=2):
                base = (t * N_CORES + core_id) * d

                T.copy(X[base : base + d], x_ub[:d])

                with T.SimtVF(threads=threads):
                    # Fragment: vectorized float4 load from UB to registers
                    x_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE):
                        x_frag[i] = x_ub[i]

                    # Reduce: sum(x^2) from registers (no UB load)
                    sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
                    T.clear(sum_sq)
                    for i in T.Parallel(TILE):
                        if i < d:
                            sum_sq[0] += x_frag[i] * x_frag[i]
                    T.finalize_reducer(sum_sq)

                    # Compute rstd
                    var = sum_sq[0] / d + eps
                    rstd_val = T.rsqrt(var)
                    z_rstd_ub[0] = rstd_val

                    # Output: y = x * rstd * w (reuse x_frag, no x reload)
                    for i in T.Parallel(TILE):
                        if i < d:
                            y_ub[i] = x_frag[i] * rstd_val * w_frag[i]

                # MTE3: store y[token] and rstd from UB to GM
                row_id = t * N_CORES + core_id
                T.copy(y_ub[:d], Y[base : base + d])
                T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])

    return main


def ref_program(x, weight, eps=1e-6):
    """Reference: PyTorch RMSNorm."""
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + eps)
    return (x.float() * rstd.unsqueeze(-1) * weight.float().unsqueeze(0)).to(x.dtype)


def run_regression_perf(
    kernel_builder=rms_norm_persistent_fwd,
    batch=4096,
    d=4096,
    eps=1e-6,
):
    dtype = torch.float32
    device = torch.device("npu")

    x = torch.randn(batch, d, dtype=dtype, device=device)
    weight = torch.randn(d, dtype=dtype, device=device)

    program = kernel_builder(batch, d, str(dtype)[len("torch.") :])
    kernel = tilelang.compile(program, target="pto", out_idx=[1, 3])

    x_flat = x.view(-1).contiguous()

    kernel(x_flat, weight, eps)
    torch.npu.synchronize()

    N_ITERS = 20

    def run_kernel(kernel=kernel, x_flat=x_flat, weight=weight, eps=eps):
        return kernel(x_flat, weight, eps)

    latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=N_ITERS)
    elapsed_us = latency_ms * 1e3
    total_bytes = batch * d * 4 * 2 + d * 4 + batch * 4
    bw_gbs = total_bytes / (elapsed_us * 1e-6) / 1e9
    print(f"    [batch={batch}, d={d}] {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s")
    return latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark one RMSNorm shape in an isolated process.")
    parser.add_argument(
        "--d",
        type=int,
        choices=(4096, 5120, 7168),
        default=7168,
        help="hidden size; run each shape in a separate process",
    )
    parser.add_argument(
        "--variant",
        choices=("ub", "gm", "both"),
        default="both",
        help="weight initialization path to benchmark",
    )
    args = parser.parse_args()

    dtype = torch.float32
    device = torch.device("npu")

    all_variants = {
        "ub": ("UB to persistent fragment", rms_norm_persistent_fwd),
        "gm": ("GM to persistent fragment", rms_norm_persistent_gm_to_fragment_fwd),
    }
    variant_names = ("ub", "gm") if args.variant == "both" else (args.variant,)

    for variant_key in variant_names:
        variant_name, kernel_builder = all_variants[variant_key]
        print(f"\n=== {variant_name} ===")

        batch = 4096
        d = args.d
        x = torch.randn(batch, d, dtype=dtype, device=device)
        weight = torch.randn(d, dtype=dtype, device=device)
        eps = 1e-6

        print(f"\n--- d={d}, batch={batch} ---")
        program = kernel_builder(batch, d, str(dtype)[len("torch.") :])
        # Y (idx 1) and RSTD (idx 3) are outputs; X, W, eps are inputs.
        kernel = tilelang.compile(program, target="pto", out_idx=[1, 3])

        x_flat = x.view(-1).contiguous()

        # Warmup + verify.
        y_flat, rstd = kernel(x_flat, weight, eps)
        torch.npu.synchronize()

        y = y_flat.view(batch, d)
        expected = ref_program(x, weight, eps)
        diff = (y.float() - expected).abs().max().item()
        ok = "PASS" if diff < 1e-3 else f"FAIL (diff={diff:.1e})"

        # Benchmark.
        N_ITERS = 20

        def run_kernel(kernel=kernel, x_flat=x_flat, weight=weight, eps=eps):
            return kernel(x_flat, weight, eps)

        latency_ms = do_bench(
            run_kernel,
            backend="msprof",
            _n_warmup=30,
            _n_repeat=N_ITERS,
        )
        elapsed_us = latency_ms * 1e3
        total_bytes = batch * d * 4 * 2 + d * 4 + batch * 4
        bw_gbs = total_bytes / (elapsed_us * 1e-6) / 1e9
        print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s  {ok}")
