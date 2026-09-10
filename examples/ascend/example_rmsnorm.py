"""RMSNorm forward kernel on Ascend NPU with fragment trick.

Key techniques demonstrated:
  1. Double buffer + persistent loop (64 cores)
  2. Fragment trick: T.alloc_fragment for vectorized float4 load + register reuse
  3. alloc_reducer + finalize_reducer for cross-warp reduce
  4. Adaptive thread count based on hidden dimension

Performance (fp32, batch=4096, HBM peak 1600 GB/s):
  d=4096: ~100 us,  1300+ GB/s (80%+ peak)
  d=7168: ~165 us,  1400+ GB/s (88%+ peak)
"""

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.profiler import do_bench


def rms_norm_fwd(batch, d, dtype="float32"):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)

    # Adaptive threads: keep ~32 elements per thread (8 float4)
    threads = 256 if TILE > 4096 else 128
    # Measured with measure_vf_latency.py (cannsim run-vf backend). Other
    # shapes retain automatic estimation until they are measured explicitly.
    vf_latency = {4096: 1824, 5120: 1825, 7168: 2436}.get(d, 0)

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

            # Load weights once
            T.copy(W[:d], w_ub[:d])

            for t in T.Pipelined(n_tokens_per_core, num_stages=2):
                base = (t * N_CORES + core_id) * d

                T.copy(X[base : base + d], x_ub[:d])

                with T.SimtVF(threads=threads, latency=vf_latency):
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
                            y_ub[i] = x_frag[i] * rstd_val * w_ub[i]

                # MTE3: store y[token] and rstd from UB to GM
                row_id = t * N_CORES + core_id
                T.copy(y_ub[:d], Y[base : base + d])
                T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])

    return main


def ref_program(x, weight, eps=1e-6):
    """Reference: PyTorch RMSNorm."""
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + eps)
    return (x.float() * rstd.unsqueeze(-1) * weight.float().unsqueeze(0)).to(x.dtype)


def run_regression_perf(batch=4096, d=4096, eps=1e-6):
    dtype = torch.float32
    device = torch.device("npu")

    x = torch.randn(batch, d, dtype=dtype, device=device)
    weight = torch.randn(d, dtype=dtype, device=device)

    program = rms_norm_fwd(batch, d, str(dtype)[len("torch.") :])
    kernel = tilelang.compile(program, out_idx=[1, 3])

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
    dtype = torch.float32
    device = torch.device("npu")

    # Test correctness across common hidden sizes
    for d in [4096, 5120, 7168]:
        batch = 4096
        x = torch.randn(batch, d, dtype=dtype, device=device)
        weight = torch.randn(d, dtype=dtype, device=device)
        eps = 1e-6

        print(f"\n--- d={d}, batch={batch} ---")
        program = rms_norm_fwd(batch, d, str(dtype)[len("torch.") :])
        # Y (idx 1) and RSTD (idx 3) are outputs; X, W, eps are inputs
        kernel = tilelang.compile(program, out_idx=[1, 3])

        x_flat = x.view(-1).contiguous()

        # Warmup + verify
        y_flat, rstd = kernel(x_flat, weight, eps)
        torch.npu.synchronize()

        y = y_flat.view(batch, d)
        expected = ref_program(x, weight, eps)
        diff = (y.float() - expected).abs().max().item()
        ok = "PASS" if diff < 1e-3 else f"FAIL (diff={diff:.1e})"

        # Benchmark
        N_ITERS = 20

        def run_kernel(kernel=kernel, x_flat=x_flat, weight=weight, eps=eps):
            return kernel(x_flat, weight, eps)

        latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=N_ITERS)
        elapsed_us = latency_ms * 1e3
        total_bytes = batch * d * 4 * 2 + d * 4 + batch * 4
        bw_gbs = total_bytes / (elapsed_us * 1e-6) / 1e9
        print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s  {ok}")

    # Print kernel source for inspection
    print("\n--- Generated Ascend Source (d=4096) ---")
    program = rms_norm_fwd(4096, 4096, "float32")
    kernel = tilelang.compile(program, out_idx=-1)
    print(kernel.get_kernel_source())
