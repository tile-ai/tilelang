"""High-level SIMT RMSNorm with pipelined row tiles and register-resident weights."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def rms_norm_fwd(batch, d, dtype="float32"):
    if batch <= 0 or batch % 64 or d <= 0:
        raise ValueError("RMSNorm expects a positive batch divisible by 64 and d > 0")
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    # Batch adjacent rows to amortize VF launches and the small RSTD stores.
    # Reserve 8 KiB of the SIMT-usable 216 KiB UB for reduction scratch/RSTD.
    # Keep room for two input tiles, reusing them for output when necessary.
    ub_budget = 208 * 1024
    rows = 4
    while rows > 1 and (batch % (N_CORES * rows) or (2 * rows + 1) * d * 4 > ub_budget):
        rows //= 2
    reuse_input = (3 * rows + 1) * d * 4 > ub_budget
    if rows == 1:
        threads = 256 if TILE > 4096 else 128
    else:
        threads = 256 if rows == 4 else 512
    # Keep the thread domain within one row, including very narrow rows.
    threads = min(threads, TILE)
    iterations = batch // (N_CORES * rows)
    # Keep the RSTD cache bounded for large batches.
    bulk_rstd = iterations <= 128
    rstd_tiles = iterations if bulk_rstd else 1
    N = batch * d

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype), W: T.Tensor((d,), dtype), RSTD: T.Tensor((batch,), "float32"), eps: T.float32
    ):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), dtype)
            # Only fragments need power-of-two padding; DMA tiles stay compact.
            x_ub = T.alloc_shared(rows * d, "float32")
            y_ub = T.alloc_shared(rows * d, "float32")
            z_rstd_ub = T.alloc_shared((rstd_tiles, 8), "float32")
            rstd_view = T.reshape(RSTD, (iterations, N_CORES * rows))
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1, z_rstd_ub: 1 if bulk_rstd else 2})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=2):
                row_id = (t * N_CORES + core_id) * rows
                base = row_id * d
                slot = t if bulk_rstd else 0
                T.copy(X[base : base + rows * d], x_ub)
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    sum_sq = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(sum_sq)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(sum_sq[r], x_frag[r, i] * x_frag[r, i])
                    sum_result = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sum_sq, sum_result)
                    for r in T.Parallel(rows):
                        sum_result[r] = T.rsqrt(sum_result[r] / d + eps)
                    w_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        w_frag[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            if reuse_input:
                                x_ub[r * d + i] = x_frag[r, i] * sum_result[r] * w_frag[i]
                            else:
                                y_ub[r * d + i] = x_frag[r, i] * sum_result[r] * w_frag[i]
                    T.copy(sum_result, z_rstd_ub[slot, :rows])
                if reuse_input:
                    T.copy(x_ub, Y[base : base + rows * d])
                else:
                    T.copy(y_ub, Y[base : base + rows * d])
                if not bulk_rstd:
                    T.copy(z_rstd_ub[0, :rows], RSTD[row_id : row_id + rows])
            if bulk_rstd:
                T.copy(z_rstd_ub[:, :rows], rstd_view[:, core_id * rows : (core_id + 1) * rows])

    return main


def ref_program(x, weight, eps=1e-06):
    """Reference: PyTorch RMSNorm."""
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + eps)
    return (x.float() * rstd.unsqueeze(-1) * weight.float().unsqueeze(0)).to(x.dtype)


def run_regression_perf(batch=4096, d=4096, eps=1e-6):
    kernel = tilelang.compile(rms_norm_fwd(batch, d), target="ascend", out_idx=[1, 3])
    x = torch.randn(batch * d, device="npu")
    weight = torch.randn(d, device="npu")
    latency = do_bench(lambda: kernel(x, weight, eps), backend="msprof", _n_warmup=30, _n_repeat=50)
    # Logical X and W reads, plus Y and RSTD writes; all tensors are FP32.
    num_bytes = (2 * x.numel() + weight.numel() + batch) * x.element_size()
    bandwidth = num_bytes / (latency * 1e6)
    print(f"rmsnorm (batch={batch}, d={d}): {latency * 1000:.2f} us | {bandwidth:.2f} GB/s")
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    batch, d, eps = 128, 4096, 1e-6
    x, weight = torch.randn(batch, d, device="npu"), torch.randn(d, device="npu")
    kernel = tilelang.compile(rms_norm_fwd(batch, d), target="ascend", out_idx=[1, 3])
    y, rstd = kernel(x.flatten(), weight, eps)
    torch.testing.assert_close(y.reshape(batch, d), ref_program(x, weight, eps), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rstd, torch.rsqrt(x.square().mean(-1) + eps), rtol=1e-5, atol=1e-5)
    print("rmsnorm: correctness passed")
    if args.bench:
        run_regression_perf()
