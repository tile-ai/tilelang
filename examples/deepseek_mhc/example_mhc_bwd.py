# NOTE: This bwd script is not an official upstream script; it is community-written and provided for reference only.
# checkout pr: https://github.com/tile-ai/tilelang/pull/1758
import torch

import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from tqdm import trange


dtype = torch.float32

seqlen = 65536
n_stream = 16
iters = 100
repeat = 512

EPS = 1e-10


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


def sinkhorn_bwd_configs(n_stream, seqlen):
    """Generate autotune configurations for different tilesize and threads"""
    configs = []

    # Explore different tile sizes and thread counts
    tilesizes = [1, 2, 4, 8, 16, 32, 64]
    thread_counts = [32, 64, 128, 256]

    for tilesize in tilesizes:
        # Skip if tilesize doesn't divide seqlen evenly (optional constraint)
        if seqlen % tilesize != 0:
            continue

        for threads in thread_counts:
            configs.append({"tilesize": tilesize, "threads": threads})

    return configs


@tilelang.autotune(
    configs=sinkhorn_bwd_configs(n_stream, seqlen),
    warmup=4,
    rep=repeat,
)
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def sinkhorn_bwd_implicit_cg(n_stream: int, tilesize: int = 32, threads: int = 128):
    seqlen = T.dynamic("seqlen")
    tensor_shape = [seqlen, n_stream, n_stream]
    dtype = T.float32

    @T.macro
    def matvec_A(R, x1, x2, buf, y1, y2):
        for i_tile, i, j in T.Parallel(tilesize, n_stream, n_stream):
            buf[i_tile, i, j] = R[i_tile, i, j] * x2[i_tile, j]
        T.reduce_sum(buf, y1, dim=-1)

        for i_tile, i, j in T.Parallel(tilesize, n_stream, n_stream):
            buf[i_tile, i, j] = R[i_tile, i, j] * x1[i_tile, i]
        T.reduce_sum(buf, y2, dim=-2)

        for i_tile, i in T.Parallel(tilesize, n_stream):
            y1[i_tile, i] += x1[i_tile, i]
            y2[i_tile, i] += x2[i_tile, i]

    @T.macro
    def dot(x1, x2, y1, y2, buf, out):
        for i_tile, i in T.Parallel(tilesize, n_stream):
            buf[i_tile, i] = x1[i_tile, i] * y1[i_tile, i] + x2[i_tile, i] * y2[i_tile, i]

        T.reduce_sum(buf, out, dim=-1)

    @T.prim_func
    def main(
        out: T.Tensor(tensor_shape, dtype),
        dout: T.Tensor(tensor_shape, dtype),
        res: T.Tensor(tensor_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seqlen, tilesize), threads=threads) as i_seq:
            R = T.alloc_fragment([tilesize, n_stream, n_stream], dtype=dtype)
            dR = T.alloc_fragment([tilesize, n_stream, n_stream], dtype=dtype)
            RdR = T.alloc_fragment([tilesize, n_stream, n_stream], dtype=dtype)
            res_tile = T.alloc_shared([tilesize, n_stream, n_stream], dtype=dtype)
            b1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            b2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            x1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            x2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            r1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            r2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            p1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            p2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            alpha = T.alloc_fragment([tilesize, n_stream], dtype=dtype)
            beta = T.alloc_fragment([tilesize, n_stream], dtype=dtype)
            r_normsq = T.alloc_fragment([tilesize], dtype=dtype)
            r_new_normsq = T.alloc_fragment([tilesize], dtype=dtype)
            Ap1 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            Ap2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)
            pAp = T.alloc_fragment([tilesize], dtype=dtype)

            # Buffers for intermediate results
            buf1 = T.alloc_shared([tilesize, n_stream, n_stream], dtype=dtype)
            buf2 = T.alloc_shared([tilesize, n_stream], dtype=dtype)

            T.copy(out[i_seq * tilesize : (i_seq + 1) * tilesize, :, :], R)
            T.copy(dout[i_seq * tilesize : (i_seq + 1) * tilesize, :, :], dR)

            for i_tile, i_nx, i_ny in T.Parallel(tilesize, n_stream, n_stream):
                RdR[i_tile, i_nx, i_ny] = R[i_tile, i_nx, i_ny] * dR[i_tile, i_nx, i_ny]

            T.reduce_sum(RdR, b1, dim=-1)
            T.reduce_sum(RdR, b2, dim=-2)

            T.fill(x1, 0.0)
            T.fill(x2, 0.0)

            matvec_A(R, x1, x2, buf1, r1, r2)

            for i_tile, i_n in T.Parallel(tilesize, n_stream):
                r1[i_tile, i_n] = b1[i_tile, i_n] - r1[i_tile, i_n]

            for i_tile, i_n in T.Parallel(tilesize, n_stream):
                r2[i_tile, i_n] = b2[i_tile, i_n] - r2[i_tile, i_n]

            T.copy(r1, p1)
            T.copy(r2, p2)

            dot(r1, r2, r1, r2, buf2, r_normsq)

            # Conjugate gradient: iteration starts
            for _ in T.serial(2 * n_stream):
                matvec_A(R, p1, p2, buf1, Ap1, Ap2)

                dot(p1, p2, Ap1, Ap2, buf2, pAp)

                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    # VERY important to avoid divide by zero
                    alpha[i_tile, i_n] = r_normsq[i_tile] / (pAp[i_tile] + EPS)
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    x1[i_tile, i_n] += alpha[i_tile, i_n] * p1[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    x2[i_tile, i_n] += alpha[i_tile, i_n] * p2[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    r1[i_tile, i_n] -= alpha[i_tile, i_n] * Ap1[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    r2[i_tile, i_n] -= alpha[i_tile, i_n] * Ap2[i_tile, i_n]

                dot(r1, r2, r1, r2, buf2, r_new_normsq)

                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    # not very important to avoid divide by zero, but it's good to have it
                    beta[i_tile, i_n] = r_new_normsq[i_tile] / (r_normsq[i_tile] + EPS)
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    p1[i_tile, i_n] = r1[i_tile, i_n] + beta[i_tile, i_n] * p1[i_tile, i_n]
                for i_tile, i_n in T.Parallel(tilesize, n_stream):
                    p2[i_tile, i_n] = r2[i_tile, i_n] + beta[i_tile, i_n] * p2[i_tile, i_n]

                T.copy(r_new_normsq, r_normsq)
            # Conjugate gradient: iteration ends

            for i_tile, i_nx, i_ny in T.Parallel(tilesize, n_stream, n_stream):
                res_tile[i_tile, i_nx, i_ny] = (dR[i_tile, i_nx, i_ny] - x1[i_tile, i_nx] - x2[i_tile, i_ny]) * R[i_tile, i_nx, i_ny]

            T.copy(res_tile, res[i_seq * tilesize : (i_seq + 1) * tilesize, :, :])

    return main


def main():
    print("Autotuning TileLang kernel for sinkhorn backward pass")
    print(f"{seqlen = }")
    print(f"{n_stream = }")
    print(f"{iters = }")
    print(f"{repeat = }")

    ######################################################################
    # Variable
    ######################################################################
    dist = torch.distributions.uniform.Uniform(0.0, 4.0)
    device = torch.device("cuda")
    M = dist.sample((seqlen, n_stream, n_stream)).to(device)
    M.requires_grad_()

    ######################################################################
    # Shared forward + one shared loss weight
    ######################################################################
    R, P = sinkhorn_forward(M, iters)
    loss_weight = torch.randn_like(R)

    ######################################################################
    # Method A: Autograd (reference)
    ######################################################################
    loss_a = (R * loss_weight).sum()
    loss_a.backward()
    grad_M_autograd = M.grad.detach().clone()

    ######################################################################
    # Method B: Implicit differentiation with autotuning
    ######################################################################
    grad_R = loss_weight

    print("\n" + "=" * 60)
    print("Starting autotuning...")
    print("=" * 60)

    # Set autotune inputs
    with set_autotune_inputs(R, grad_R):
        kernel = sinkhorn_bwd_implicit_cg(n_stream)
    print(kernel.get_kernel_source())
    print("\n" + "=" * 60)
    print("Autotuning completed! Running with best configuration...")
    print("=" * 60)

    # Warmup and timing with best config
    a = torch.randn(8192, 8192, device=device)
    for _ in trange(4, desc="Warmup"):
        _ = a @ a
        grad_M_implicit = kernel(R, grad_R)
        torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(repeat):
        grad_M_implicit = kernel(R, grad_R)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)

    print(f"\nKernel execution time ({repeat = }): {elapsed_time_ms:.3f} ms")
    print(f"Average time per iteration: {elapsed_time_ms / repeat:.3f} ms")

    ######################################################################
    # Compare
    ######################################################################
    g1 = grad_M_autograd
    g2 = grad_M_implicit

    abs_diff = (g1 - g2).abs()
    # Use max of absolute values for more stable relative error
    rel_diff = abs_diff / (torch.maximum(g1.abs(), g2.abs()) + 1e-8)

    print("\n" + "=" * 60)
    print("Comparison of gradients dL/dM")
    print("=" * 60)

    def format_list(ls):
        return [f"{x:.2e}" for x in ls]

    MAE = abs_diff.mean(dim=(-1, -2)).tolist()
    max_abs_diff = abs_diff.reshape(seqlen, -1).max(-1).values.tolist()
    mean_rel_diff = rel_diff.mean(dim=(-1, -2)).tolist()
    max_rel_diff = rel_diff.reshape(seqlen, -1).max(-1).values.tolist()

    print(f"Max MAE = {max(MAE):.6e}")
    print(f"Max max_abs_diff = {max(max_abs_diff):.6e}")
    print(f"Max mean_rel_diff = {max(mean_rel_diff):.6e}")
    print(f"Max max_rel_diff = {max(max_rel_diff):.6e}")

    print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
    print("\nGrad (implicit) sample:\n", g2[0, :3, :3])


if __name__ == "__main__":
    main()
