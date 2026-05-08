import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-3, -2, -1])
def _layernorm_fwd(N, D, eps=1e-5, blk_m=1, threads=256, in_dtype="bfloat16", out_dtype="bfloat16"):
    accum_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((N, D), in_dtype),
        gamma: T.Tensor((D,), in_dtype),
        beta: T.Tensor((D,), in_dtype),
        Y: T.Tensor((N, D), out_dtype),
        Mean: T.Tensor((N,), accum_dtype),
        Rstd: T.Tensor((N,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, blk_m), threads=threads) as bx:
            X_smem = T.alloc_shared((blk_m, D), in_dtype)
            G_smem = T.alloc_shared((D,), in_dtype)
            B_smem = T.alloc_shared((D,), in_dtype)
            X_local = T.alloc_fragment((blk_m, D), accum_dtype)
            X_sq_local = T.alloc_fragment((blk_m, D), accum_dtype)
            sum_row = T.alloc_fragment((blk_m,), accum_dtype)
            sumsq_row = T.alloc_fragment((blk_m,), accum_dtype)
            mean_row = T.alloc_fragment((blk_m,), accum_dtype)
            rstd_row = T.alloc_fragment((blk_m,), accum_dtype)

            T.copy(X[bx * blk_m, 0], X_smem)
            T.copy(gamma, G_smem)
            T.copy(beta, B_smem)

            for i, j in T.Parallel(blk_m, D):
                X_local[i, j] = T.Cast(accum_dtype, X_smem[i, j])
            for i, j in T.Parallel(blk_m, D):
                X_sq_local[i, j] = X_local[i, j] * X_local[i, j]

            T.reduce_sum(X_local, sum_row, dim=1)
            T.reduce_sum(X_sq_local, sumsq_row, dim=1)

            inv_D = T.float32(1.0) / T.Cast(accum_dtype, D)
            for i in T.Parallel(blk_m):
                mean_row[i] = sum_row[i] * inv_D
                rstd_row[i] = T.rsqrt(sumsq_row[i] * inv_D - mean_row[i] * mean_row[i] + T.Cast(accum_dtype, eps))
                Mean[bx * blk_m + i] = mean_row[i]
                Rstd[bx * blk_m + i] = rstd_row[i]

            for i, j in T.Parallel(blk_m, D):
                norm = (X_local[i, j] - mean_row[i]) * rstd_row[i]
                X_smem[i, j] = T.Cast(
                    out_dtype,
                    norm * T.Cast(accum_dtype, G_smem[j]) + T.Cast(accum_dtype, B_smem[j]),
                )

            T.copy(X_smem, Y[bx * blk_m, 0])

    return main


@tilelang.jit(out_idx=[-3])
def _layernorm_bwd(N, D, blk_m=32, threads=256, in_dtype="bfloat16"):
    accum_dtype = "float"

    @T.prim_func
    def main(
        DY: T.Tensor((N, D), in_dtype),
        X: T.Tensor((N, D), in_dtype),
        gamma: T.Tensor((D,), in_dtype),
        Mean: T.Tensor((N,), accum_dtype),
        Rstd: T.Tensor((N,), accum_dtype),
        DX: T.Tensor((N, D), in_dtype),
        DGamma: T.Tensor((D,), accum_dtype),
        DBeta: T.Tensor((D,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, blk_m), threads=threads) as bx:
            DY_smem = T.alloc_shared((1, D), in_dtype)
            X_smem = T.alloc_shared((1, D), in_dtype)
            G_smem = T.alloc_shared((D,), in_dtype)
            DX_smem = T.alloc_shared((1, D), in_dtype)

            x_hat = T.alloc_fragment((1, D), accum_dtype)
            dx_hat = T.alloc_fragment((1, D), accum_dtype)
            dx_hat_x_hat = T.alloc_fragment((1, D), accum_dtype)
            sum_dx_hat = T.alloc_fragment((1,), accum_dtype)
            sum_dx_hat_x_hat = T.alloc_fragment((1,), accum_dtype)
            dgamma_acc = T.alloc_fragment((D,), accum_dtype)
            dbeta_acc = T.alloc_fragment((D,), accum_dtype)

            T.copy(gamma, G_smem)
            T.clear(dgamma_acc)
            T.clear(dbeta_acc)

            inv_D = T.float32(1.0) / T.Cast(accum_dtype, D)

            for k in T.serial(blk_m):
                row = bx * blk_m + k
                T.copy(DY[row, 0], DY_smem)
                T.copy(X[row, 0], X_smem)
                m_v = Mean[row]
                r_v = Rstd[row]

                for i, j in T.Parallel(1, D):
                    xv = T.Cast(accum_dtype, X_smem[i, j])
                    dy_v = T.Cast(accum_dtype, DY_smem[i, j])
                    g_v = T.Cast(accum_dtype, G_smem[j])
                    x_hat[i, j] = (xv - m_v) * r_v
                    dx_hat[i, j] = dy_v * g_v
                    dx_hat_x_hat[i, j] = dx_hat[i, j] * x_hat[i, j]

                T.reduce_sum(dx_hat, sum_dx_hat, dim=1)
                T.reduce_sum(dx_hat_x_hat, sum_dx_hat_x_hat, dim=1)

                for i, j in T.Parallel(1, D):
                    c1 = sum_dx_hat_x_hat[0] * inv_D
                    c2 = sum_dx_hat[0] * inv_D
                    dx_v = r_v * (dx_hat[i, j] - c2 - x_hat[i, j] * c1)
                    DX_smem[i, j] = T.Cast(in_dtype, dx_v)
                    dy_v = T.Cast(accum_dtype, DY_smem[i, j])
                    dgamma_acc[j] += dy_v * x_hat[i, j]
                    dbeta_acc[j] += dy_v

                T.copy(DX_smem, DX[row, 0])

            for j in T.Parallel(D):
                T.atomic_add(DGamma[j], dgamma_acc[j], memory_order="relaxed")
                T.atomic_add(DBeta[j], dbeta_acc[j], memory_order="relaxed")

    return main


_TORCH_DTYPE_TO_TL = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        N, D = x.shape
        in_dtype = _TORCH_DTYPE_TO_TL[x.dtype]
        kernel = _layernorm_fwd(N, D, eps=eps, in_dtype=in_dtype, out_dtype=in_dtype)
        y, mean, rstd = kernel(x, gamma, beta)
        ctx.save_for_backward(x, gamma, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, gamma, mean, rstd = ctx.saved_tensors
        N, D = x.shape
        if not dy.is_contiguous():
            dy = dy.contiguous()
        in_dtype = _TORCH_DTYPE_TO_TL[x.dtype]
        kernel = _layernorm_bwd(N, D, in_dtype=in_dtype)
        dgamma = torch.zeros(D, dtype=torch.float32, device=x.device)
        dbeta = torch.zeros(D, dtype=torch.float32, device=x.device)
        dx = kernel(dy, x, gamma, mean, rstd, dgamma, dbeta)
        return dx, dgamma.to(gamma.dtype), dbeta.to(gamma.dtype), None


def layer_norm(x, gamma, beta, eps=1e-5):
    return LayerNormFn.apply(x, gamma, beta, eps)


def ref_program(x, gamma, beta, eps):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps=eps)


if __name__ == "__main__":
    N, D = 4096, 8192
    eps = 1e-5
    x = torch.randn(N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn(D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn(D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")

    y = layer_norm(x, g, b, eps)
    y_ref = ref_program(x, g, b, eps)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    y.backward(dy)
    dx_ours, dg_ours, db_ours = x.grad, g.grad, b.grad

    x.grad = g.grad = b.grad = None
    y_ref = ref_program(x, g, b, eps)
    y_ref.backward(dy)
    torch.testing.assert_close(dx_ours, x.grad, rtol=1e-2, atol=1e-2)
    print("All checks pass.")

    from tilelang.profiler import do_bench

    ms_fwd = do_bench(lambda: layer_norm(x.detach(), g.detach(), b.detach(), eps), backend="event")
    ms_fwd_ref = do_bench(lambda: ref_program(x.detach(), g.detach(), b.detach(), eps), backend="event")
    print(f"fwd  tilelang: {ms_fwd:.4f} ms   ref: {ms_fwd_ref:.4f} ms")

    xx = x.detach().clone().requires_grad_(True)
    gg = g.detach().clone().requires_grad_(True)
    bb = b.detach().clone().requires_grad_(True)
    yy = layer_norm(xx, gg, bb, eps)
    ms_bwd = do_bench(lambda: yy.backward(dy, retain_graph=True), backend="event")

    xx_r = x.detach().clone().requires_grad_(True)
    gg_r = g.detach().clone().requires_grad_(True)
    bb_r = b.detach().clone().requires_grad_(True)
    yy_r = ref_program(xx_r, gg_r, bb_r, eps)
    ms_bwd_ref = do_bench(lambda: yy_r.backward(dy, retain_graph=True), backend="event")
    print(f"bwd  tilelang: {ms_bwd:.4f} ms   ref: {ms_bwd_ref:.4f} ms")
