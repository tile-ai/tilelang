"""MHC norm-fn kernels on Ascend NPU.

Groups the norm-weight merge forward/backward and the pre-norm normalization
forward, mirroring `norm_fn_kernel.py` in the TileKernels mHC module. All
kernels are written in plain TileLang; the AutoSimtVF pass promotes the
`T.Parallel` loops into SIMT vector regions automatically.
"""

import torch
import tilelang
import tilelang.ascend.language as T


def mhc_fn_normw_merge_fwd(m, n, dtype=T.float32):
    n_blk = 256
    num_n_blocks = (n + n_blk - 1) // n_blk

    @T.prim_func
    def main(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[(n,), dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        with T.Kernel(m * num_n_blocks) as pid:
            pid_m = pid // num_n_blocks
            pid_n = pid % num_n_blocks
            for i1_n in T.Parallel(n_blk):
                i_n = pid_n * n_blk + i1_n
                if i_n < n:
                    out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return main


def mhc_fn_normw_merge_bwd(m, n, dtype=T.float32):
    n_blk = 256

    @T.prim_func
    def main(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[(n,), dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[(n,), dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
            normw_frag = T.alloc_fragment(n_blk, dtype)
            T.copy(normw[pid_n * n_blk], normw_frag)

            normw_grad_frag = T.alloc_fragment(n_blk, dtype)
            T.clear(normw_grad_frag)

            for i_m in T.serial(m):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        fn_grad[i_m, i_n] = out_fn_grad[i_m, i_n] * normw_frag[i1_n]
                        normw_grad_frag[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

            for i1_n in T.Parallel(n_blk):
                normw_grad[pid_n * n_blk + i1_n] = normw_grad_frag[i1_n]

    return main


def mhc_pre_norm_fn_fwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps, n_splits):
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def main(
        out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
    ) -> None:
        with T.Kernel(num_tokens) as pid:
            rms = T.alloc_var(T.float32)
            out_l = T.alloc_fragment(mhc_mult3, T.float32)
            out_l0 = T.alloc_fragment(mhc_mult3, T.float32)
            T.clear(out_l)
            for k in T.serial(n_rms_group):
                rms = 0
                for i_split in T.serial(n_splits):
                    rms += sqrsum_splitted[i_split, pid, k]
                sqrsum[pid, k] = rms
                rms = T.rsqrt(rms / rms_group_size + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    out_l0[j] = 0
                    for i_split in T.serial(n_splits):
                        out_l0[j] += out_mul_splitted[i_split, pid, k, j]
                    out_l[j] += out_l0[j] * rms
                T.copy(out_l0, out_mul[pid, k, :])
            T.copy(out_l[:], out[pid, :])

    return main


def ref_program_fn_normw_merge_fwd(fn, normw):
    return fn * normw


def ref_program_fn_normw_merge_bwd(fn, normw, out_fn_grad):
    return out_fn_grad * normw, (out_fn_grad * fn).sum(dim=0)


def ref_program_pre_norm_fn_fwd_norm(out_mul_splitted, sqrsum_splitted, rms_group_size, rms_eps):
    out_mul = out_mul_splitted.sum(dim=0)
    sqrsum = sqrsum_splitted.sum(dim=0)
    rms = torch.rsqrt(sqrsum / rms_group_size + rms_eps)
    out = (out_mul * rms.unsqueeze(-1)).sum(dim=1)
    return out_mul, sqrsum, out


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("npu")

    m, n = 24, 7168
    fn = torch.randn((m, n), device=device, dtype=torch.float32)
    normw = torch.randn((n,), device=device, dtype=torch.float32)

    kernel = tilelang.compile(mhc_fn_normw_merge_fwd(m, n), target="ascend", out_idx=-1)
    actual = kernel(fn, normw)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program_fn_normw_merge_fwd(fn, normw), rtol=1e-5, atol=1e-6)
    print("PASS: mhc_fn_normw_merge_fwd")

    m, n = 128, 512
    fn = torch.randn((m, n), device=device, dtype=torch.float32)
    normw = torch.randn((n,), device=device, dtype=torch.float32)
    out_fn_grad = torch.randn((m, n), device=device, dtype=torch.float32)

    kernel = tilelang.compile(mhc_fn_normw_merge_bwd(m, n), target="ascend", out_idx=[-2, -1])
    fn_grad, normw_grad = kernel(fn, normw, out_fn_grad)
    torch.npu.synchronize()
    expected_fn_grad, expected_normw_grad = ref_program_fn_normw_merge_bwd(fn, normw, out_fn_grad)
    torch.testing.assert_close(fn_grad, expected_fn_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(normw_grad, expected_normw_grad, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_fn_normw_merge_bwd")

    num_tokens, mhc_mult3, n_rms_group = 128, 4, 8
    rms_group_size, rms_eps, n_splits = 896, 1e-6, 4
    out_mul_splitted = torch.randn((n_splits, num_tokens, n_rms_group, mhc_mult3), device=device, dtype=torch.float32)
    sqrsum_splitted = torch.rand((n_splits, num_tokens, n_rms_group), device=device, dtype=torch.float32)

    kernel = tilelang.compile(
        mhc_pre_norm_fn_fwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps, n_splits),
        target="ascend",
        out_idx=[2, 3, 4],
    )
    actual_out_mul, actual_sqrsum, actual_out = kernel(out_mul_splitted, sqrsum_splitted)
    torch.npu.synchronize()
    expected_out_mul, expected_sqrsum, expected_out = ref_program_pre_norm_fn_fwd_norm(
        out_mul_splitted, sqrsum_splitted, rms_group_size, rms_eps
    )
    torch.testing.assert_close(actual_out_mul, expected_out_mul, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_sqrsum, expected_sqrsum, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_pre_norm_fn_fwd_norm")
