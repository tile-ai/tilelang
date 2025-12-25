import torch
import tilelang
import tilelang.language as T
from typing import Tuple
from tilelang.utils.tensor import torch_assert_close


@tilelang.jit(out_idx=[1, 2])
def per_token_cast_to_fp8(M, N, blk_m):
    dtype = T.float
    group_size = 128
    fp8_min = -448.0
    fp8_max = 448.0

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((M, N), dtype), X_fp8: T.Tensor((M, N), T.float8_e4m3fn), X_amax: T.Tensor((M, T.ceildiv(N, group_size)), dtype)
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (bx, by):
            row = bx
            row_g_id = by
            y_local = T.alloc_fragment((blk_m, group_size), dtype)
            y_amax_local = T.alloc_fragment((blk_m,), dtype)
            y_s_local = T.alloc_fragment((blk_m,), dtype)
            y_q_local = T.alloc_fragment((blk_m, group_size), dtype)
            y_q_local_fp8 = T.alloc_fragment((blk_m, group_size), T.float8_e4m3fn)

            T.copy(X[row * blk_m : (row + 1) * blk_m, row_g_id * group_size : (row_g_id + 1) * group_size], y_local)
            T.reduce_absmax(y_local, y_amax_local, dim=1)
            for i in T.Parallel(blk_m):
                y_amax_local[i] = T.max(y_amax_local[i], 1e-4)
                y_s_local[i] = y_amax_local[i] / fp8_max
            for i, j in T.Parallel(blk_m, group_size):
                y_q_local[i, j] = T.clamp(y_local[i, j] / y_s_local[i], fp8_min, fp8_max)
            T.copy(y_q_local, y_q_local_fp8)
            for i in T.Parallel(blk_m):
                X_amax[row * blk_m + i, row_g_id] = y_s_local[i]
            T.copy(y_q_local_fp8, X_fp8[row * blk_m : (row + 1) * blk_m, row_g_id * group_size : (row_g_id + 1) * group_size])

    return per_token_cast


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def ref_program(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # this function don't support cpu tensor
    assert x.dim() == 2
    m, n = x.shape
    new_n = ceil_div(n, 128) * 128
    x_padded = torch.nn.functional.pad(x, (0, new_n - n))
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    x_fp8 = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(m, -1)[:, :n].contiguous()
    return x_fp8, (x_amax / 448.0).view(m, -1)


def main(M=8192, N=8192, blk_m=8):
    kernel = per_token_cast_to_fp8(M, N, blk_m)
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    x_fp8, x_amax = kernel(x)
    x_fp8_ref, x_amax_ref = ref_program(x)

    print("x_fp8:", x_fp8, x_fp8.shape)
    print("x_amax:", x_amax, x_amax.shape)
    print("x_fp8_ref:", x_fp8_ref, x_fp8_ref.shape)
    print("x_amax_ref:", x_amax_ref, x_amax_ref.shape)

    torch_assert_close(x_fp8.to(torch.float32), x_fp8_ref.to(torch.float32), rtol=0.01, atol=0.01)
    torch_assert_close(x_amax, x_amax_ref, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench()
    print("Tile-lang: {:.2f} ms".format(latency))

    from tilelang.profiler import do_bench
    from example_triton_cast_to_fp8 import per_token_group_quant_fp8

    def run_triton():
        x_fp8_triton_, x_amax_triton_ = per_token_group_quant_fp8(x, 128, 1e-4, dtype=torch.float8_e4m3fn, column_major_scales=False)
        return x_fp8_triton_, x_amax_triton_

    x_fp8_triton, x_amax_triton = run_triton()
    latency = do_bench(run_triton)
    print("Triton: {:.2f} ms".format(latency))


def run_regression_perf(M=8192, N=8192, blk_m=8):
    kernel = per_token_cast_to_fp8(M, N, blk_m)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(x)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    main()
