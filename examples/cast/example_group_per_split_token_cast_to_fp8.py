import torch
import tilelang
import tilelang.language as T
from typing import Tuple
from tilelang.utils.tensor import torch_assert_close

# support bfloat16, float, float16
dtype = T.bfloat16
accum_dtype = T.float32


@tilelang.jit(out_idx=[2, 3])
def group_per_split_token_cast_to_fp8(M, M_max, N, BG, blk_m):
    group_size = 128
    fp8_min = -448.0
    fp8_max = 448.0

    @T.prim_func
    def group_per_split_token_cast(
        X: T.Tensor((M, N), dtype),
        batch_sizes: T.Tensor((BG,), T.int32),
        X_fp8: T.Tensor((BG, M_max, N), T.float8_e4m3fn),
        X_amax: T.Tensor((BG, M_max, T.ceildiv(N, group_size)), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M_max, blk_m), T.ceildiv(N, group_size), BG, threads=128) as (bx, by, bz):
            row = bx
            row_g_id = by
            bg = bz
            y_local = T.alloc_fragment((blk_m, group_size), accum_dtype)
            y_amax_local = T.alloc_fragment((blk_m,), accum_dtype)
            y_s_local = T.alloc_fragment((blk_m,), accum_dtype)
            y_q_local = T.alloc_fragment((blk_m, group_size), accum_dtype)
            y_q_local_fp8 = T.alloc_fragment((blk_m, group_size), T.float8_e4m3fn)
            row_offset = T.alloc_var(dtype=T.int32)

            row_offset = 0
            for i in T.serial(bg):
                row_offset += batch_sizes[i]

            T.copy(
                X[row_offset + row * blk_m : row_offset + (row + 1) * blk_m, row_g_id * group_size : (row_g_id + 1) * group_size],
                y_local,
            )
            T.reduce_absmax(y_local, y_amax_local, dim=1)
            for i in T.Parallel(blk_m):
                y_amax_local[i] = T.max(y_amax_local[i], 1e-4)
                y_s_local[i] = T.if_then_else(row * blk_m + i < batch_sizes[bg], y_amax_local[i] / fp8_max, 0)
            for i, j in T.Parallel(blk_m, group_size):
                y_q_local[i, j] = T.clamp(y_local[i, j] / y_s_local[i], fp8_min, fp8_max)
            T.copy(y_q_local, y_q_local_fp8)
            for i, j in T.Parallel(blk_m, group_size):
                y_q_local_fp8[i, j] = T.if_then_else(row * blk_m + i < batch_sizes[bg], y_q_local[i, j], 0)
            for i in T.Parallel(blk_m):
                X_amax[bg, row * blk_m + i, row_g_id] = y_s_local[i]
            T.copy(y_q_local_fp8, X_fp8[bg, row * blk_m : (row + 1) * blk_m, row_g_id * group_size : (row_g_id + 1) * group_size])

    return group_per_split_token_cast


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


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    m, n = x.shape[-2], x.shape[-1]
    aligned_m = get_tma_aligned_size(m, x.element_size())
    if x.dim() == 2:
        if x.stride(0) == 1 and x.stride(1) == aligned_m:
            return x
        x, remove_dim = x.unsqueeze(0), True

    b = x.shape[0]

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def ref_per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def ref_program(x: torch.Tensor, batch_sizes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # assert x.shape[0] == batch_sizes.sum()
    M_max = ceil_div(batch_sizes.max(), 128) * 128
    split_x = torch.split(x, batch_sizes.tolist(), dim=0)
    padded_x = [torch.nn.functional.pad(t, (0, 0, 0, M_max - t.shape[0])) for t in split_x]
    num_groups, m, n = batch_sizes.shape[0], M_max, x.shape[1]
    x_fp8 = (
        torch.empty((num_groups, m, n), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, m, n // 128), device="cuda", dtype=torch.float),
    )
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = ref_per_token_cast_to_fp8(padded_x[i])
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8


def main(M=8192, N=8192, BG=2, blk_m=8, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [2048, 6144]
    if dtype == T.float:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    elif dtype == T.float16:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    elif dtype == T.bfloat16:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    batch_sizes = torch.tensor(batch_sizes, device="cuda", dtype=torch.int32)
    M_max = int(ceil_div(batch_sizes.max(), 128) * 128)

    print("batch_sizes:", batch_sizes)
    print("M_max:", M_max)

    kernel = group_per_split_token_cast_to_fp8(M, M_max, N, BG, blk_m)
    print(kernel.get_kernel_source())
    # profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    x_fp8, x_amax = kernel(x, batch_sizes)
    x_fp8_ref, x_amax_ref = ref_program(x, batch_sizes)

    torch_assert_close(x_fp8.to(torch.float32), x_fp8_ref.to(torch.float32), rtol=0.01, atol=0.01)
    torch_assert_close(x_amax, x_amax_ref, rtol=0.01, atol=0.01)
    print("All checks pass.")

    from tilelang.profiler import do_bench

    def run_tilelang():
        x_fp8_tilelang_, x_amax_tilelang_ = kernel(x, batch_sizes)
        return x_fp8_tilelang_, x_amax_tilelang_

    def run_torch():
        x_fp8_torch_, x_amax_torch_ = ref_program(x, batch_sizes)
        return x_fp8_torch_, x_amax_torch_

    latency = do_bench(run_tilelang)
    print("Tile-lang: {:.2f} ms".format(latency))

    latency = do_bench(run_torch)
    print("Torch: {:.2f} ms".format(latency))


def run_regression_perf(M=8192, N=8192, BG=2, blk_m=8, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [2048, 6144]
    if dtype == "float":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    elif dtype == "float16":
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    elif dtype == "bfloat16":
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    batch_sizes = torch.tensor(batch_sizes, device="cuda", dtype=torch.int32)
    M_max = int(ceil_div(batch_sizes.max(), 128) * 128)

    kernel = group_per_split_token_cast_to_fp8(M, M_max, N, BG, blk_m)

    x_fp8, x_amax = kernel(x, batch_sizes)
    x_fp8_ref, x_amax_ref = ref_program(x, batch_sizes)

    torch_assert_close(x_fp8.to(torch.float32), x_fp8_ref.to(torch.float32), rtol=0.01, atol=0.01)
    torch_assert_close(x_amax, x_amax_ref, rtol=0.01, atol=0.01)

    from tilelang.profiler import do_bench

    def run_tilelang():
        kernel(x, batch_sizes)

    return do_bench(run_tilelang, backend="cupti")


if __name__ == "__main__":
    main()
