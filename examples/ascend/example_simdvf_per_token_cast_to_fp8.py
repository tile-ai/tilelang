import torch
import tilelang
import tilelang.ascend.language as T
from typing import Tuple
from tilelang.utils.tensor import torch_assert_close

from tilelang.profiler import do_bench
from tilelang.transform import PassConfigKey


def per_token_cast_to_fp8(M, N, backend="asc"):
    dtype = T.float
    group_size = 128
    fp8_max = 448.0
    lanes = 64

    N_CORES = 64
    NUM_STAGES = 2
    num_groups = (N + group_size - 1) // group_size

    if num_groups >= 64 and num_groups % 64 == 0:
        group_block = 64
    elif num_groups >= 32 and num_groups % 32 == 0:
        group_block = 32
    elif num_groups >= 16 and num_groups % 16 == 0:
        group_block = 16
    elif num_groups >= 8 and num_groups % 8 == 0:
        group_block = 8
    else:
        raise ValueError(f"optimized Ascend SimdVF path expects ceildiv(N, 128) to be a multiple of 8, got N={N}")

    blk_m = 128 // group_block
    tile_n = group_size * group_block

    if M % blk_m != 0 or N % group_size != 0:
        raise ValueError(f"optimized Ascend SimdVF path expects M % {blk_m} == 0 and N % {group_size} == 0, got M={M}, N={N}")

    @tilelang.jit(
        out_idx=[1, 2],
        target=backend,
        pass_configs={PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _build():
        @T.prim_func
        def per_token_cast(
            X: T.Tensor((M, N), dtype),
            X_fp8: T.Tensor((M, N), T.float8_e4m3fn),
            X_amax: T.Tensor((M, T.ceildiv(N, group_size)), dtype),
        ):
            with T.Kernel(N_CORES) as core_id:
                for row, row_g_id in T.Persistent(
                    [T.ceildiv(M, blk_m), T.ceildiv(num_groups, group_block)],
                    N_CORES,
                    core_id,
                    group_size=1,
                    num_stages=NUM_STAGES,
                ):
                    y_ub = T.alloc_shared((blk_m, tile_n), dtype)
                    y_q_ub_fp8 = T.alloc_shared((blk_m, tile_n), T.float8_e4m3fn)
                    y_s_ub = T.alloc_shared((blk_m, group_block), dtype)
                    T.annotate_buffer_versions({y_ub: NUM_STAGES, y_q_ub_fp8: NUM_STAGES, y_s_ub: NUM_STAGES})

                    T.copy(
                        X[row * blk_m : (row + 1) * blk_m, row_g_id * tile_n : (row_g_id + 1) * tile_n],
                        y_ub,
                    )
                    with T.SimdVF():
                        if backend == "pto":
                            full = T.vmi.create_mask(lanes, size=lanes)
                            one = T.vmi.create_mask(1, size=1)
                            eps = T.vmi.vbrc(T.float32(1e-4), size=1)
                            fp8_max_reg = T.vmi.vbrc(T.float32(fp8_max), size=1)
                            for i in range(blk_m):
                                for j in range(group_block):
                                    col = j * group_size
                                    x0 = T.vmi.vload(y_ub[i, col], size=lanes)
                                    x1 = T.vmi.vload(y_ub[i, col + lanes], size=lanes)
                                    amax0 = T.vmi.vcmax(T.vmi.vabs(x0, full), full)
                                    amax1 = T.vmi.vcmax(T.vmi.vabs(x1, full), full)
                                    amax = T.vmi.vmax(T.vmi.vmax(amax0, amax1, one), eps, one)
                                    scale = T.vmi.vdiv(amax, fp8_max_reg, one)
                                    T.vmi.vstore(scale, y_s_ub[i, j], stride=1, group=1)
                                    scale_brc = T.vmi.vbrc(scale, size=lanes)
                                    q0 = T.vmi.vcvt(
                                        T.vmi.vdiv(x0, scale_brc, full),
                                        "float8_e4m3fn",
                                        rounding="R",
                                        saturate="SAT",
                                    )
                                    q1 = T.vmi.vcvt(
                                        T.vmi.vdiv(x1, scale_brc, full),
                                        "float8_e4m3fn",
                                        rounding="R",
                                        saturate="SAT",
                                    )
                                    T.vmi.vstore(q0, y_q_ub_fp8[i, col], full)
                                    T.vmi.vstore(q1, y_q_ub_fp8[i, col + lanes], full)
                        else:
                            eps = T.simd.vdup(1e-4, "float32")
                            fp8_max_reg = T.simd.vdup(fp8_max, "float32")

                            for i in range(blk_m):
                                for j in range(group_block):
                                    col = j * group_size
                                    x0 = T.simd.vld(y_ub[i, col])
                                    x1 = T.simd.vld(y_ub[i, col + 64])
                                    abs0 = T.simd.vabs(x0)
                                    abs1 = T.simd.vabs(x1)
                                    amax_0 = T.simd.vmax(abs0, abs1)
                                    amax_1 = T.simd.vcmax(amax_0)
                                    amax = T.simd.vmax(amax_1, eps)
                                    scale = T.simd.vdiv(amax, fp8_max_reg)
                                    scale_brc = T.simd.vdupv(scale)
                                    T.simd.vsts(y_s_ub[i, j], scale, dist="ONEPT_B32")

                                    q0 = T.simd.vdiv(x0, scale_brc)
                                    q0_fp8 = T.simd.vcvt(q0, "float8_e4m3fn")
                                    T.simd.vsts(y_q_ub_fp8[i, col], q0_fp8, dist="PK4_B32")

                                    q1 = T.simd.vdiv(x1, scale_brc)
                                    q1_fp8 = T.simd.vcvt(q1, "float8_e4m3fn")
                                    T.simd.vsts(y_q_ub_fp8[i, col + 64], q1_fp8, dist="PK4_B32")

                    T.copy(
                        y_s_ub,
                        X_amax[
                            row * blk_m : (row + 1) * blk_m,
                            row_g_id * group_block : (row_g_id + 1) * group_block,
                        ],
                    )
                    T.copy(
                        y_q_ub_fp8,
                        X_fp8[
                            row * blk_m : (row + 1) * blk_m,
                            row_g_id * tile_n : (row_g_id + 1) * tile_n,
                        ],
                    )

        return per_token_cast

    return _build()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def ref_program(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    new_n = ceil_div(n, 128) * 128
    x_padded = torch.nn.functional.pad(x, (0, new_n - n))
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    x_fp8 = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(m, -1)[:, :n].contiguous()
    return x_fp8, (x_amax / 448.0).view(m, -1)


def effective_io_gb(m: int, n: int) -> float:
    total_bytes = m * n * 4 + m * n + m * ceil_div(n, 128) * 4
    return total_bytes / 1e9


def print_latency_bandwidth(name: str, latency_ms: float, io_gb: float) -> None:
    bandwidth_gbs = io_gb / (latency_ms / 1e3)
    print(f"{name}: {latency_ms:.2f} ms | {bandwidth_gbs:.2f} GB/s")


def test(M=8192, N=8192, *, backend="asc", print_source=False):
    kernel = per_token_cast_to_fp8(M, N, backend)
    if print_source:
        print(kernel.get_kernel_source())

    x = torch.randn(M, N, device="cpu", dtype=torch.float32).to("npu")

    x_fp8, x_amax = kernel(x)
    x_fp8_ref, x_amax_ref = ref_program(x)

    torch_assert_close(x_fp8.to(torch.float32), x_fp8_ref.to(torch.float32), rtol=0.01, atol=0.01)
    torch_assert_close(x_amax, x_amax_ref, rtol=0.01, atol=0.01)
    print("All checks pass.")


def main(M=8192, N=8192, backend="asc"):
    test(M, N, backend=backend, print_source=True)
    run_regression_perf(M, N, backend=backend)


def run_regression_perf(M=8192, N=8192, backend="asc"):
    kernel = per_token_cast_to_fp8(M, N, backend)
    x = torch.randn(M, N, device="cpu", dtype=torch.float32).to("npu")
    io_gb = effective_io_gb(M, N)

    def run_kernel_only():
        kernel(x)

    latency_ms = do_bench(run_kernel_only, backend="msprof", _n_warmup=30, _n_repeat=100)
    print_latency_bandwidth("Tile-lang", latency_ms, io_gb)
    return latency_ms


if __name__ == "__main__":
    main()
