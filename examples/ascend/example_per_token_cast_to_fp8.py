"""Group-wise FP8 activation quantization with SIMT and explicit SIMD.

Each token's groups of 128 values share a scale. Compare the fragment reduction
in per_token_cast_simt with the vector-register operations in per_token_cast_simd.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench
from tilelang.utils.tensor import torch_assert_close
from tilelang.transform import PassConfigKey


@tilelang.jit(out_idx=[1, 2], pass_configs={PassConfigKey.TL_ENABLE_FAST_MATH: True})
def per_token_cast_simt(M, N):
    if M <= 0 or N <= 0 or M % 32 or N % 128:
        raise ValueError("SIMT quantization expects M % 32 == 0 and N % 128 == 0")
    dtype = T.float32
    group_size = 128
    fp8_max = 448.0
    num_groups = M * (N // group_size)
    # Keep enough tiles to occupy all cores.
    if num_groups >= 64 * 128 and num_groups % 128 == 0:
        block_groups = 128
    elif num_groups >= 64 * 64 and num_groups % 64 == 0:
        block_groups = 64
    else:
        block_groups = 32
    num_tiles = num_groups // block_groups
    N_CORES = min(64, num_tiles)
    NUM_STAGES = 2

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((M, N), dtype), X_fp8: T.Tensor((M, N), T.float8_e4m3fn), X_amax: T.Tensor((M, T.ceildiv(N, group_size)), dtype)
    ):
        # Adjacent groups, including across tokens, use contiguous DMA transfers.
        x_groups = T.Tensor((num_groups, group_size), dtype, X.data)
        q_groups = T.Tensor((num_groups, group_size), T.float8_e4m3fn, X_fp8.data)
        scales = T.Tensor((num_groups,), dtype, X_amax.data)
        with T.Kernel(N_CORES) as core_id:
            for tile in T.Persistent([num_tiles], N_CORES, core_id, num_stages=NUM_STAGES):
                y_ub = T.alloc_shared((block_groups, group_size), dtype)
                y_q_ub_fp8 = T.alloc_shared((block_groups, group_size), T.float8_e4m3fn)
                y_s_ub = T.alloc_shared((block_groups,), dtype)
                T.annotate_buffer_versions({y_ub: NUM_STAGES, y_q_ub_fp8: NUM_STAGES, y_s_ub: NUM_STAGES})
                T.copy(x_groups[tile * block_groups : (tile + 1) * block_groups, :], y_ub)
                with T.SimtVF(threads=1024):
                    y_local = T.alloc_fragment((block_groups, group_size), dtype)
                    y_abs_local = T.alloc_fragment((block_groups, group_size), dtype)
                    y_amax_local = T.alloc_fragment((block_groups,), dtype)
                    y_s_local = T.alloc_fragment((block_groups,), dtype)
                    y_q_local = T.alloc_fragment((block_groups, group_size), dtype)
                    y_q_local_fp8 = T.alloc_fragment((block_groups, group_size), T.float8_e4m3fn)
                    # A warp loads two 64-value strips per group. Each lane owns
                    # a float2 in each strip, giving contiguous accesses in UB.
                    T.annotate_layout(
                        {
                            y_local: T.Fragment(
                                (block_groups, group_size),
                                forward_thread_fn=lambda i, j: (i % 32) * 32 + (j // 2) % 32,
                                forward_index_fn=lambda i, j: (i // 32) * 4 + (j // 64) * 2 + j % 2,
                            )
                        }
                    )
                    T.copy(y_ub, y_local)
                    # Explicit abs lowers to fabsf in the SIMT VF.
                    for i, j in T.Parallel(block_groups, group_size):
                        y_abs_local[i, j] = T.abs(y_local[i, j])
                    T.reduce_max(y_abs_local, y_amax_local, dim=1)
                    for i in T.Parallel(block_groups):
                        y_amax_local[i] = T.max(y_amax_local[i], 0.0001)
                        y_s_local[i] = y_amax_local[i] / fp8_max
                    for i, j in T.Parallel(block_groups, group_size):
                        y_q_local[i, j] = y_local[i, j] / y_s_local[i]
                    # The FP8 conversion saturates values at the finite limits.
                    T.copy(y_q_local, y_q_local_fp8)
                    T.copy(y_s_local, y_s_ub)
                    T.copy(y_q_local_fp8, y_q_ub_fp8)
                T.copy(y_s_ub, scales[tile * block_groups : (tile + 1) * block_groups])
                T.copy(y_q_ub_fp8, q_groups[tile * block_groups : (tile + 1) * block_groups, :])

    return per_token_cast


def per_token_cast_simd(M, N):
    dtype = T.float32
    group_size = 128
    fp8_max = 448.0
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

    @tilelang.jit(out_idx=[1, 2], pass_configs={PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _build():
        @T.prim_func
        def per_token_cast(
            X: T.Tensor((M, N), dtype), X_fp8: T.Tensor((M, N), T.float8_e4m3fn), X_amax: T.Tensor((M, T.ceildiv(N, group_size)), dtype)
        ):
            with T.Kernel(N_CORES) as core_id:
                for row, row_g_id in T.Persistent(
                    [T.ceildiv(M, blk_m), T.ceildiv(num_groups, group_block)],
                    N_CORES,
                    core_id,
                    group_size=T.ceildiv(num_groups, group_block),
                    num_stages=NUM_STAGES,
                ):
                    y_ub = T.alloc_shared((blk_m, tile_n), dtype)
                    y_q_ub_fp8 = T.alloc_shared((blk_m, tile_n), T.float8_e4m3fn)
                    y_s_ub = T.alloc_shared((blk_m, group_block), dtype)
                    T.annotate_buffer_versions({y_ub: NUM_STAGES, y_q_ub_fp8: NUM_STAGES, y_s_ub: NUM_STAGES})
                    T.copy(X[row * blk_m : (row + 1) * blk_m, row_g_id * tile_n : (row_g_id + 1) * tile_n], y_ub)
                    # VF-call cycles: CANN 9.2 / npusim Ascend950, one AIV.
                    # Maximum over group_block=8/16/32/64 with inputs ready in UB.
                    with T.SimdVF(latency=1873):
                        eps = T.simd.vdup(0.0001, "float32")
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
                    T.copy(y_s_ub, X_amax[row * blk_m : (row + 1) * blk_m, row_g_id * group_block : (row_g_id + 1) * group_block])
                    T.copy(y_q_ub_fp8, X_fp8[row * blk_m : (row + 1) * blk_m, row_g_id * tile_n : (row_g_id + 1) * tile_n])

        return per_token_cast

    return _build()


def ref_program(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    new_n = tilelang.cdiv(n, 128) * 128
    x_padded = torch.nn.functional.pad(x, (0, new_n - n))
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(0.0001)
    x_fp8 = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(m, -1)[:, :n].contiguous()
    return (x_fp8, (x_amax / 448.0).view(m, -1))


def per_token_cast_to_fp8(M, N, mode="simd"):
    if mode == "simd":
        return per_token_cast_simd(M, N)
    if mode == "simt":
        return per_token_cast_simt(M, N)
    raise ValueError(f"Unknown VF mode: {mode}")


def run_regression_perf(M=8192, N=8192, mode="simd"):
    kernel = per_token_cast_to_fp8(M, N, mode)
    x = torch.randn(M, N, dtype=torch.float32, device="npu")
    latency = do_bench(lambda: kernel(x), backend="msprof", _n_warmup=30, _n_repeat=50)
    # Read FP32 input; write FP8 values and one FP32 scale per group of 128.
    num_bytes = M * N * (x.element_size() + 1) + M * tilelang.cdiv(N, 128) * 4
    bandwidth = num_bytes / (latency * 1e6)
    print(f"fp8_quantization ({mode}, M={M}, N={N}): {latency * 1000:.2f} us | {bandwidth:.2f} GB/s")
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "simt", "simd"], default="all")
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    x = torch.randn(128, 1024, dtype=torch.float32, device="npu")
    expected_quant, expected_scale = ref_program(x)
    for mode in ("simt", "simd") if args.mode == "all" else (args.mode,):
        quant, scale = per_token_cast_to_fp8(*x.shape, mode=mode)(x)
        # FP32 rounding can place values on either side of an FP8 midpoint.
        torch_assert_close(quant.float(), expected_quant.float(), rtol=0.01, atol=0.01)
        torch.testing.assert_close(scale, expected_scale, rtol=1e-5, atol=1e-7)
        print(f"fp8_quantization ({mode}): correctness passed")
        if args.bench:
            run_regression_perf(mode=mode)
