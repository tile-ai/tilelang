# Schedule adapted from DeepGEMM.

import argparse
from typing import Tuple
import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@tilelang.jit(out_idx=[4])
def fp8_fp4_gemm_1d1d_persistent(
    M: int,
    N: int,
    K: int,
    out_dtype,
):
    block_M = 128
    block_N = 256
    block_K = 128
    accum_dtype = T.float32
    num_stages = 6
    sf_granularity_k = 128
    store_block_N = 64

    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert sf_granularity_k == 128

    half_N = block_N // 2
    sf_k_groups = _ceil_div(_ceil_div(K, sf_granularity_k), 4)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
        B: T.Tensor((N, K), T.float8_e4m3fn),
        SFA: T.Tensor((sf_k_groups * M,), T.uint32),
        SFB: T.Tensor((sf_k_groups * N,), T.uint32),
        D: T.Tensor((M, N), out_dtype),
    ):
        sm_num = driver.get_num_sms()
        num_clusters = sm_num // 2
        m_blocks = T.ceildiv(M, block_M)
        m_clusters = m_blocks // 2
        n_blocks = T.ceildiv(N, block_N)
        k_iters = T.ceildiv(K, block_K)
        sf_load_period = sf_granularity_k * 4 // block_K
        waves = T.ceildiv(m_blocks * n_blocks, sm_num)
        group_size = 16

        with T.ClusterKernel(sm_num, threads=256, cluster_dims=2) as (block_id):
            cta_id = T.block_rank_in_cluster()
            T.assume(cta_id < 2)

            # Use T.float4_e2m1_unpacked for A_shared, rather than T.float4_e2m1fn as a hint
            # for TileLang to use desired layout for blockscaled tcgen5mma.
            A_shared = T.alloc_shared((num_stages, block_M, block_K), T.float4_e2m1_unpacked)
            B_shared = T.alloc_shared((num_stages, half_N, block_K), T.float8_e4m3fn)
            SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
            SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            SFA_tmem = T.alloc_tmem([block_M, 4], "uint32")
            SFB_tmem = T.alloc_tmem([block_M, 8], "uint32")

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)

            loaded = T.alloc_barrier([32] * num_stages)
            with_sf_full = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_cluster_barrier([1])
            tmem_empty = T.alloc_cluster_barrier([128 * 2])

            tx = T.get_thread_binding()

            if tx < 32:  # issue TMA, load operands and SFs
                for w in T.unroll(waves):
                    cluster_id = block_id // 2
                    tile_id = num_clusters * w + cluster_id
                    bx_cluster = (tile_id // group_size) % m_clusters
                    bx = bx_cluster * 2 + cta_id
                    by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size

                    if bx * block_M < M and by * block_N < N:
                        for k in T.serial(k_iters):
                            phase = w * k_iters + k
                            stage = phase % num_stages
                            parity = (phase // num_stages) & 1
                            T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
                            T.tma_copy(
                                A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                                A_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                            T.tma_copy(
                                B[
                                    by * block_N + cta_id * half_N : by * block_N + (cta_id + 1) * half_N,
                                    k * block_K : (k + 1) * block_K,
                                ],
                                B_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                            if k % sf_load_period == 0:
                                sf_group_idx = k // sf_load_period
                                T.tma_copy(
                                    SFA[sf_group_idx * M + bx * block_M : sf_group_idx * M + (bx + 1) * block_M],
                                    SFA_shared[stage, :],
                                    barrier=loaded[stage],
                                )
                                T.tma_copy(
                                    SFB[sf_group_idx * N + by * block_N : sf_group_idx * N + (by + 1) * block_N],
                                    SFB_shared[stage, :],
                                    barrier=loaded[stage],
                                )
                            T.mbarrier_arrive(loaded[stage])

            elif 32 <= tx < 64 and cta_id == 0:  # issue tcgen5
                for w in T.unroll(waves):
                    cluster_id = block_id // 2
                    tile_id = num_clusters * w + cluster_id

                    if tile_id < m_clusters * n_blocks:
                        T.mbarrier_wait_parity(tmem_empty, (w & 1) ^ 1)
                        for k in T.serial(k_iters):
                            phase = w * k_iters + k
                            stage = phase % num_stages
                            parity = (phase // num_stages) & 1
                            T.mbarrier_wait_parity(with_sf_full[stage], parity)
                            if k % sf_load_period == 0:
                                T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem, use_2cta=True)
                                T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem, use_2cta=True)
                            T.tcgen05_gemm_blockscaled(
                                A_shared[stage, :, :],
                                B_shared[stage, :, :],
                                C_tmem,
                                SFA_tmem,
                                SFB_tmem,
                                transpose_B=True,
                                mbar=consumed[stage],
                                clear_accum=k == 0,
                                k_start=k * block_K,  # global K offset (to help compiler infer sf_id)
                                sf_a_granularity_k=sf_granularity_k,
                                sf_b_granularity_k=sf_granularity_k,
                                use_2cta=True,
                            )
                        T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

            elif 64 <= tx < 96:  # transpose SFs
                for w in T.unroll(waves):
                    cluster_id = block_id // 2
                    tile_id = num_clusters * w + cluster_id

                    if tile_id < m_clusters * n_blocks:
                        for k in T.serial(k_iters):
                            phase = w * k_iters + k
                            stage = phase % num_stages
                            parity = (phase // num_stages) & 1
                            T.mbarrier_wait_parity(loaded[stage], parity)
                            if k % sf_load_period == 0:
                                T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                                T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                                T.fence_proxy_async()
                            T.mbarrier_arrive(with_sf_full[stage], 0)

            elif 128 <= tx < 256:  # epilogue
                for w in T.unroll(waves):
                    cluster_id = block_id // 2
                    tile_id = num_clusters * w + cluster_id
                    bx_cluster = (tile_id // group_size) % m_clusters
                    bx = bx_cluster * 2 + cta_id
                    by = (tile_id % group_size) + (tile_id // group_size) // m_clusters * group_size

                    if bx * block_M < M and by * block_N < N:
                        T.mbarrier_wait_parity(tmem_full, w & 1)
                        T.copy(C_tmem, C_local)
                        T.mbarrier_arrive(tmem_empty, 0)

                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                            T.copy(C_shared, D[bx * block_M, by * block_N + i * store_block_N])

    return main


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def pack_sf_u8_to_u32_1d(sf_u8: torch.Tensor) -> torch.Tensor:
    assert sf_u8.dtype == torch.uint8
    assert sf_u8.dim() == 2
    mn, sf_k_padded = sf_u8.shape
    assert sf_k_padded % 4 == 0
    words = sf_u8.to(torch.int64)
    packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


def unpack_sf_u32_1d(packed_sf: torch.Tensor, mn: int, sf_k_blocks: int) -> torch.Tensor:
    sf_k_groups = _ceil_div(sf_k_blocks, 4)
    packed_2d = packed_sf.view(sf_k_groups, mn).T.contiguous().to(torch.int64)
    unpacked = torch.empty((mn, sf_k_groups * 4), device=packed_sf.device, dtype=torch.uint8)
    for i in range(4):
        unpacked[:, i::4] = ((packed_2d >> (8 * i)) & 0xFF).to(torch.uint8)
    return unpacked[:, :sf_k_blocks].contiguous()


_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_FP4_E2M1_VALUES, device=device, dtype=torch.float32)


def _packed_fp4_to_float(packed: torch.Tensor, logical_k: int) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    if u.shape[1] != logical_k // 2:
        u = u[:, : logical_k // 2]
    lut = _fp4_lut(u.device)
    lo = lut[(u & 0x0F).long()]
    hi = lut[((u >> 4) & 0x0F).long()]
    out = torch.empty((u.shape[0], logical_k), device=u.device, dtype=torch.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def _quantize_float_to_fp4_packed(x: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    assert k % 2 == 0
    lut = _fp4_lut(x.device)
    idx = (x.reshape(-1, 1) - lut.reshape(1, -1)).abs().argmin(dim=1).reshape(m, k)
    lo = idx[:, 0::2].to(torch.uint8)
    hi = idx[:, 1::2].to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def quantize_mxfp4_with_packed_ue8m0(x: torch.Tensor, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert x.size(1) % 2 == 0
    mn, k = x.shape
    padded_k = _align_up(k, gran_k)
    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)
    amax = x_view.abs().float().amax(dim=2).clamp_min(6.0 * (2.0**-126))
    sf = _ceil_to_ue8m0(amax / 6.0)
    x_fp4 = _quantize_float_to_fp4_packed((x_view * (1.0 / sf.unsqueeze(2))).reshape(mn, padded_k))[:, : k // 2].contiguous()
    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_padded = _align_up(sf_u8.shape[1], 4)
    if sf_k_padded != sf_u8.shape[1]:
        sf_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_padded[:, : sf_u8.shape[1]] = sf_u8
    else:
        sf_padded = sf_u8
    return x_fp4, pack_sf_u8_to_u32_1d(sf_padded), sf_u8


def quantize_mxfp8_with_packed_ue8m0(x: torch.Tensor, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    mn, k = x.shape
    padded_k = _align_up(k, gran_k)
    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)
    amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(mn, padded_k)[:, :k].contiguous()
    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_padded = _align_up(sf_u8.shape[1], 4)
    if sf_k_padded != sf_u8.shape[1]:
        sf_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_padded[:, : sf_u8.shape[1]] = sf_u8
    else:
        sf_padded = sf_u8
    return x_fp8, pack_sf_u8_to_u32_1d(sf_padded), sf_u8


def fp8_fp4_gemm_ref(a_fp4, b_fp8, sfa_packed, sfb_packed, sf_granularity_k=128):
    m, k_packed = a_fp4.shape
    k = k_packed * 2
    n, k2 = b_fp8.shape
    assert k == k2
    sf_k_blocks = _ceil_div(k, sf_granularity_k)
    sfa = unpack_sf_u32_1d(sfa_packed, m, sf_k_blocks)
    sfb = unpack_sf_u32_1d(sfb_packed, n, sf_k_blocks)
    a_f32 = _packed_fp4_to_float(a_fp4, k)
    b_f32 = b_fp8.to(torch.float32)
    sfa_scales = torch.pow(2.0, sfa.to(torch.float32) - 127.0)
    sfb_scales = torch.pow(2.0, sfb.to(torch.float32) - 127.0)

    c = torch.zeros((m, n), device=a_fp4.device, dtype=torch.float32)
    for bi in range(sf_k_blocks):
        k_start = bi * sf_granularity_k
        k_end = min(k_start + sf_granularity_k, k)
        a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi : bi + 1]
        b_block = b_f32[:, k_start:k_end] * sfb_scales[:, bi : bi + 1]
        c += a_block @ b_block.T
    return c


def cosine_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.double().flatten()
    y = y.double().flatten()
    return 1.0 - 2.0 * (x * y).sum() / ((x * x + y * y).sum())


def run_fp8_fp4_gemm_1d1d(
    a_fp4,
    b_fp8,
    sfa,
    sfb,
    out_dtype=T.bfloat16,
):
    block_M, block_N, block_K = 128, 256, 128
    m, k_packed = a_fp4.shape
    n, k = b_fp8.shape
    assert k_packed * 2 == k
    assert m % (2 * block_M) == 0, f"M={m} must be divisible by {2 * block_M}"
    assert n % (32 * block_N) == 0, f"N={n} must be divisible by {32 * block_N}"
    assert k % (2 * block_K) == 0, f"K={k} must be divisible by {2 * block_K}"
    kernel = fp8_fp4_gemm_1d1d_persistent(m, n, k, out_dtype)
    return kernel(a_fp4, b_fp8, sfa, sfb)


def check_correctness(M=256, N=8192, K=1024, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((N, K), device="cuda", dtype=torch.float32)
    a_fp4, sfa, _ = quantize_mxfp4_with_packed_ue8m0(a)
    b_fp8, sfb, _ = quantize_mxfp8_with_packed_ue8m0(b)

    c = run_fp8_fp4_gemm_1d1d(a_fp4, b_fp8, sfa, sfb)
    ref = fp8_fp4_gemm_ref(a_fp4, b_fp8, sfa, sfb).to(torch.bfloat16)
    diff = cosine_diff(c, ref)
    max_abs_diff = (c.float() - ref.float()).abs().max()
    print(f"cosine diff={float(diff):.6g}, max abs diff={float(max_abs_diff):.6g}")
    assert diff < 1e-3
    return c


def benchmark(M=512, N=8192, K=8192, seed=0, warmup=25, rep=100):
    torch.manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((N, K), device="cuda", dtype=torch.float32)
    a_fp4, sfa, _ = quantize_mxfp4_with_packed_ue8m0(a)
    b_fp8, sfb, _ = quantize_mxfp8_with_packed_ue8m0(b)

    torch.cuda.synchronize()
    latency = do_bench(lambda: run_fp8_fp4_gemm_1d1d(a_fp4, b_fp8, sfa, sfb), warmup=warmup, rep=rep)
    tflops = 2 * M * N * K / latency / 1e9
    print(f"M={M} N={N} K={K}: {latency:.4f} ms, {tflops:.2f} TFLOP/s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    check_correctness(args.M, args.N, args.K, args.seed)
    benchmark(args.M, args.N, args.K, args.seed)


if __name__ == "__main__":
    main()
