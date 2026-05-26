# python maint/gemm/correctness_evaluation_tcgen05_fp4_unpacked.py
"""E2E correctness for tcgen05 f8f6f4 GEMM with float4_e2m1_unpacked SMEM.

Cases:
  1. fp4 (packed global, unpacked SMEM) x fp8 tcgen05 GEMM
  2. mxfp4 (unpacked SMEM) x mxfp8 block-scaled tcgen05 GEMM
"""

import sys
import tilelang
import tilelang.language as T


def _require_sm100() -> None:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required")
        sys.exit(0)
    major, _minor = torch.cuda.get_device_capability()
    if major != 10:
        print(f"SKIP: SM100 (major=10) required, got capability ({major}, {_minor})")
        sys.exit(0)


def fp4_unpacked_x_fp8_gemm(M: int, N: int, K: int):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
        B: T.Tensor((N, K), T.float8_e4m3fn),
        D: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.float4_e2m1_unpacked)
            B_shared = T.alloc_shared((N, K), T.float8_e4m3fn)
            C_tmem = T.alloc_tmem((M, N), T.float32)
            loaded = T.alloc_barrier(128)
            consumed = T.alloc_barrier(1)
            C_local = T.alloc_fragment((M, N), T.float32)

            T.tma_copy(A[0:M, 0:K], A_shared, barrier=loaded)
            T.copy(B[0:N, 0:K], B_shared)
            T.mbarrier_arrive(loaded)
            T.mbarrier_wait_parity(loaded, 0)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                mbar=consumed,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(consumed, 0)
            T.sync_threads()
            T.copy(C_tmem, C_local)
            T.copy(C_local, D[0:M, 0:N])

    return main


def blockscaled_mxfp4_unpacked_x_mxfp8_gemm(M: int, N: int, K: int):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
        B: T.Tensor((N, K), T.float8_e4m3fn),
        SFA: T.Tensor((M,), T.uint32),
        SFB: T.Tensor((N,), T.uint32),
        D: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.float4_e2m1_unpacked)
            B_shared = T.alloc_shared((N, K), T.float8_e4m3fn)
            SFA_shared = T.alloc_shared((M,), T.uint32)
            SFB_shared = T.alloc_shared((N,), T.uint32)
            C_tmem = T.alloc_tmem((M, N), T.float32)
            SFA_tmem = T.alloc_tmem((M, 4), T.uint32)
            SFB_tmem = T.alloc_tmem((M, 4), T.uint32)
            loaded = T.alloc_barrier(32)
            with_sf_full = T.alloc_barrier(32)
            consumed = T.alloc_barrier(1)
            C_local = T.alloc_fragment((M, N), T.float32)
            tx = T.get_thread_binding()

            if tx < 32:
                T.tma_copy(A[0:M, 0:K], A_shared, barrier=loaded)
                T.copy(B[0:N, 0:K], B_shared)
                T.copy(SFA, SFA_shared)
                T.copy(SFB, SFB_shared)
                T.mbarrier_arrive(loaded)
            elif tx < 64:
                T.mbarrier_wait_parity(loaded, 0)
                T.mbarrier_wait_parity(with_sf_full, 0)
                T.tcgen05_cp_warpx4(SFA_shared, SFA_tmem)
                T.tcgen05_cp_warpx4(SFB_shared, SFB_tmem)
                T.tcgen05_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    transpose_B=True,
                    mbar=consumed,
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=128,
                    sf_b_granularity_k=128,
                )
            elif tx < 96:
                T.mbarrier_wait_parity(loaded, 0)
                T.tcgen05_sf_warp_transpose(SFA_shared)
                T.tcgen05_sf_warp_transpose(SFB_shared)
                T.fence_proxy_async()
                T.mbarrier_arrive(with_sf_full)

            T.mbarrier_wait_parity(consumed, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, D[0:M, 0:N])

    return main


def _quantize_mxfp8_with_packed_ue8m0(x, gran_k: int = 128):
    import torch

    def ceil_div_int(xv, yv):
        return (xv + yv - 1) // yv

    def align_up(xv, yv):
        return ceil_div_int(xv, yv) * yv

    def ceil_to_ue8m0(t):
        bits = t.abs().float().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
        return (exp.clamp(1, 254) << 23).view(torch.float32)

    mn, k = x.shape
    padded_k = align_up(k, gran_k)
    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.view(mn, padded_k)[:, :k].contiguous()
    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_blocks = sf_u8.shape[1]
    sf_k_padded = align_up(sf_k_blocks, 4)
    if sf_k_padded != sf_k_blocks:
        sf_u8_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_u8_padded[:, :sf_k_blocks] = sf_u8
    else:
        sf_u8_padded = sf_u8
    words = sf_u8_padded.to(torch.int64)
    packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32)
    sf_packed_u32 = packed.T.contiguous().reshape(-1)
    return x_fp8, sf_packed_u32


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


def _fp4_lut(device: str = "cuda"):
    import torch

    return torch.tensor(_FP4_E2M1_VALUES, device=device, dtype=torch.float32)


def _packed_fp4_bytes_to_f32(packed, k: int):
    """Decode packed fp4 bytes (m, k//2) to logical (m, k) float32."""
    import torch

    u = packed.contiguous().view(torch.uint8)
    lut = _fp4_lut(u.device)
    lo = lut[(u & 0x0F).long()]
    hi = lut[((u >> 4) & 0x0F).long()]
    logical = torch.empty(u.shape[0], k, device=u.device, dtype=torch.float32)
    logical[:, 0::2] = lo
    logical[:, 1::2] = hi
    return logical


def _quantize_f32_to_packed_fp4(x):
    """Quantize float32 (m, k) to packed fp4 bytes (m, k//2)."""
    import torch

    m, k = x.shape
    assert k % 2 == 0
    lut = _fp4_lut(x.device)
    idx = (x.reshape(-1, 1) - lut.reshape(1, -1)).abs().argmin(dim=1).reshape(m, k)
    lo = idx[:, 0::2].to(torch.uint8)
    hi = idx[:, 1::2].to(torch.uint8)
    return (lo | (hi << 4)).to(torch.int8)


def _random_fp4_tensor(m: int, k: int, device: str = "cuda"):
    """Create random packed fp4 tensor for logical shape (m, k)."""
    import torch

    return torch.randint(-128, 128, (m, k // 2), device=device, dtype=torch.int8)


def _quantize_mxfp4_with_packed_ue8m0(x, gran_k: int = 128):
    import torch

    def ceil_div_int(xv, yv):
        return (xv + yv - 1) // yv

    def align_up(xv, yv):
        return ceil_div_int(xv, yv) * yv

    def ceil_to_ue8m0(t):
        bits = t.abs().float().view(torch.int32)
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
        return (exp.clamp(1, 254) << 23).view(torch.float32)

    mn, k = x.shape
    padded_k = align_up(k, gran_k)
    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    x_fp4 = _quantize_f32_to_packed_fp4(x_scaled.reshape(mn, padded_k))[:, : k // 2].contiguous()
    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_blocks = sf_u8.shape[1]
    sf_k_padded = align_up(sf_k_blocks, 4)
    if sf_k_padded != sf_k_blocks:
        sf_u8_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_u8_padded[:, :sf_k_blocks] = sf_u8
    else:
        sf_u8_padded = sf_u8
    words = sf_u8_padded.to(torch.int64)
    packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32)
    sf_packed_u32 = packed.T.contiguous().reshape(-1)
    return x_fp4, sf_packed_u32


def _unpack_sf_u32_1d(packed_sf, mn, sf_k_blocks):
    import torch

    sf_k_groups = (sf_k_blocks + 3) // 4
    packed_2d = packed_sf.view(sf_k_groups, mn).T.contiguous().to(torch.int64)
    unpacked = torch.empty((mn, sf_k_groups * 4), device=packed_sf.device, dtype=torch.uint8)
    for i in range(4):
        unpacked[:, i::4] = ((packed_2d >> (8 * i)) & 0xFF).to(torch.uint8)
    return unpacked[:, :sf_k_blocks].contiguous()


def _blockscaled_gemm_ref(a, b, sfa_packed, sfb_packed, *, sf_granularity_k=128, transpose_B=False):
    import torch

    m, k_packed = a.shape
    k = k_packed * 2
    if transpose_B:
        n, k2 = b.shape
    else:
        k2, n = b.shape
    assert k == k2
    sf_k_blocks = (k + sf_granularity_k - 1) // sf_granularity_k
    sfa_unpacked = _unpack_sf_u32_1d(sfa_packed, m, sf_k_blocks)
    sfb_unpacked = _unpack_sf_u32_1d(sfb_packed, n, sf_k_blocks)
    a_f32 = _packed_fp4_bytes_to_f32(a, k)
    b_f32 = b.to(torch.float32)
    sfa_scales = torch.pow(2.0, sfa_unpacked.to(torch.float32) - 127.0)
    sfb_scales = torch.pow(2.0, sfb_unpacked.to(torch.float32) - 127.0)
    c = torch.zeros(m, n, device=a.device, dtype=torch.float32)
    for bi in range(sf_k_blocks):
        k_start = bi * sf_granularity_k
        k_end = min(k_start + sf_granularity_k, k)
        a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi : bi + 1]
        if transpose_B:
            b_block = b_f32[:, k_start:k_end] * sfb_scales[:, bi : bi + 1]
            c += a_block @ b_block.T
        else:
            b_block = b_f32[k_start:k_end, :] * sfb_scales[:, bi : bi + 1].T
            c += a_block @ b_block
    return c


def run_fp4_unpacked_x_fp8(M: int = 128, N: int = 256, K: int = 128) -> None:
    import torch

    print(f"=== fp4_unpacked x fp8 tcgen05 GEMM correctness M={M} N={N} K={K} ===")
    program = fp4_unpacked_x_fp8_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    src = kernel.get_kernel_source()
    print(src)
    assert "tcgen05mma_ws_ss" in src
    assert "kFloat4_e2m1fn" in src
    assert "uint8_t*)A_shared" in src

    profiler = kernel.get_profiler()

    torch.manual_seed(0)
    a_fp4 = _random_fp4_tensor(M, K)
    b_fp8 = torch.randint(-128, 128, (N, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)
    input_tensors = [a_fp4, b_fp8]

    def ref_program(A, B):
        a = _packed_fp4_bytes_to_f32(A, K)
        b = B.to(torch.float32)
        return torch.matmul(a, b.T)

    profiler.assert_allclose(ref_program, input_tensors=input_tensors, atol=0.25, rtol=0.25)
    print("assert_allclose passed")


def run_blockscaled_mxfp4_x_mxfp8(M: int = 128, N: int = 256, K: int = 128) -> None:
    import torch

    print(f"=== blockscaled mxfp4_unpacked x mxfp8 tcgen05 GEMM correctness M={M} N={N} K={K} ===")
    program = blockscaled_mxfp4_unpacked_x_mxfp8_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[4],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    src = kernel.get_kernel_source()
    assert "tcgen05mma_blockscaled_ss" in src
    assert "kFloat4_e2m1fn" in src

    profiler = kernel.get_profiler()

    torch.manual_seed(0)
    a_f32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b_f32 = torch.randn(N, K, device="cuda", dtype=torch.float32)
    a_fp4, sfa = _quantize_mxfp4_with_packed_ue8m0(a_f32)
    b_fp8, sfb = _quantize_mxfp8_with_packed_ue8m0(b_f32)
    input_tensors = [a_fp4, b_fp8, sfa, sfb]

    def ref_program(A, B, SFA, SFB):
        return _blockscaled_gemm_ref(A, B, SFA, SFB, transpose_B=True)

    profiler.assert_allclose(ref_program, input_tensors=input_tensors, atol=0.5, rtol=0.5)
    print("assert_allclose passed")


def main() -> None:
    _require_sm100()
    run_fp4_unpacked_x_fp8()
    run_blockscaled_mxfp4_x_mxfp8()
    print("All fp4_unpacked tcgen05 correctness checks passed.")


if __name__ == "__main__":
    main()
