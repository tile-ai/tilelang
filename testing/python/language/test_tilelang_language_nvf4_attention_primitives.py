"""White-box pins for the SM120 FP4 attention building blocks (SageAttention3 port).

(a) e2m1 m16n8k64 A-operand register order: the ldmatrix.x4 output (r0=(g,k lo), r1=(g+8,k lo),
    r2=(g,k hi), r3=(g+8,k hi), PTX a0..a3 convention) must equal both the documented inverse
    layout function and the forward fragment layout used by the register-A (rs) path.
(c) Whole-word writes into a packed fp4 shared tile through T.view(uint32) land on the expected
    nibbles (the v1 kernel writes P-hat this way, avoiding nibble read-modify-write races).
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.intrinsics.layout import mma_layout as ml

torch = pytest.importorskip("torch")


def test_fp4_a_fragment_layout_round_trip():
    """forward (m,k)->(lane,local) must invert the CUTLASS-pinned inverse for all 16x64 elements."""
    bad = [
        (i, j)
        for i in range(16)
        for j in range(64)
        if ml.mma_load_a_32x32_to_shared_16x64_layout(*ml.shared_16x64_to_mma_a_32x32_layout(i, j)) != (i, j)
    ]
    assert bad == [], f"{len(bad)} mismatches, first {bad[:4]}"
    # and the sr alias used by make_mma_load_layout is the same function
    assert ml.shared_16x64_to_mma_32x32_layout_sr_a(3, 45) == ml.shared_16x64_to_mma_a_32x32_layout(3, 45)


def _make_ldmatrix_a_e2m1_probe():
    @T.prim_func
    def main(
        SRC: T.Tensor((512,), T.uint8),  # 16 rows x 64 nibbles, row-major, 32 bytes/row
        OUT: T.Tensor((32, 16), T.uint8),  # per lane: 4 u32 = 16 bytes
    ):
        with T.Kernel(1, threads=32) as _:
            tx = T.get_thread_binding()
            smem = T.alloc_shared((512,), T.uint8, scope="shared.dyn")
            regs = T.alloc_local((16,), T.uint8)
            for i in T.serial(16):
                smem[tx * 16 + i] = SRC[tx * 16 + i]
            T.sync_threads()
            # Same lane addressing as TensorCoreIntrinEmitterSM120._warp_ld_a_e2m1:
            # row = tx % 16, nibble col = (tx // 16) * 32 -> byte = row * 32 + (tx // 16) * 16
            T.ptx_ldmatrix(
                T.bool(False),
                4,
                T.access_ptr(smem[(tx % 16) * 32 + (tx // 16) * 16], "r", extent=16),
                T.access_ptr(regs[0], "w", extent=16),
            )
            for i in T.serial(16):
                OUT[tx, i] = regs[i]

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_ldmatrix_a_e2m1_register_order_matches_layout_functions():
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (16, 64), dtype=torch.uint8)
    src = (codes[:, 0::2] | (codes[:, 1::2] << 4)).reshape(-1).contiguous().cuda()
    kernel = tilelang.compile(_make_ldmatrix_a_e2m1_probe(), target="cuda", out_idx=[1])
    out = kernel(src).cpu()
    for lane in range(32):
        for r in range(4):
            for t in range(8):
                nib = int(out[lane, 4 * r + t // 2]) >> (4 * (t % 2)) & 0xF
                local_id = 8 * r + t
                row, col = ml.mma_load_a_32x32_to_shared_16x64_layout(lane, local_id)
                assert (row, col) == (lane // 4 + 8 * (r % 2), (lane % 4) * 8 + t + 32 * (r // 2))
                assert nib == int(codes[row, col]), (lane, r, t, row, col)
                assert ml.shared_16x64_to_mma_a_32x32_layout(row, col) == (lane, local_id)


def _make_packed_fp4_word_write_probe():
    @T.prim_func
    def main(
        W: T.Tensor((16, 8), T.uint32),
        OUT: T.Tensor((16, 64), T.float4_e2m1fn),
    ):
        with T.Kernel(1, threads=32) as _:
            P_sh = T.alloc_shared((16, 64), T.float4_e2m1fn, scope="shared.dyn")
            P_u32 = T.view(P_sh, (16, 8), dtype=T.uint32)
            for i, w in T.Parallel(16, 8):
                P_u32[i, w] = W[i, w]
            T.copy(P_sh, OUT)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_packed_fp4_shared_word_writes_land_on_expected_nibbles():
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (16, 64), dtype=torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()  # [16, 32] bytes
    words = packed.view(torch.uint32).reshape(16, 8).cuda()  # 64 nibbles = 32 bytes = 8 words per row
    kernel = tilelang.compile(_make_packed_fp4_word_write_probe(), target="cuda", out_idx=[1])
    out = kernel(words)
    got = out.view(torch.uint8).cpu().reshape(16, 32)
    assert torch.equal(got, packed)


def _make_rs_vs_ss_kernel(M, N, K, threads):
    """Same NVFP4 blockscaled GEMM twice: A from shared (ss) and A as a uint32 word fragment
    (rs, a_packed_words) filled from the same shared tile; both accumulate into separate outputs."""
    fp4 = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((M, K), fp4),
        B: T.Tensor((N, K), fp4),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C_ss: T.Tensor((M, N), T.float32),
        C_rs: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_sh = T.alloc_shared((M, K), fp4)
            B_sh = T.alloc_shared((N, K), fp4)
            SFA_sh = T.alloc_shared((M, K // 64), T.uint32)
            SFB_sh = T.alloc_shared((N, K // 64), T.uint32)
            A_words = T.view(A_sh, (M, K // 8), dtype=T.uint32)
            Aw = T.alloc_fragment((M, K // 8), T.uint32)
            acc_ss = T.alloc_fragment((M, N), T.float32)
            acc_rs = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_sh)
            T.copy(B, B_sh)
            T.copy(SFA, SFA_sh)
            T.copy(SFB, SFB_sh)
            T.clear(acc_ss)
            T.mma_gemm_blockscaled(
                A_sh,
                B_sh,
                acc_ss,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
            )
            for i, w in T.Parallel(M, K // 8):
                Aw[i, w] = A_words[i, w]
            T.clear(acc_rs)
            T.mma_gemm_blockscaled(
                Aw,
                B_sh,
                acc_rs,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
                a_packed_words=True,
            )
            T.copy(acc_ss, C_ss)
            T.copy(acc_rs, C_rs)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("M,N,K,threads", [(16, 32, 64, 32), (32, 32, 128, 32), (128, 128, 128, 256)])
def test_blockscaled_register_a_packed_words_matches_shared_a(M, N, K, threads):
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    # ue4m3 scale bytes in the normal range, packed 4 per word
    sfa_b = torch.randint(0x30, 0x40, (M, K // 16), device="cuda", dtype=torch.uint8).to(torch.int64).reshape(M, K // 64, 4)
    sfb_b = torch.randint(0x30, 0x40, (N, K // 16), device="cuda", dtype=torch.uint8).to(torch.int64).reshape(N, K // 64, 4)
    sfa = (sfa_b[..., 0] | (sfa_b[..., 1] << 8) | (sfa_b[..., 2] << 16) | (sfa_b[..., 3] << 24)).to(torch.uint32).contiguous()
    sfb = (sfb_b[..., 0] | (sfb_b[..., 1] << 8) | (sfb_b[..., 2] << 16) | (sfb_b[..., 3] << 24)).to(torch.uint32).contiguous()
    kernel = tilelang.compile(_make_rs_vs_ss_kernel(M, N, K, threads), target="cuda", out_idx=[4, 5])
    c_ss, c_rs = kernel(a, b, sfa, sfb)
    assert torch.isfinite(c_ss).all() and float(c_ss.abs().max()) > 0
    assert torch.equal(c_ss.view(torch.int32), c_rs.view(torch.int32))
    src = kernel.get_kernel_source()
    assert src.count("sm120_mma_sync_blockscaled") >= 2


def _swizzled_sfb_kernel(M: int, N: int, K: int, threads: int):
    """One tile computed twice: row-major B scales, and the swizzled layout that lets each lane
    load its scale rows contiguously (T.mma_gemm_blockscaled(..., sf_b_swizzled=True))."""
    fp4 = T.float4_e2m1fn
    words = K // 64
    rows = N // 2  # the swizzled buffer holds 4 words (2 original rows) per row
    padded = rows + rows // 8  # one pad row per lane block, to keep the blocks off the same banks

    @T.prim_func
    def main(
        A: T.Tensor((M, K), fp4),
        B: T.Tensor((N, K), fp4),
        SFA: T.Tensor((M, words), T.uint32),
        SFB: T.Tensor((N, words), T.uint32),
        SFB_sw: T.Tensor((N, words), T.uint32),
        C_plain: T.Tensor((M, N), T.float32),
        C_sw: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_sh = T.alloc_shared((M, K), fp4)
            B_sh = T.alloc_shared((N, K), fp4)
            SFA_sh = T.alloc_shared((M, words), T.uint32)
            SFB_sh = T.alloc_shared((N, words), T.uint32)
            SFB_sw_sh = T.alloc_shared((padded, 2 * words), T.uint32)
            acc_plain = T.alloc_fragment((M, N), T.float32)
            acc_sw = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_sh)
            T.copy(B, B_sh)
            T.copy(SFA, SFA_sh)
            T.copy(SFB, SFB_sh)
            for r, w in T.Parallel(rows, 2 * words):
                SFB_sw_sh[(r // 8) * 9 + r % 8, w] = SFB_sw[2 * r + w // words, w % words]
            T.clear(acc_plain)
            T.mma_gemm_blockscaled(
                A_sh,
                B_sh,
                acc_plain,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
            )
            T.clear(acc_sw)
            T.mma_gemm_blockscaled(
                A_sh,
                B_sh,
                acc_sw,
                SFA_sh,
                SFB_sw_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
                sf_b_swizzled=True,
            )
            T.copy(acc_plain, C_plain)
            T.copy(acc_sw, C_sw)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_blockscaled_swizzled_b_scales_match_rowmajor():
    M = N = K = 128
    threads = 256
    kernel = tilelang.jit(out_idx=[5, 6])(_swizzled_sfb_kernel)(M, N, K, threads)
    torch.manual_seed(0)
    a = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")

    def _scale_words(rows, cols):  # four ue4m3 bytes per word, all finite (1.0 .. 3.5)
        by = torch.randint(0x38, 0x42, (rows, cols, 4), dtype=torch.int64, device="cuda")
        return (by[..., 0] | (by[..., 1] << 8) | (by[..., 2] << 16) | (by[..., 3] << 24)).to(torch.uint32)

    sfa = _scale_words(M, K // 64)
    sfb = _scale_words(N, K // 64)
    # row n of the B scales moves to (n % 8) * 16 + n // 8, i.e. gather new[p] = old[(p % 16) * 8 + p // 16]
    p_idx = torch.arange(N, device="cuda")
    sfb_sw = sfb.index_select(0, (p_idx % 16) * 8 + p_idx // 16).contiguous()
    c_plain, c_sw = kernel(a.view(torch.int8), b.view(torch.int8), sfa, sfb, sfb_sw)
    assert torch.equal(c_plain, c_sw), (c_plain - c_sw).abs().max()
    assert c_plain.abs().sum() > 0


if __name__ == "__main__":
    tilelang.testing.main()
