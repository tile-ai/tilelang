"""fp4/fp6 operands inside kind::mxf8f6f4 (W4A8-style mixed GEMMs).

Global storage stays PACKED (fp4: 2 elems/byte - the compression invariant);
shared memory holds the 16U4_ALIGN16B padded-packed form (16 elements = 8B
payload + 8B padding), which the su4 ldmatrix unpacks into 8-bit register
containers, shifted to the bits[5:2] position the MMA expects.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import simplify_prim_func


_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@simplify_prim_func
def _make_w4a8_matmul_kernel(M, N, K, b_dtype, num_stages=1, *, block_M=64, block_N=64, block_K=128):
    accum_dtype = T.float32
    scale_words = block_K // 128

    @T.prim_func
    def main(
        A_packed: T.Tensor((M, K // 2), T.uint8),
        B: T.Tensor((N, K), b_dtype),
        SFA: T.Tensor((M, K // 128), T.uint32),
        SFB: T.Tensor((N, K // 128), T.uint32),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float4_e2m1_unpacked, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), b_dtype, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, scale_words), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, scale_words), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if K // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=num_stages):
                # SIMT producer of the 16U4_ALIGN16B padded-packed form:
                # bytes [0..7] of each 16-slot group carry the 16 packed
                # nibbles, bytes [8..15] are padding the ldmatrix ignores.
                for i, g, j in T.Parallel(block_M, block_K // 16, 8):
                    A_shared[i, 16 * g + j] = T.reinterpret(
                        T.float4_e2m1_unpacked, A_packed[by * block_M + i, ko * (block_K // 2) + 8 * g + j]
                    )
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]
                for i, w in T.Parallel(block_M, scale_words):
                    SFA_shared[i, w] = SFA[by * block_M + i, ko * scale_words + w]
                for j, w in T.Parallel(block_N, scale_words):
                    SFB_shared[j, w] = SFB[bx * block_N + j, ko * scale_words + w]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=(K // block_K == 1),
                    k_start=0,
                    sf_a_granularity_k=32,
                    sf_b_granularity_k=32,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _decode_packed_fp4(packed, rows, cols):
    import torch

    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(packed & 0x0F).long()]
    out[:, 1::2] = lut[((packed >> 4) & 0x0F).long()]
    return out


def _pack_scale_words(scale_bytes):
    import torch

    s = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    w = s[:, :, 0] | (s[:, :, 1] << 8) | (s[:, :, 2] << 16) | (s[:, :, 3] << 24)
    return w.to(torch.uint32).contiguous()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("b_name", ["e4m3", "e5m2"])
@pytest.mark.parametrize("K,block_K", [(128, 128), (256, 128), (512, 256)])
def test_w4a8_mxf8f6f4_rowmajor_correctness(b_name, K, block_K):
    import torch

    torch.manual_seed(0)
    M = N = 128
    b_dtype = {"e4m3": T.float8_e4m3fn, "e5m2": T.float8_e5m2}[b_name]
    kernel = tilelang.compile(_make_w4a8_matmul_kernel(M, N, K, b_dtype, block_K=block_K), target="cuda", out_idx=[4])
    src = kernel.get_kernel_source()
    assert "tl::ptx_ldmatrix_su4_x4" in src
    assert "tl::fp4_e2m1_container_shift" in src
    assert "tl::SM120MmaOperandType::kE2M1, tl::SM120MmaOperandType::k" in src

    A_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    B = torch.randint(-4, 5, (N, K), device="cuda", dtype=torch.float32).to(
        {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[b_name]
    )
    sfa_bytes = (torch.randint(-1, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(-1, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)

    C = kernel(A_packed, B, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))

    sa = torch.pow(2.0, sfa_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, sfb_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    ref = (_decode_packed_fp4(A_packed, M, K) * sa) @ (B.to(torch.float32) * sb).T
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


@simplify_prim_func
def _make_a8w4_matmul_kernel(M, N, K, a_dtype, *, block_M=64, block_N=64, block_K=128, a_is_fp4=False):
    """A fp8 (or fp4 when a_is_fp4) x B packed-fp4: exercises the B-side su4 path."""
    accum_dtype = T.float32
    scale_words = block_K // 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), a_dtype),
        B_packed: T.Tensor((N, K // 2), T.uint8),
        SFA: T.Tensor((M, K // 128), T.uint32),
        SFB: T.Tensor((N, K // 128), T.uint32),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1_unpacked, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, scale_words), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, scale_words), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if K // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=1):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, g, b in T.Parallel(block_N, block_K // 16, 8):
                    B_shared[j, 16 * g + b] = T.reinterpret(
                        T.float4_e2m1_unpacked, B_packed[bx * block_N + j, ko * (block_K // 2) + 8 * g + b]
                    )
                for i, w in T.Parallel(block_M, scale_words):
                    SFA_shared[i, w] = SFA[by * block_M + i, ko * scale_words + w]
                for j, w in T.Parallel(block_N, scale_words):
                    SFB_shared[j, w] = SFB[bx * block_N + j, ko * scale_words + w]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=(K // block_K == 1),
                    k_start=0,
                    sf_a_granularity_k=32,
                    sf_b_granularity_k=32,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("a_name", ["e4m3", "e5m2"])
def test_a8w4_mxf8f6f4_rowmajor_correctness(a_name):
    """B-side su4 path: fp8 activations x packed-fp4 weights."""
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 256
    a_dtype = {"e4m3": T.float8_e4m3fn, "e5m2": T.float8_e5m2}[a_name]
    kernel = tilelang.compile(_make_a8w4_matmul_kernel(M, N, K, a_dtype), target="cuda", out_idx=[4])
    src = kernel.get_kernel_source()
    assert "tl::ptx_ldmatrix_su4_x4" in src
    assert ", tl::SM120MmaOperandType::kE2M1>" in src

    A = torch.randint(-4, 5, (M, K), device="cuda", dtype=torch.float32).to(
        {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[a_name]
    )
    B_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    sfa_bytes = (torch.randint(-1, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(-1, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)

    C = kernel(A, B_packed, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))

    sa = torch.pow(2.0, sfa_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, sfb_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    ref = (A.to(torch.float32) * sa) @ (_decode_packed_fp4(B_packed, N, K) * sb).T
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


@simplify_prim_func
def _make_w4a4_k32_matmul_kernel(M, N, K, *, block_M=64, block_N=64, block_K=128):
    """Both operands packed fp4 under kind::mxf8f6f4 (k32 atoms, 32-elem scales)."""
    accum_dtype = T.float32
    scale_words = block_K // 128

    @T.prim_func
    def main(
        A_packed: T.Tensor((M, K // 2), T.uint8),
        B_packed: T.Tensor((N, K // 2), T.uint8),
        SFA: T.Tensor((M, K // 128), T.uint32),
        SFB: T.Tensor((N, K // 128), T.uint32),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float4_e2m1_unpacked, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1_unpacked, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, scale_words), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, scale_words), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if K // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=1):
                for i, g, b in T.Parallel(block_M, block_K // 16, 8):
                    A_shared[i, 16 * g + b] = T.reinterpret(
                        T.float4_e2m1_unpacked, A_packed[by * block_M + i, ko * (block_K // 2) + 8 * g + b]
                    )
                for j, g, b in T.Parallel(block_N, block_K // 16, 8):
                    B_shared[j, 16 * g + b] = T.reinterpret(
                        T.float4_e2m1_unpacked, B_packed[bx * block_N + j, ko * (block_K // 2) + 8 * g + b]
                    )
                for i, w in T.Parallel(block_M, scale_words):
                    SFA_shared[i, w] = SFA[by * block_M + i, ko * scale_words + w]
                for j, w in T.Parallel(block_N, scale_words):
                    SFB_shared[j, w] = SFB[bx * block_N + j, ko * scale_words + w]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=(K // block_K == 1),
                    k_start=0,
                    sf_a_granularity_k=32,
                    sf_b_granularity_k=32,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w4a4_mxf8f6f4_rowmajor_correctness():
    """fp4 x fp4 under kind::mxf8f6f4: unpacked containers on BOTH operands.

    Note this is a different instruction than the packed mxf4nvf4 kind
    (k32 vs k64 atoms), so its oracle is the python reference, not the
    mxf4nvf4 path.
    """
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 256
    kernel = tilelang.compile(_make_w4a4_k32_matmul_kernel(M, N, K), target="cuda", out_idx=[4])
    src = kernel.get_kernel_source()
    assert "tl::SM120MmaOperandType::kE2M1, tl::SM120MmaOperandType::kE2M1>" in src

    A_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    B_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    sfa_bytes = (torch.randint(-1, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(-1, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)

    C = kernel(A_packed, B_packed, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))

    sa = torch.pow(2.0, sfa_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, sfb_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    ref = (_decode_packed_fp4(A_packed, M, K) * sa) @ (_decode_packed_fp4(B_packed, N, K) * sb).T
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    tilelang.testing.main()
