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
def _make_w4a8_tma_matmul_kernel(M, N, K, b_dtype, num_stages=2, *, block_M=64, block_N=64, block_K=128):
    """W4A8 with the TMA producer: A is a packed-fp4 GLOBAL tensor and the
    16U4_ALIGN16B bulk-TMA path unpacks it into the padded smem form (the
    same byte layout the SIMT producer writes - pinned by the equivalence
    test below)."""
    accum_dtype = T.float32
    scale_words = block_K // 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
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
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
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
def test_w4a8_tma_producer_matches_simt_producer_bitwise():
    """Producer-equivalence pin: the 16U4_ALIGN16B TMA unpack must stage
    byte-identical smem to the SIMT padded-group producer, so the two
    kernels' outputs must be bitwise equal on identical inputs."""
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 256
    tma = tilelang.compile(_make_w4a8_tma_matmul_kernel(M, N, K, T.float8_e4m3fn), target="cuda", out_idx=[4])
    simt = tilelang.compile(_make_w4a8_matmul_kernel(M, N, K, T.float8_e4m3fn), target="cuda", out_idx=[4])

    # The TMA producer must actually engage - a silent SIMT fallback would
    # make this test vacuous.
    device_src = tma.get_kernel_source()
    assert "tl::tma_load" in device_src
    host_src = tma.get_host_source()
    assert "__tvm_tensormap_create_tiled" in host_src

    A_bytes = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    B = torch.randint(-4, 5, (N, K), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    sfa_bytes = (torch.randint(-1, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(-1, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    SFA = _pack_scale_words(sfa_bytes)
    SFB = _pack_scale_words(sfb_bytes)

    C_tma = tma(A_bytes.view(torch.int8), B, SFA, SFB)
    C_simt = simt(A_bytes, B, SFA, SFB)
    assert torch.equal(C_tma.view(torch.int32), C_simt.view(torch.int32))

    # And both agree with the dequantized reference at atol=0.
    sa = torch.pow(2.0, sfa_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, sfb_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    ref = (_decode_packed_fp4(A_bytes, M, K) * sa) @ (B.to(torch.float32) * sb).T
    torch.testing.assert_close(C_tma, ref, rtol=0.0, atol=0.0)


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


@simplify_prim_func
def _make_fp4_unpack_copy_kernel(rows, k_global, k_copy):
    @T.prim_func
    def main(
        A: T.Tensor((rows, k_global), T.float4_e2m1fn),
        OUT: T.Tensor((rows,), T.uint8),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((rows, k_copy), T.float4_e2m1_unpacked, scope="shared.dyn")
            T.copy(A[0, 0], A_shared, prefer_instruction="tma")
            for i in T.Parallel(rows):
                OUT[i] = T.reinterpret(T.uint8, A_shared[i, 0])

    return main


@simplify_prim_func
def _make_fp4_unpack_copy_1d_kernel(n):
    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float4_e2m1fn),
        OUT: T.Tensor((1,), T.uint8),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((n,), T.float4_e2m1_unpacked, scope="shared.dyn")
            T.copy(A, A_shared, prefer_instruction="tma")
            OUT[0] = T.reinterpret(T.uint8, A_shared[0])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_fp4_unpack_tma_rejects_non_multiple_of_128_global_dim():
    """The CUDA driver requires globalDim[0] % 128 == 0 for 16U4_ALIGN16B.

    Before the guard this only exploded at kernel launch (runtime tensor-map
    validation); now static shapes fail at compile time with a message
    naming the rule.
    """
    with pytest.raises(Exception, match="multiple of 128 elements"):
        tilelang.compile(_make_fp4_unpack_copy_kernel(128, 192, 128), target="cuda", out_idx=[1])
    # Control: a 128-multiple global dim compiles.
    tilelang.compile(_make_fp4_unpack_copy_kernel(128, 256, 128), target="cuda", out_idx=[1])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_fp4_unpack_copy_never_takes_the_1d_bulk_path():
    """The descriptorless 1D bulk path cannot unpack packed fp4.

    Before the guard, a 1D-shaped unpack copy would classify as BulkLoad1D
    and silently emit a raw byte copy (no unpacking, inconsistent sizing).
    With the guard it is diverted off the 1D path; wherever it lands it
    must either be a descriptor TMA (correct) or fail loudly - never a
    silent 1D byte copy.
    """
    try:
        kernel = tilelang.compile(_make_fp4_unpack_copy_1d_kernel(4096), target="cuda", out_idx=[1])
    except Exception:
        return  # loud failure is acceptable; silent 1D byte copy is not
    src = kernel.get_kernel_source()
    assert "tma_load_1d" not in src and "cp.async.bulk.global.shared" not in src


def _load_w4a8_example():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "examples/gemm_sm120/sm120_w4a8_mxf8f6f4_gemm.py"
    spec = importlib.util.spec_from_file_location("sm120_w4a8_mxf8f6f4_gemm_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _w4a8_example_run(module, M, N, K, b_name="e4m3", seed=0):
    import torch

    kernel = module.sm120_w4a8_mxf8f6f4_gemm(M, N, K, 128, 128, 128, 2, b_name, T.float32)
    A = module._make_packed_fp4(M, K, seed=seed)
    B = module._make_fp8(N, K, b_name, seed=seed + 1)
    SFA_semantic = module._make_pow2_scale_words(M, K, seed=seed + 100)
    SFB_semantic = module._make_pow2_scale_words(N, K, seed=seed + 200)
    SFA = module.swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic, block_words=1).reshape(-1, 1)
    SFB = module.swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic, block_words=1).reshape(-1, 1)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(A, B, SFA, SFB, C)
    return kernel, A, B, SFA_semantic, SFB_semantic, C


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("b_name", ["e4m3", "e5m2"])
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (768, 1024, 1536)])
def test_w4a8_example_synthetic_band_bitwise(b_name, shape):
    """Acceptance A1: default band, square and non-square, atol=0."""
    import torch

    torch.manual_seed(0)
    module = _load_w4a8_example()
    M, N, K = shape
    _, A, B, SFA_semantic, SFB_semantic, C = _w4a8_example_run(module, M, N, K, b_name)
    module._verify(A, B, SFA_semantic, SFB_semantic, C, torch.float32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w4a8_example_quantize_band():
    """Acceptance A2: bf16 -> mxfp4/mxfp8 quantizers -> GEMM, tolerance form.

    The bitwise (engine-vs-engine) version of this band is the maint CUTLASS
    comparison; this pins the CI-runnable plumbing.
    """
    import torch

    from examples.dequantize_gemm.quantize import (
        quantize_bf16_to_mxfp4_blockscaled,
        quantize_bf16_to_mxfp8_blockscaled,
    )

    torch.manual_seed(0)
    module = _load_w4a8_example()
    M = N = K = 512
    kernel = module.sm120_w4a8_mxf8f6f4_gemm(M, N, K, 128, 128, 128, 2, "e4m3", T.float32)
    x_a = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
    x_b = (torch.randn(N, K, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
    A, SFA_packed, sfa_bytes = quantize_bf16_to_mxfp4_blockscaled(x_a, block_words=1, return_scale_bytes=True)
    B, SFB_packed, sfb_bytes = quantize_bf16_to_mxfp8_blockscaled(x_b, block_words=1, return_scale_bytes=True)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(A, B, SFA_packed.reshape(-1, 1), SFB_packed.reshape(-1, 1), C)

    scale = lambda b: torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32)).repeat_interleave(32, dim=1)  # noqa: E731
    ref = (module._decode_rowmajor_fp4(A, M, K) * scale(sfa_bytes)) @ (B.to(torch.float32) * scale(sfb_bytes)).T
    # f32 output: only summation rounding-order noise remains.
    torch.testing.assert_close(C, ref, rtol=1e-5, atol=1e-3)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w4a8_example_tma_engaged_and_compression():
    """Acceptance A3 + A4: TMA actually engaged; packed global compression."""
    module = _load_w4a8_example()
    M = N = K = 256
    kernel, A, _, _, _, _ = _w4a8_example_run(module, M, N, K)
    device_src = kernel.get_kernel_source()
    assert "tl::tma_load" in device_src, "A staging silently fell back off the TMA path"
    assert "tl::ptx_ldmatrix_su4_x4" in device_src
    assert "tl::fp4_e2m1_container_shift" in device_src
    host_src = kernel.get_host_source()
    assert "__tvm_tensormap_create_tiled" in host_src
    # A4: the weights the example materializes are exactly M*K/2 bytes.
    assert A.element_size() * A.numel() == M * K // 2


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w4a8_example_rejects_non_multiple_of_128_k():
    """Acceptance A5: the TMA legality rule surfaces as a friendly error."""
    module = _load_w4a8_example()
    with pytest.raises(Exception, match="multiple of 128"):
        module.sm120_w4a8_mxf8f6f4_gemm(128, 128, 192, 128, 128, 64, 2, "e4m3", T.float32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w4a8_example_perf_smoke():
    """Acceptance A6: the bench path runs and reports a sane number."""
    import torch

    from tilelang.profiler import do_bench

    module = _load_w4a8_example()
    M = N = K = 1024
    kernel, A, B, _, _, C = _w4a8_example_run(module, M, N, K)
    SFA = module.swizzle_blockscaled_chunk_kmajor_scale_words(module._make_pow2_scale_words(M, K, seed=100), block_words=1).reshape(-1, 1)
    SFB = module.swizzle_blockscaled_chunk_kmajor_scale_words(module._make_pow2_scale_words(N, K, seed=200), block_words=1).reshape(-1, 1)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    ms = do_bench(lambda: kernel(A, B, SFA, SFB, C), warmup=10, rep=20)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    assert tflops > 0


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w4a8_kmajor_matches_rowmajor_bitwise_full_domain():
    """Order pin for the W4A8 kmajor fulltile path.

    Full-entropy data (random fp4 bytes x random e5m2 bytes incl. Inf/NaN)
    makes every partial sum order-sensitive; strict integer-view equality
    between the kmajor example kernel (TMA producer) and the rowmajor
    serial kernel (SIMT producer) pins the K-atom accumulation order and
    the NaN payloads across both pipelines and both producers at once.
    """
    import torch

    torch.manual_seed(0)
    M = N = 256
    K = 512
    module = _load_w4a8_example()

    A_bytes = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    B = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8).view(torch.float8_e5m2)
    sfa_bytes = (torch.randint(-1, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(-1, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)

    rowmajor = tilelang.compile(_make_w4a8_matmul_kernel(M, N, K, T.float8_e5m2), target="cuda", out_idx=[4])
    C_row = rowmajor(A_bytes, B, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))

    kernel = module.sm120_w4a8_mxf8f6f4_gemm(M, N, K, 128, 128, 128, 2, "e5m2", T.float32)
    SFA_km = module.swizzle_blockscaled_chunk_kmajor_scale_words(_pack_scale_words(sfa_bytes), block_words=1).reshape(-1, 1)
    SFB_km = module.swizzle_blockscaled_chunk_kmajor_scale_words(_pack_scale_words(sfb_bytes), block_words=1).reshape(-1, 1)
    C_km = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(A_bytes.view(torch.int8), B, SFA_km, SFB_km, C_km)

    assert torch.equal(C_row.view(torch.int32), C_km.view(torch.int32))


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize(
    "sfa_byte,sfb_byte,expected_bits",
    [
        (0xFE, 0xFE, 0x7F800000),
        (0xFE, 0x7F, 0x7F000000),
        (0x00, 0x7F, 0x00400000),
        (0x00, 0x00, 0x00000000),
        (0x01, 0x7F, 0x00800000),
    ],
)
def test_w4a8_extreme_scale_semantics(sfa_byte, sfb_byte, expected_bits):
    """The UE8M0 extreme-scale recordings re-run with an fp4 A operand.

    Same expected table as the pure-fp8 recordings (the scale datapath is
    operand-format independent) - re-measured here rather than assumed.
    """
    import torch

    M = N = 128
    K = 128
    kernel = tilelang.compile(_make_w4a8_matmul_kernel(M, N, K, T.float8_e4m3fn), target="cuda", out_idx=[4])
    A_bytes = torch.zeros((M, K // 2), device="cuda", dtype=torch.uint8)
    A_bytes[0, 0] = 0x02  # element k=0 (low nibble) = e2m1 code for 1.0
    B = torch.zeros((N, K), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    B[0, 0] = 1.0
    sfa = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
    sfb = torch.full((N, K // 32), 127, device="cuda", dtype=torch.uint8)
    sfa[0, 0] = sfa_byte
    sfb[0, 0] = sfb_byte
    C = kernel(A_bytes, B, _pack_scale_words(sfa), _pack_scale_words(sfb))
    assert (C[0, 0].view(torch.int32).item() & 0xFFFFFFFF) == expected_bits
    assert bool((C.flatten()[1:] == 0).all())


_FAMILY_SPECS = {
    # name -> (tilelang smem dtype attr, global form, torch feed)
    "e4m3": ("float8_e4m3fn", "fp8", None),
    "e5m2": ("float8_e5m2", "fp8", None),
    "e2m1": ("float4_e2m1_unpacked", "fp4_blob", None),
    "e2m3": ("float6_e2m3fn_unpacked", "fp6_blob", None),
    "e3m2": ("float6_e3m2fn_unpacked", "fp6_blob", None),
}


@simplify_prim_func
def _make_family_matmul_kernel(M, N, K, a_spec, b_spec, *, block_M=64, block_N=64, block_K=128):
    """Rowmajor kernel over any f8f6f4-family operand pair.

    fp8 operands are normal fp8 tensors; fp4 is a packed uint8 blob (2
    elems/byte -> 8-byte payload per 16-element smem group); fp6 is a packed
    uint8 blob (LSB-first 6-bit stream, 4 elems/3 bytes -> 12-byte payload
    per group). SIMT producers write the b4x16_p64 / b6x16_p32 padded forms.
    """
    accum_dtype = T.float32
    scale_words = block_K // 128

    def global_decl(name, rows, spec):
        form = _FAMILY_SPECS[spec][1]
        if form == "fp8":
            return T.Tensor((rows, K), getattr(T, _FAMILY_SPECS[spec][0]))
        if form == "fp4_blob":
            return T.Tensor((rows, K // 2), T.uint8)
        return T.Tensor((rows, K * 3 // 4), T.uint8)

    a_smem_dtype = getattr(T, _FAMILY_SPECS[a_spec][0])
    b_smem_dtype = getattr(T, _FAMILY_SPECS[b_spec][0])
    a_form = _FAMILY_SPECS[a_spec][1]
    b_form = _FAMILY_SPECS[b_spec][1]

    @T.prim_func
    def main(
        A: global_decl("A", M, a_spec),
        B: global_decl("B", N, b_spec),
        SFA: T.Tensor((M, K // 128), T.uint32),
        SFB: T.Tensor((N, K // 128), T.uint32),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_smem_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), b_smem_dtype, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, scale_words), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, scale_words), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if K // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=1):
                if a_form == "fp8":
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                elif a_form == "fp4_blob":
                    for i, g, j in T.Parallel(block_M, block_K // 16, 8):
                        A_shared[i, 16 * g + j] = T.reinterpret(a_smem_dtype, A[by * block_M + i, ko * (block_K // 2) + 8 * g + j])
                else:
                    for i, g, j in T.Parallel(block_M, block_K // 16, 12):
                        A_shared[i, 16 * g + j] = T.reinterpret(a_smem_dtype, A[by * block_M + i, ko * (block_K * 3 // 4) + 12 * g + j])
                if b_form == "fp8":
                    for i, k in T.Parallel(block_N, block_K):
                        B_shared[i, k] = B[bx * block_N + i, ko * block_K + k]
                elif b_form == "fp4_blob":
                    for i, g, j in T.Parallel(block_N, block_K // 16, 8):
                        B_shared[i, 16 * g + j] = T.reinterpret(b_smem_dtype, B[bx * block_N + i, ko * (block_K // 2) + 8 * g + j])
                else:
                    for i, g, j in T.Parallel(block_N, block_K // 16, 12):
                        B_shared[i, 16 * g + j] = T.reinterpret(b_smem_dtype, B[bx * block_N + i, ko * (block_K * 3 // 4) + 12 * g + j])
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


def _family_data(spec, rows, K, controlled=True):
    """Returns (feed_tensor, decoded_float32) for one operand."""
    import torch

    from examples.dequantize_gemm.quantize import decode_fp6_values, unpack_fp6_bytes

    form = _FAMILY_SPECS[spec][1]
    if form == "fp8":
        tdt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[spec]
        t = torch.randint(-4, 5, (rows, K), device="cuda", dtype=torch.float32).to(tdt)
        return t, t.to(torch.float32)
    if form == "fp4_blob":
        blob = torch.randint(0, 256, (rows, K // 2), device="cuda", dtype=torch.uint8)
        return blob, _decode_packed_fp4(blob, rows, K)
    blob = torch.randint(0, 256, (rows, K * 3 // 4), device="cuda", dtype=torch.uint8)
    codes = unpack_fp6_bytes(blob.cpu(), K)
    return blob, decode_fp6_values(codes, spec).cuda()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize(
    "a_spec,b_spec",
    [
        ("e4m3", "e2m3"),
        ("e4m3", "e3m2"),
        ("e5m2", "e3m2"),
        ("e2m3", "e4m3"),
        ("e3m2", "e3m2"),
        ("e2m3", "e3m2"),
        ("e3m2", "e2m1"),
        ("e2m1", "e2m3"),
    ],
)
def test_fp6_family_rowmajor_correctness(a_spec, b_spec):
    """fp6 operands in every direction: vs fp8, fp6, and fp4 partners."""
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 128
    kernel = tilelang.compile(_make_family_matmul_kernel(M, N, K, a_spec, b_spec), target="cuda", out_idx=[4])
    src = kernel.get_kernel_source()
    enum_of = {"e2m1": "kE2M1", "e2m3": "kE2M3", "e3m2": "kE3M2", "e4m3": "kE4M3", "e5m2": "kE5M2"}
    assert f"tl::SM120MmaOperandType::{enum_of[a_spec]}, tl::SM120MmaOperandType::{enum_of[b_spec]}>" in src
    if "e2m3" in (a_spec, b_spec) or "e3m2" in (a_spec, b_spec):
        assert "tl::ptx_ldmatrix_su6_x" in src

    A, a_dec = _family_data(a_spec, M, K)
    B, b_dec = _family_data(b_spec, N, K)
    sfa_bytes = (torch.randint(0, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(0, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    C = kernel(A, B, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))

    sa = torch.pow(2.0, sfa_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, sfb_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    ref = (a_dec * sa) @ (b_dec * sb).T
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_fp6_flavor_codec_discrimination():
    """e2m3 x e3m2 with values exact in exactly one flavor.

    A carries 7.5-family values (need e2m3's third mantissa bit; e3m2
    rounds them), B carries 16..28 (beyond e2m3's 7.5 max). A swapped
    flavor mnemonic changes the decode and fails the bitwise assert.
    """
    import torch

    from examples.dequantize_gemm.quantize import decode_fp6_values, encode_fp6_values, pack_fp6_codes

    torch.manual_seed(0)
    M = N = 128
    K = 128
    a_vals = torch.tensor([7.5, 6.5, 5.5, -7.5, -6.5, -5.5, 3.75, -3.75], dtype=torch.float32)
    b_vals = torch.tensor([16.0, 20.0, 24.0, 28.0, -16.0, -20.0, -24.0, -28.0], dtype=torch.float32)
    # Self-check the discriminating power under the swapped codec.
    assert not torch.equal(
        decode_fp6_values(encode_fp6_values(a_vals, "e2m3"), "e2m3"), decode_fp6_values(encode_fp6_values(a_vals, "e2m3"), "e3m2")
    )
    assert torch.equal(decode_fp6_values(encode_fp6_values(a_vals, "e2m3"), "e2m3"), a_vals)
    assert torch.equal(decode_fp6_values(encode_fp6_values(b_vals, "e3m2"), "e3m2"), b_vals)

    a_floats = a_vals[torch.randint(0, 8, (M, K))]
    b_floats = b_vals[torch.randint(0, 8, (N, K))]
    A_blob = pack_fp6_codes(encode_fp6_values(a_floats, "e2m3")).cuda()
    B_blob = pack_fp6_codes(encode_fp6_values(b_floats, "e3m2")).cuda()

    kernel = tilelang.compile(_make_family_matmul_kernel(M, N, K, "e2m3", "e3m2"), target="cuda", out_idx=[4])
    unit = torch.full((M, K // 128), 0x7F7F7F7F, device="cuda", dtype=torch.int64).to(torch.uint32)
    C = kernel(A_blob, B_blob, unit, unit.clone())
    ref = a_floats.cuda() @ b_floats.cuda().T
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


def _load_w6a8_example():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "examples/gemm_sm120/sm120_w6a8_mxf8f6f4_gemm.py"
    spec = importlib.util.spec_from_file_location("sm120_w6a8_mxf8f6f4_gemm_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _w6a8_example_run(module, M, N, K, a_name="e4m3", b_flavor="e3m2", seed=0):
    import torch

    kernel = module.sm120_w6a8_mxf8f6f4_gemm(M, N, K, 128, 128, 128, 2, a_name, b_flavor, T.float32)
    A = module._make_fp8(M, K, a_name, seed=seed)
    B = module._make_packed_fp6(N, K, seed=seed + 1)
    SFA_semantic = module._make_pow2_scale_words(M, K, seed=seed + 100)
    SFB_semantic = module._make_pow2_scale_words(N, K, seed=seed + 200)
    SFA = module.swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic, block_words=1).reshape(-1, 1)
    SFB = module.swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic, block_words=1).reshape(-1, 1)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(A, B, SFA, SFB, C)
    return kernel, A, B, SFA_semantic, SFB_semantic, C


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("a_name,b_flavor", [("e4m3", "e3m2"), ("e5m2", "e3m2"), ("e4m3", "e2m3")])
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (768, 1024, 1536)])
def test_w6a8_example_synthetic_band_bitwise(a_name, b_flavor, shape):
    """W6A8 acceptance A1: default band, square and non-square, atol=0."""
    import torch

    torch.manual_seed(0)
    module = _load_w6a8_example()
    M, N, K = shape
    _, A, B, SFA_semantic, SFB_semantic, C = _w6a8_example_run(module, M, N, K, a_name, b_flavor)
    module._verify(A, B, SFA_semantic, SFB_semantic, C, torch.float32, b_flavor)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w6a8_example_su6_engaged_and_compression():
    """W6A8 acceptance A3 + A4: TMA + su6 engaged (no shift), 0.75 B/elem."""
    module = _load_w6a8_example()
    M = N = K = 256
    kernel, _, B, _, _, _ = _w6a8_example_run(module, M, N, K)
    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src  # 16U6_ALIGN16B producer engaged
    assert "tl::ptx_ldmatrix_su6_x" in src
    assert "tl::fp4_e2m1_container_shift" not in src  # fp6 containers need no shift
    assert ", tl::SM120MmaOperandType::kE3M2>" in src
    assert B.element_size() * B.numel() == N * K * 3 // 4


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w6a8_example_rejects_non_multiple_of_128_k():
    """W6A8 acceptance A5."""
    module = _load_w6a8_example()
    with pytest.raises(Exception, match="multiple of 128"):
        module.sm120_w6a8_mxf8f6f4_gemm(128, 128, 192, 128, 128, 64, 2, "e4m3", "e3m2", T.float32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_w6a8_example_quantize_band():
    """W6A8 acceptance A2: mxfp8 + mxfp6 quantizers end to end (CI form)."""
    import torch

    from examples.dequantize_gemm.quantize import (
        quantize_bf16_to_mxfp6_blockscaled,
        quantize_bf16_to_mxfp8_blockscaled,
    )

    torch.manual_seed(0)
    module = _load_w6a8_example()
    M = N = K = 512
    kernel = module.sm120_w6a8_mxf8f6f4_gemm(M, N, K, 128, 128, 128, 2, "e4m3", "e3m2", T.float32)
    x_a = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
    x_b = torch.randn(N, K, dtype=torch.float32) * 2.0
    A, SFA_packed, sfa_bytes = quantize_bf16_to_mxfp8_blockscaled(x_a, block_words=1, return_scale_bytes=True)
    B_cpu, SFB_packed, sfb_bytes = quantize_bf16_to_mxfp6_blockscaled(x_b, dtype="e3m2", block_words=1, return_scale_bytes=True)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(A, B_cpu.cuda(), SFA_packed.reshape(-1, 1), SFB_packed.cuda().reshape(-1, 1), C)

    scale = lambda b: torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32)).repeat_interleave(32, dim=1)  # noqa: E731
    b_dec = module._decode_fp6_blob(B_cpu.cuda(), N, K, "e3m2")
    ref = (A.to(torch.float32) * scale(sfa_bytes)) @ (b_dec * scale(sfb_bytes).cuda()).T
    torch.testing.assert_close(C, ref, rtol=1e-5, atol=1e-3)


@simplify_prim_func
def _make_a8w6_tma_matmul_kernel(M, N, K, a_dtype, num_stages=2, *, block_M=64, block_N=64, block_K=128):
    """A8W6 with the 16U6_ALIGN16B TMA producer on the fp6 B operand.

    B is declared as a packed float6 GLOBAL tensor (fed as a uint8 blob of
    N*K*3/4 bytes through the sub-byte binder bypass); the TMA unpacks it
    into the padded smem form the su6 ldmatrix consumes.
    """
    accum_dtype = T.float32
    scale_words = block_K // 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), a_dtype),
        B: T.Tensor((N, K), T.float6_e3m2fn),
        SFA: T.Tensor((M, K // 128), T.uint32),
        SFB: T.Tensor((N, K // 128), T.uint32),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), T.float6_e3m2fn_unpacked, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, scale_words), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, scale_words), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if K // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
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
def test_a8w6_tma_producer_matches_simt_producer_bitwise():
    """S9 acceptance: the 16U6_ALIGN16B TMA unpack must stage byte-identical
    smem to the SIMT padded-group fp6 producer."""
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 256
    tma = tilelang.compile(_make_a8w6_tma_matmul_kernel(M, N, K, T.float8_e4m3fn), target="cuda", out_idx=[4])
    device_src = tma.get_kernel_source()
    assert "tl::tma_load" in device_src
    assert "tl::ptx_ldmatrix_su6_x" in device_src

    simt = tilelang.compile(_make_family_matmul_kernel(M, N, K, "e4m3", "e3m2"), target="cuda", out_idx=[4])

    A = torch.randint(-4, 5, (M, K), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    B_blob = torch.randint(0, 256, (N, K * 3 // 4), device="cuda", dtype=torch.uint8)
    sfa_bytes = (torch.randint(0, 2, (M, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    sfb_bytes = (torch.randint(0, 2, (N, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)
    SFA = _pack_scale_words(sfa_bytes)
    SFB = _pack_scale_words(sfb_bytes)

    C_tma = tma(A, B_blob.view(torch.int8), SFA, SFB)
    C_simt = simt(A, B_blob, SFA, SFB)
    assert torch.equal(C_tma.view(torch.int32), C_simt.view(torch.int32))


if __name__ == "__main__":
    tilelang.testing.main()
