import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_gemm_kernel(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _assert_mma_sync_shape(source, a_type, c_type, m, n, k):
    expected = f"tl::mma_sync<tl::DataType::{a_type}, tl::DataType::{a_type}, tl::DataType::{c_type}, {m}, {n}, {k}, false, true>"
    assert expected in source
    assert "mma_sync_sm70" not in source


def _pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    tensor_i16 = tensor.to(torch.int16)
    packed = (tensor_i16[..., ::2] & 0x0F) | ((tensor_i16[..., 1::2] & 0x0F) << 4)
    return packed.to(torch.int8).contiguous()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_f16_gemm_uses_m16n8k8_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 32, T.float16, T.float32, T.float32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kFloat16", "kFloat32", 16, 8, 8)

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref = a.float() @ b.float().T

    tilelang.testing.torch_assert_close(c, ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_int8_gemm_uses_m8n8k16_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 64, T.int8, T.int32, T.int32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kInt8", "kInt32", 8, 8, 16)

    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
    c = kernel(a, b)
    ref = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda")

    tilelang.testing.torch_assert_close(c, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_int4_gemm_uses_m8n8k32_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 64, T.int4, T.int32, T.int32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kInt4", "kInt32", 8, 8, 32)

    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
    c = kernel(_pack_int4(a), _pack_int4(b))
    ref = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda")

    tilelang.testing.torch_assert_close(c, ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
