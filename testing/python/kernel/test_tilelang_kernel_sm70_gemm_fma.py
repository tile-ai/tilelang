import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_bf16_ss_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.bfloat16),
        B: T.Tensor((16, 16), T.bfloat16),
        C: T.Tensor((64, 16), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.bfloat16)
            B_shared = T.alloc_shared((16, 16), T.bfloat16)
            C_local = T.alloc_fragment((64, 16), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)

    return main


def _make_bf16_ss_and_rs_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.bfloat16),
        B: T.Tensor((16, 16), T.bfloat16),
        C: T.Tensor((64, 16), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.bfloat16)
            B_shared = T.alloc_shared((16, 16), T.bfloat16)
            P = T.alloc_fragment((64, 16), T.float32)
            P_bf16 = T.alloc_fragment((64, 16), T.bfloat16)
            C_local = T.alloc_fragment((64, 16), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)

            T.clear(P)
            T.gemm(A_shared, B_shared, P, policy=T.GemmWarpPolicy.FullRow)
            T.copy(P, P_bf16)

            T.clear(C_local)
            T.gemm(P_bf16, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 0)
def test_sm70_bf16_ss_gemm_uses_fma_fallback():
    kernel = tilelang.compile(_make_bf16_ss_gemm(), target="cuda", out_idx=[2])
    source = kernel.get_kernel_source()
    assert "mma_sync_sm70<tl::DataType::kBFloat16" not in source

    a = torch.randn((64, 16), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float()

    tilelang.testing.torch_assert_close(c, ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 0)
def test_sm70_bf16_rs_gemm_uses_fma_fallback():
    kernel = tilelang.compile(_make_bf16_ss_and_rs_gemm(), target="cuda", out_idx=[2])
    source = kernel.get_kernel_source()
    assert "mma_sync_sm70<tl::DataType::kBFloat16" not in source

    a = torch.randn((64, 16), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16) * 0.25
    c = kernel(a, b)
    p = (a.float() @ b.float()).to(torch.bfloat16)
    ref = p.float() @ b.float()

    tilelang.testing.torch_assert_close(c, ref, rtol=2e-2, atol=2e-2, max_mismatched_ratio=0.01)


if __name__ == "__main__":
    tilelang.testing.main()
