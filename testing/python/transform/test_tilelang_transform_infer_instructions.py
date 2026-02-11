import re
import torch
import tilelang
import tilelang.language as T
from tilelang.tileop.gemm import GemmInst
import tilelang.testing
import pytest


def _kernel_uses_mma(kernel_src: str, use_v1: bool) -> bool:
    """Detect whether the kernel uses MMA (vs WGMMA) from generated source.

    - GemmPy (use_v2): emits explicit mma_sync intrinsics.
    - C++ path (use_v1): uses tl::gemm_ss with use_wgmma template param;
      use_wgmma=false means MMA, use_wgmma=true means WGMMA.
    """
    if not use_v1:
        return "mma_sync" in kernel_src
    # C++ path: parse gemm_ss template args; 13th param (index 12) is use_wgmma
    match = re.search(r"gemm_ss\s*<\s*([^>]+)\s*>", kernel_src)
    if not match:
        return False
    params = [p.strip() for p in match.group(1).split(",")]
    return len(params) >= 13 and params[12] == "false"


@tilelang.jit
def gemm(A, B, block_M=128, block_N=128, block_K=128, dtype=T.float16, accum_dtype=T.float32, use_v1=False, use_mma=False):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    ann = {"instruction": GemmInst.MMA} if use_mma else {}

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            if use_v1:
                T.gemm_v1(A_shared, B_shared, C_local, annotations=ann)
            else:
                T.gemm_v2(A_shared, B_shared, C_local, annotations=ann)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.testing.requires_cuda()
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize("use_v1", [True, False])
@pytest.mark.parametrize("use_mma", [True, False])
def test_annotate_instructions(use_v1, use_mma):
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    ref_C = A @ B
    C = gemm(A, B, use_v1=use_v1, use_mma=use_mma)

    assert torch.allclose(C, ref_C, atol=1e-2, rtol=1e-2)
    kernel_src = gemm.get_kernel_source(A, B, use_v1=use_v1, use_mma=use_mma)
    uses_mma = _kernel_uses_mma(kernel_src, use_v1)
    if use_mma:
        assert uses_mma, "expected MMA codegen"
    else:
        assert not uses_mma, "expected WGMMA codegen"


if __name__ == "__main__":
    tilelang.testing.main()
