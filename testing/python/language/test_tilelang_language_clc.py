import torch

import tilelang
import tilelang.language as T
import tilelang.testing

from examples.gemm_sm100.gemm_tcgen5mma_ws_clc import gemm_clc, gemm_clc_2cta


def _make_args(M=256, N=256, K=128):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    common = (128, 256, 64, T.bfloat16, T.bfloat16, T.float, 2)
    return a, b, common


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_clc_source_codegen():
    a, b, common = _make_args()
    src = gemm_clc.get_kernel_source(a, b, *common)
    assert "tl::clc_try_cancel" in src
    assert "tl::clc_is_canceled" in src
    assert "tl::clc_get_first_ctaid_x" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_clc_multicast_source_codegen():
    a, b, common = _make_args()
    src = gemm_clc_2cta.get_kernel_source(a, b, *common)
    assert "tl::clc_try_cancel_multicast" in src
    assert "tl::clc_is_canceled" in src


if __name__ == "__main__":
    tilelang.testing.main()
