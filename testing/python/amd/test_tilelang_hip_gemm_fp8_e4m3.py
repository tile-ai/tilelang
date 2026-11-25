"""Regression tests for T.gemm with ``float8_e4m3`` operands on ROCm (MFMA)."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.language.fp8 import determine_fp8_type


def _gemm(M, N, K, dtype):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((M, K), dtype)
            B_shared = T.alloc_shared((K, N), dtype)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    return main


@pytest.mark.parametrize("mcpu", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["float8_e4m3", "float8_e4m3fn", "float8_e4m3fnuz"])
def test_gemm_fp8_e4m3_lower(mcpu, dtype):
    """All E4M3 names lower to the 16x16x32 fp8 MFMA (no GPU needed)."""
    if tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang HIP codegen is not enabled")
    target = {"kind": "hip", "mcpu": mcpu}
    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        artifact = tilelang.lower(_gemm(64, 64, 32, dtype), target=target)
    src = artifact.kernel_source
    assert "__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(*(((long*)" in src


@tilelang.testing.requires_cdna
@pytest.mark.parametrize("M, N, K", [(64, 64, 32), (128, 128, 128)])
def test_gemm_fp8_e4m3_correctness(M, N, K):
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith(("gfx94", "gfx95")):
        pytest.skip(f"FP8 MFMA requires gfx94x/gfx950, got {arch}")
    kernel = tilelang.compile(_gemm(M, N, K, T.float8_e4m3))
    # fp8_e4_t is the device-native E4M3 encoding (FNUZ on gfx942, OCP on gfx950).
    fp8 = getattr(torch, str(determine_fp8_type("e4m3")))
    A = torch.randn(M, K, device="cuda").to(fp8)
    B = torch.randn(K, N, device="cuda").to(fp8)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel(A, B, C)
    ref = A.to(torch.float32) @ B.to(torch.float32)
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
