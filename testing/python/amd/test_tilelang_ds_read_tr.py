"""Tests for ds_read_tr16_b64 and ds_read_tr8_b64 intrinsics on gfx950.

Covers:
  - Codegen: generated HIP source contains the correct tl:: call.
  - Runtime: kernel compiles and executes on gfx950 without errors.

ds_read_tr16_b64  – LDS transpose read, 64-bit, 16-element transpose.
                    Used for FP16/BF16 MFMA B-loads on MI350/MI355X (gfx950).
ds_read_tr8_b64   – LDS transpose read, 64-bit, 8-element transpose.
                    Used for FP32 MFMA B-loads on MI350/MI355X (gfx950).
"""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.target import target_is_gfx950, determine_target


def requires_gfx950():
    """Skip the test when the current ROCm target is not gfx950."""
    target = determine_target("auto", return_object=True)
    if not target_is_gfx950(target):
        pytest.skip("gfx950 (MI350/MI355X) not detected")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


# ds_read_tr16_b64: each thread reads 2 fp16 elements from LDS with a
# 16-element transpose and stores the result (as float32x2) into a staging
# shared buffer, which is then copied to global memory.
@tilelang.jit(target="hip")
def _kernel_tr16(X, Out):
    NV = T.const("NV")
    X: T.Tensor[[NV], T.float16]
    Out: T.Tensor[[NV // 2], T.float32]

    with T.Kernel(1, threads=NV // 2) as _:
        smem = T.alloc_shared([NV], T.float16)
        smem2 = T.alloc_shared([NV // 2], T.float32)
        T.copy(X[:NV], smem[:NV])
        T.sync_threads()
        for i in T.Parallel(NV // 2):
            val = T.reinterpret(T.ds_read_tr16_b64(smem[i * 2]), T.float32x2)
            smem2[i * 2 : i * 2 + 2] = val
        T.sync_threads()
        T.copy(smem2[: NV // 2], Out[: NV // 2])


# ds_read_tr8_b64: same pattern but reads float32 elements.
@tilelang.jit(target="hip")
def _kernel_tr8(X, Out):
    NV = T.const("NV")
    X: T.Tensor[[NV], T.float32]
    Out: T.Tensor[[NV // 2], T.float32]

    with T.Kernel(1, threads=NV // 2) as _:
        smem = T.alloc_shared([NV], T.float32)
        smem2 = T.alloc_shared([NV // 2], T.float32)
        T.copy(X[:NV], smem[:NV])
        T.sync_threads()
        for i in T.Parallel(NV // 2):
            val = T.reinterpret(T.ds_read_tr8_b64(smem[i * 2]), T.float32x2)
            smem2[i * 2 : i * 2 + 2] = val
        T.sync_threads()
        T.copy(smem2[: NV // 2], Out[: NV // 2])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N = 128  # number of fp16 elements in shared memory


@tilelang.testing.requires_rocm
def test_ds_read_tr16_b64_codegen():
    """Generated HIP source must contain tl::ds_read_tr16_b64(...)."""
    requires_gfx950()

    src = _kernel_tr16.get_kernel_source(NV=N)
    print("=== ds_read_tr16_b64 codegen ===")
    print(src)
    assert "ds_read_tr16_b64" in src, "Expected tl::ds_read_tr16_b64 call in generated HIP source"


@tilelang.testing.requires_rocm
def test_ds_read_tr8_b64_codegen():
    """Generated HIP source must contain tl::ds_read_tr8_b64(...)."""
    requires_gfx950()

    src = _kernel_tr8.get_kernel_source(NV=N)
    print("=== ds_read_tr8_b64 codegen ===")
    print(src)
    assert "ds_read_tr8_b64" in src, "Expected tl::ds_read_tr8_b64 call in generated HIP source"


@tilelang.testing.requires_rocm
def test_ds_read_tr16_b64_runtime():
    """ds_read_tr16_b64 kernel must execute without error on gfx950."""
    requires_gfx950()

    X = torch.randn(N, dtype=torch.float16, device="cuda")
    Out = torch.empty(N // 2, dtype=torch.float32, device="cuda")
    _kernel_tr16(X, Out)
    torch.cuda.synchronize()


@tilelang.testing.requires_rocm
def test_ds_read_tr8_b64_runtime():
    """ds_read_tr8_b64 kernel must execute without error on gfx950."""
    requires_gfx950()

    X = torch.randn(N, dtype=torch.float32, device="cuda")
    Out = torch.empty(N // 2, dtype=torch.float32, device="cuda")
    _kernel_tr8(X, Out)
    torch.cuda.synchronize()


if __name__ == "__main__":
    tilelang.testing.main()
