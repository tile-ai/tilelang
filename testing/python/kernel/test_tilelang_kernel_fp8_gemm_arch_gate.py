"""fp8 ``T.gemm`` must be refused below SM89 instead of trapping at launch.

``.e4m3``/``.e5m2`` ``mma.sync`` atoms first exist on SM89. The MMA path picks
the fp8 atom from the operand dtype alone, so an SM80/SM86 target used to reach
``cute::SM89_16x8x32_F32E4M3E4M3F32_TN``, whose body below ``__CUDA_ARCH__ 890``
is ``CUTE_INVALID_CONTROL_PATH`` -- an ``assert(0)``. nvcc then compiled the
kernel without a word and the device trapped at launch with a bare
``device-side assert triggered``.

The checks are compile-only and pinned to explicit targets, so they run on any
CUDA runner regardless of the host GPU.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
import tvm
from tilelang.contrib import nvcc

_FP8_GATE_MESSAGE = r"T\.gemm\(\) with fp8 operands requires a CUDA target with fp8 mma\.sync atoms \(SM89\+\)"


def _make_gemm(in_dtype, M=64, N=64, K=64, threads=64):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared((M, K), in_dtype)
            B_shared = T.alloc_shared((N, K), in_dtype)
            C_local = T.alloc_fragment((M, N), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


def _compile(in_dtype, arch):
    return tilelang.compile(_make_gemm(in_dtype), target={"kind": "cuda", "arch": arch}, out_idx=[2])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_86"])
@pytest.mark.parametrize("in_dtype", [T.float8_e4m3fn, T.float8_e5m2])
def test_fp8_gemm_rejects_pre_sm89_target(in_dtype, arch):
    with pytest.raises(tvm.error.InternalError, match=_FP8_GATE_MESSAGE):
        _compile(in_dtype, arch)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("in_dtype", [T.float8_e4m3fn, T.float8_e5m2])
def test_fp8_gemm_keeps_mma_path_on_sm89(in_dtype):
    """SM89 has the atoms; the gate must not demote or reject them."""
    kernel = _compile(in_dtype, "sm_89")
    assert "tl::mma_sync" in kernel.get_kernel_source()


def _compile_ptx(in_dtype, arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        artifact = tilelang.lower(_make_gemm(in_dtype), target=target, enable_device_compile=False)
        return nvcc.compile_cuda(artifact.kernel_source, target_format="ptx", options=nvcc.default_compile_options()).decode()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("arch", "block_scaled"), [("sm_89", False), ("sm_120a", True)])
@pytest.mark.parametrize(("in_dtype", "ptx_dtype"), [(T.float8_e4m3fn, "e4m3"), (T.float8_e5m2, "e5m2")])
def test_fp8_gemm_fp32_accumulation_mma(in_dtype, ptx_dtype, arch, block_scaled):
    """SM120 runs an fp32-accumulating fp8 ``mma.sync`` at half the rate of the
    block-scaled MXF8F6F4 MMA, so there the MMA uses the block-scaled form with
    unit scale factors. Other targets keep the plain fp8 ``mma.sync``."""
    if arch.startswith("sm_120") and nvcc.get_cuda_version() < (12, 8):
        pytest.skip("SM120a compilation requires CUDA 12.8+")
    ptx = _compile_ptx(in_dtype, arch)
    block_scaled_mma = f"kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.{ptx_dtype}.{ptx_dtype}.f32.ue8m0"
    plain_mma = f"mma.sync.aligned.m16n8k32.row.col.f32.{ptx_dtype}.{ptx_dtype}.f32"
    assert (block_scaled_mma in ptx) == block_scaled
    assert (plain_mma in ptx) != block_scaled


@tilelang.testing.requires_cuda
def test_fp16_gemm_on_sm80_is_unaffected():
    """The gate is dtype-specific: a 16-bit operand still lowers to mma.sync."""
    kernel = _compile(T.float16, "sm_80")
    assert "tl::mma_sync" in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
