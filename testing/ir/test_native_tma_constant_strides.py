"""Static descriptor strides must reach the TMA byte-alignment checks."""

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import tvm_tirx as tvm


@pytest.mark.parametrize(
    "dtype,row_stride,accepted",
    [
        ("int8", 16, True),
        ("int8", 17, False),
        ("float16", 16, True),
        ("float16", 17, False),
        ("float32", 16, True),
        ("float32", 17, False),
        ("float32", 2**38 - 4, True),
        ("float32", 2**38, False),
    ],
)
@tilelang.testing.requires_cuda
def test_tma_global_byte_stride_contract(dtype, row_stride, accepted):
    # Lower only: the large-stride boundaries need no allocation or GPU run.
    stride = tvm.ir.const(row_stride, "int64")

    @T.prim_func
    def main(
        A: T.StridedTensor((2, 16), (stride, 1), dtype),
        B: T.Tensor((2, 16), dtype),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((2, 16), dtype)
            barrier = T.alloc_barrier(128)
            T.tma_copy(A, shared, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            T.copy(shared, B)

    target = {"kind": "cuda", "arch": "sm_90"}

    def lower():
        with tvm.transform.PassContext(), tvm.target.Target(target):
            return tilelang.lower(main, target=target)

    if accepted:
        artifact = lower()
        assert "tma_load" in artifact.kernel_source
    else:
        # Explicit tiled copies reject invalid physical strides at the
        # eligibility check, before descriptor geometry is constructed.
        with pytest.raises(Exception, match=r"T\.tma_copy\(\).*TMA is not available for src=A, dst=shared"):
            lower()


@pytest.mark.parametrize("dtype", ["int8", "float16", "float32"])
@pytest.mark.parametrize("channels,accepted", [(1, False), (16, True)])
@tilelang.testing.requires_cuda
def test_im2col_channel_geometry_byte_stride(dtype, channels, accepted):
    @T.prim_func
    def main(
        A: T.Tensor((1, 8, 8, channels), dtype),
        B: T.Tensor((16, channels), dtype),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, channels), dtype)
            T.im2col(A, shared, 0, 0, 3, 1, 1, 1)
            T.copy(shared, B)

    target = {"kind": "cuda", "arch": "sm_90"}

    def lower():
        with tvm.transform.PassContext(), tvm.target.Target(target):
            return tilelang.lower(main, target=target)

    if accepted:
        artifact = lower()
        assert "tma_load_im2col" in artifact.kernel_source
    else:
        # A single channel lets the shared pixel axis pair with TMA mode 0.
        # Its slower channel axis then has a sub-16-byte global stride.
        with pytest.raises(Exception, match=r"TMA global stride .* must be a 16-byte multiple below 2\^40"):
            lower()
