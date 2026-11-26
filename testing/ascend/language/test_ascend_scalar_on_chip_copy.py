"""Runtime regression: a DMA fill followed by a scalar on-chip copy.

A `T.copy` whose endpoints are both on-chip lowers to element-wise
`BufferStores`, i.e. a scalar loop outside any VF body. The scalar reader
must therefore be ordered against the MTE2 that filled the source buffer;
a PIPE_V wait does not do that.

Before the fix the first row was corrupted on every run.
"""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@pytest.mark.parametrize("hops", [1, 2])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_scalar_on_chip_copy_waits_for_its_dma_producer(hops, dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    rows, cols = 16, 128
    torch_dtype = getattr(torch, dtype)

    def build():
        if hops == 1:

            @T.prim_func
            def main(A: T.Tensor((rows, cols), dtype), C: T.Tensor((rows, cols), dtype)):
                with T.Kernel(1):
                    first = T.alloc_shared((rows, cols), dtype)
                    second = T.alloc_shared((rows, cols), dtype)
                    T.copy(A, first)
                    T.copy(first, second)
                    T.copy(second, C)

            return main

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), C: T.Tensor((rows, cols), dtype)):
            with T.Kernel(1):
                first = T.alloc_shared((rows, cols), dtype)
                second = T.alloc_shared((rows, cols), dtype)
                third = T.alloc_shared((rows, cols), dtype)
                T.copy(A, first)
                T.copy(first, second)
                T.copy(second, third)
                T.copy(third, C)

        return main

    kernel = tilelang.compile(build(), target="ascend")
    source = kernel.get_kernel_source()
    # The DMA that fills `first` must be ordered against the scalar pipe.
    assert "asc_sync_notify(PIPE_MTE2, PIPE_S" in source or "asc_sync()" in source, (
        "the DMA that fills the source buffer is not ordered against PIPE_S; the scalar copy would race it"
    )

    torch.manual_seed(0)
    for _ in range(4):
        A = torch.randn(rows, cols, device="npu").to(torch_dtype)
        C = torch.full((rows, cols), -999.0, device="npu", dtype=torch_dtype)
        kernel(A, C)
        torch.npu.synchronize()
        torch.testing.assert_close(C, A, rtol=0, atol=0)
