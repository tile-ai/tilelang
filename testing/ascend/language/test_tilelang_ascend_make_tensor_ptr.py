"""GM pointer tables reconstruct DMA buffers, including conditional accesses."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T


def _copy_from_pointer_table(output_table, guarded):
    out_shape = (1,) if output_table else (64,)
    out_dtype = T.ptr if output_table else "float32"

    @T.prim_func
    def copy(src_ptrs: T.Tensor((1,), T.ptr), out: T.Tensor(out_shape, out_dtype), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            if not guarded or enabled > 0:
                src = T.make_tensor(src_ptrs[0], (64,), "float32")
                T.copy(src, ub)
                if output_table:
                    dst = T.make_tensor(out[0], (64,), "float32")
                    T.copy(ub, dst)
                else:
                    T.copy(ub, out)

    return copy


@pytest.mark.parametrize("output_table,guarded", [(False, False), (True, False), (False, True)], ids=["source", "both", "guarded-source"])
def test_dma_from_pointer_tables(output_table, guarded):
    kernel = tilelang.compile(_copy_from_pointer_table(output_table, guarded), target="ascend")
    src = torch.arange(64, dtype=torch.float32, device="npu")
    out = torch.empty_like(src)
    src_ptrs = torch.tensor([src.data_ptr()], device=src.device, dtype=torch.int64)
    destination = torch.tensor([out.data_ptr()], device=out.device, dtype=torch.int64) if output_table else out
    for enabled in (0, 1) if guarded else (1,):
        out.fill_(-1)
        kernel(src_ptrs, destination, enabled)
        torch.testing.assert_close(out, src if enabled else torch.full_like(out, -1), rtol=0, atol=0)
