"""One ND2NZ descriptor covers a complete leading batch axis."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T


def batch_copy(num_batches, rows, padded_rows, cols, dtype, interleaved=False, copy_rows=None):
    if copy_rows is None:
        copy_rows = rows
    hidden = cols * 4
    shape = (num_batches, rows, hidden)
    strides = (hidden, num_batches * hidden, 1) if interleaved else (rows * hidden, hidden, 1)

    @T.prim_func
    def main(source: T.StridedTensor(shape, strides, dtype)):
        with T.Kernel(1):
            target = T.alloc_l1((num_batches, padded_rows, cols), dtype)
            with T.Cube():
                T.copy(source[:, :copy_rows, :cols], target[:, :copy_rows, :])

    return main


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_batched_nd2nz_descriptor(interleaved, dtype):
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(batch_copy(4, 24, 32, 128, dtype, interleaved), target="ascend").kernel_source
    elem_bytes = tvm.DataType(dtype).bits // 8
    assert source.count("asc_copy_gm2l1_nd2nz(") == 1
    assert f"asc_set_gm2l1_nz_para(4, 1, static_cast<uint16_t>(32), {32 * 128 * elem_bytes // 32});" in source
    row_stride = (4 * 512 if interleaved else 512) * elem_bytes
    batch_stride = (512 if interleaved else 24 * 512) * elem_bytes
    assert f", {row_stride}, static_cast<asc_load_l2_cache_mode>(0), 24, 128, {batch_stride}, 0);" in source


def test_batched_nd2nz_matrix_tail_padding():
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(batch_copy(3, 20, 32, 32, "bfloat16", copy_rows=24), target="ascend").kernel_source
    assert "asc_fill_l1(" in source
    assert "asc_set_gm2l1_nz_para(3, 1," in source
    assert ", 20, 32," in source


def test_batched_nd2nz_rejects_out_of_bounds_batch():
    @T.prim_func
    def main(source: T.Tensor((2, 16, 16), T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1((4, 16, 16), T.bfloat16)
            T.copy(source[:4, :, :], target)

    with pytest.raises(Exception, match="in-bounds batch axis"):
        tilelang.lower(main, target="ascend")
