"""One ND2NZ descriptor covers a complete leading batch axis."""

import pytest
import tilelang
import tilelang.testing
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


def batch_tail_copy(batch_begin, rows=16, copy_rows=16, cols=16, copy_cols=16, dst_begin=0, pad_value=0):
    @T.prim_func
    def main(source: T.Tensor((3, rows, cols), T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1((dst_begin + 4, 32, 32), T.bfloat16)
            with T.Cube():
                T.copy(
                    source[batch_begin : batch_begin + 4, :copy_rows, :copy_cols],
                    target[dst_begin : dst_begin + 4, :copy_rows, :copy_cols],
                    pad_value=pad_value,
                )

    return main


@pytest.mark.parametrize("batch_begin", [0, 2, 3, 5])
def test_batched_nd2nz_batch_tail_padding(batch_begin):
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(batch_tail_copy(batch_begin), target="ascend").kernel_source
    assert "asc_fill_l1(" in source
    valid_batches = max(3 - batch_begin, 0)
    if valid_batches:
        assert source.count("asc_copy_gm2l1_nd2nz(") == 1
        assert f"asc_set_gm2l1_nz_para({valid_batches}, 1," in source
    else:
        assert "asc_copy_gm2l1_nd2nz(" not in source


def test_batched_nd2nz_batch_and_matrix_tails():
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(batch_tail_copy(1, 20, 32, 20, 32, dst_begin=1), target="ascend").kernel_source
    assert "asc_set_gm2l1_nz_para(2, 1," in source
    assert ", 20, 20," in source
    assert "asc_fill_l1(" in source


@pytest.mark.parametrize(
    "source_shape,destination_shape,axis",
    [
        ((32, 16), (16, 16), 0),
        ((16, 32), (16, 16), 1),
        ((4, 16, 16), (2, 16, 16), 0),
    ],
    ids=["row", "column", "batch"],
)
def test_nd2nz_rejects_out_of_bounds_destination(source_shape, destination_shape, axis):
    @T.prim_func
    def main(source: T.Tensor(source_shape, T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1(destination_shape, T.bfloat16)
            if len(source_shape) == 2:
                T.copy(source, target[: source_shape[0], : source_shape[1]])
            else:
                T.copy(source, target[: source_shape[0], :, :])

    # Previously the 32x16 -> 16x16 case clipped the DMA but emitted a fill
    # starting at target + 512 bytes, beyond the 512-byte L1 allocation.
    with pytest.raises(Exception, match=f"in-bounds destination axis {axis}"):
        tilelang.lower(main, target="ascend")


@pytest.mark.parametrize("auto_schedule", [False, True])
def test_batched_nd2nz_dynamic_batch_tail(auto_schedule):
    @T.prim_func
    def main(source: T.Tensor((3, 16, 16), T.bfloat16)):
        with T.Kernel(3) as block_idx:
            target = T.alloc_l1((4, 16, 16), T.bfloat16)
            T.copy(source[block_idx * 2 : block_idx * 2 + 4, :, :], target)

    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": auto_schedule}):
        source = tilelang.lower(main, target="ascend").kernel_source
    assert "asc_copy_gm2l1_nd2nz(" in source
    assert "asc_fill_l1(" in source
    assert "if (" in source  # A fully out-of-bounds batch must not issue a DMA.


def test_batched_nd2nz_single_missing_batch():
    @T.prim_func
    def main(source: T.Tensor((3, 16, 16), T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1((1, 16, 16), T.bfloat16)
            with T.Cube():
                T.copy(source[3:4, :, :], target)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert "asc_copy_gm2l1_nd2nz(" not in source
    assert "asc_fill_l1(" in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize(
    "source_batches,batch_begin,rows,cols,dynamic",
    [
        (4, 0, 32, 32, False),
        (3, 1, 32, 32, False),
        (3, 3, 32, 32, False),
        (3, 1, 20, 20, False),
        (3, 0, 20, 20, True),
    ],
    ids=["in-bounds", "batch-tail", "all-batches-oob", "batch-m-n-tails", "dynamic-batch-m-n-tails"],
)
def test_batched_nd2nz_device(dtype, source_batches, batch_begin, rows, cols, dynamic):
    import torch

    pytest.importorskip("torch_npu")
    num_blocks = 3 if dynamic else 1

    @T.prim_func
    def main(
        source: T.Tensor((source_batches, rows, cols), dtype),
        identity: T.Tensor((32, 32), dtype),
        poison: T.Tensor((5, 32, 32), dtype),
        output: T.Tensor((num_blocks, 5, 32, 32), "float32"),
    ):
        with T.Kernel(num_blocks) as block:
            target = T.alloc_l1((5, 32, 32), dtype)
            weight = T.alloc_l1((32, 32), dtype)
            a = T.alloc_l0a((32, 32), dtype)
            b = T.alloc_l0b((32, 32), dtype)
            acc = T.alloc_l0c((32, 32), "float32")
            begin = block * 2 if dynamic else batch_begin
            # Seed every matrix with nonzero data so missing fills cannot pass.
            T.copy(poison, target)
            T.copy(source[begin : begin + 4, :32, :32], target[1:5, :, :])
            T.copy(identity, weight)
            T.copy(weight, b)
            # Multiplication by identity exposes all L1 elements, including tails.
            for batch in T.serial(5):
                T.copy(target[batch : batch + 1, :, :], a)
                T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, output[block, batch, :, :])

    kernel = tilelang.compile(main, target="ascend", out_idx=-1, pass_configs={"tl.enable_auto_schedule": True})
    torch_dtype = getattr(torch, dtype)
    source = ((torch.arange(source_batches * rows * cols).reshape(source_batches, rows, cols) % 31) + 1).to(torch_dtype)
    expected = torch.full((num_blocks, 5, 32, 32), -7.0)
    expected[:, 1:] = 0
    for block in range(num_blocks):
        begin = block * 2 if dynamic else batch_begin
        valid_batches = min(4, max(source_batches - begin, 0))
        expected[block, 1 : 1 + valid_batches, :rows, :cols] = source[begin : begin + valid_batches].float()
    identity = torch.eye(32, dtype=torch_dtype, device="npu")
    poison = torch.full((5, 32, 32), -7.0, dtype=torch_dtype, device="npu")
    actual = kernel(source.npu(), identity, poison)
    torch.npu.synchronize()
    # Batch 0 is outside the copy destination and must retain its sentinel.
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
