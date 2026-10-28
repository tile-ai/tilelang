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
def test_batched_nd2nz_skips_batch_tail(batch_begin):
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(batch_tail_copy(batch_begin), target="ascend").kernel_source
    assert "asc_fill_l1(" not in source
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
    assert "asc_fill_l1(" not in source
    assert "if (" in source  # A fully out-of-bounds batch must not issue a DMA.


def test_single_matrix_oob_preserves_legacy_behavior():
    @T.prim_func
    def main(source: T.Tensor((3, 16, 16), T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1((1, 16, 16), T.bfloat16)
            with T.Cube():
                T.copy(source[3:4, :, :], target)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert "asc_copy_gm2l1_nd2nz(" not in source
    assert "asc_fill_l1(" not in source


@pytest.mark.parametrize("singleton_axis", [1, 2])
@pytest.mark.parametrize("rows", [16, 20, 32])
def test_single_matrix_nonleading_singleton(singleton_axis, rows):
    shape = (rows, 1, 32) if singleton_axis == 1 else (rows, 32, 1)

    @T.prim_func
    def main(source: T.Tensor(shape, T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1((1, 32, 32), T.bfloat16)
            with T.Cube():
                T.copy(source[:32, :, :], target)

    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(main, target="ascend").kernel_source
    assert source.count("asc_copy_gm2l1_nd2nz(") == 1
    assert "asc_set_gm2l1_nz_para(1, 1," in source
    assert ("asc_fill_l1(" in source) == (rows < 32)


@pytest.mark.parametrize("dst_batches", [1, 2])
def test_batched_nd2nz_rejects_smaller_destination(dst_batches):
    @T.prim_func
    def main(source: T.Tensor((4, 16, 16), T.bfloat16)):
        with T.Kernel(1):
            target = T.alloc_l1((dst_batches, 16, 16), T.bfloat16)
            with T.Cube():
                T.copy(source[:, :, :], target[:, :, :])

    with pytest.raises(tvm.error.InternalError, match="source region.*destination"):
        tilelang.lower(main, target="ascend")


def test_batched_nd2nz_single_row_matrix():
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(batch_copy(4, 1, 16, 32, "bfloat16"), target="ascend").kernel_source
    assert "asc_set_gm2l1_nz_para(4, 1," in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize(
    "source_batches,batch_begin,rows,cols,dynamic,copy_batches,destination_origin",
    [
        (4, 0, 32, 32, False, 4, False),
        (4, 0, 32, 32, False, 4, True),
        (3, 1, 32, 32, False, 4, False),
        (3, 3, 32, 32, False, 4, False),
        (3, 3, 20, 20, False, 4, False),
        (3, 1, 20, 20, False, 4, False),
        (3, 0, 20, 20, True, 4, False),
        (3, 0, 32, 32, False, 2, False),
        (3, 2, 20, 20, False, 2, False),
    ],
    ids=[
        "in-bounds",
        "destination-origin",
        "batch-tail",
        "all-batches-oob",
        "all-batches-oob-with-m-n-tails",
        "batch-m-n-tails",
        "dynamic-batch-m-n-tails",
        "larger-destination",
        "larger-destination-batch-m-n-tails",
    ],
)
def test_batched_nd2nz_device(dtype, source_batches, batch_begin, rows, cols, dynamic, copy_batches, destination_origin):
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
            # Seed every matrix to detect missing M/N fills or writes to skipped batches.
            T.copy(poison, target)
            if destination_origin:
                T.copy(source[begin : begin + copy_batches, :32, :32], target[1, 0, 0])
            else:
                T.copy(source[begin : begin + copy_batches, :32, :32], target[1:5, :, :])
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
    for block in range(num_blocks):
        begin = block * 2 if dynamic else batch_begin
        valid_batches = min(copy_batches, max(source_batches - begin, 0))
        expected[block, 1 : 1 + valid_batches] = 0
        expected[block, 1 : 1 + valid_batches, :rows, :cols] = source[begin : begin + valid_batches].float()
    identity = torch.eye(32, dtype=torch_dtype, device="npu")
    poison = torch.full((5, 32, 32), -7.0, dtype=torch_dtype, device="npu")
    actual = kernel(source.npu(), identity, poison)
    torch.npu.synchronize()
    # Batch 0, skipped batches, and extra capacity must retain their sentinels.
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize(
    "rows,cols,copy_rows,copy_cols,interleaved,transpose,batch_begin",
    [
        (32, 32, 32, 32, True, False, 0),
        (32, 32, 32, 32, False, False, 0),
        (20, 10, 32, 32, False, False, 1),
        (32, 64, 32, 64, False, True, 0),
        (32, 64, 32, 64, True, True, 0),
        (20, 10, 32, 64, False, True, 1),
    ],
    ids=[
        "interleaved-source",
        "larger-matrix-pitch",
        "explicit-column-fill",
        "transpose-rectangular",
        "transpose-interleaved-source",
        "transpose-batch-m-n-tails",
    ],
)
def test_batched_nd2nz_strides_and_transpose_device(dtype, rows, cols, copy_rows, copy_cols, interleaved, transpose, batch_begin):
    import torch

    pytest.importorskip("torch_npu")
    source_batches = 3
    # Leave gaps between source rows, and optionally interleave the batch axis.
    row_pitch = cols * 2
    strides = (row_pitch, source_batches * row_pitch, 1) if interleaved else (rows * row_pitch, row_pitch, 1)
    dst_rows, dst_cols = (copy_cols, copy_rows) if transpose else (copy_rows, copy_cols)
    # Existing M/N padding requires the destination's full column axis.
    target_rows, target_cols = (dst_rows, dst_cols) if rows < copy_rows or cols < copy_cols else (64, 64)

    @T.prim_func
    def main(
        source: T.StridedTensor((source_batches, rows, cols), strides, dtype),
        identity: T.Tensor((target_cols, target_cols), dtype),
        poison: T.Tensor((5, target_rows, target_cols), dtype),
        output: T.Tensor((5, target_rows, target_cols), "float32"),
    ):
        with T.Kernel(1):
            target = T.alloc_l1((5, target_rows, target_cols), dtype)
            weight = T.alloc_l1((target_cols, target_cols), dtype)
            a = T.alloc_l0a((target_rows, target_cols), dtype)
            b = T.alloc_l0b((target_cols, target_cols), dtype)
            acc = T.alloc_l0c((target_rows, target_cols), "float32")
            T.copy(poison, target)
            T.copy(
                source[batch_begin : batch_begin + 4, :copy_rows, :copy_cols],
                target[1:5, :dst_rows, :dst_cols],
                transpose=transpose,
            )
            T.copy(identity, weight)
            T.copy(weight, b)
            for batch in T.serial(5):
                T.copy(target[batch : batch + 1, :, :], a)
                T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, output[batch, :, :])

    kernel = tilelang.compile(main, target="ascend", out_idx=-1, pass_configs={"tl.enable_auto_schedule": True})
    torch_dtype = getattr(torch, dtype)
    storage_shape = (rows, source_batches, row_pitch) if interleaved else (source_batches, rows, row_pitch)
    storage = ((torch.arange(source_batches * rows * row_pitch).reshape(storage_shape) % 31) + 1).to(torch_dtype).npu()
    source = (storage.permute(1, 0, 2) if interleaved else storage)[:, :, :cols]
    assert source.stride() == strides
    valid_batches = min(4, max(source_batches - batch_begin, 0))
    expected = torch.full((5, target_rows, target_cols), -7.0)
    expected[1 : 1 + valid_batches, :dst_rows, :dst_cols] = 0
    values = source[batch_begin : batch_begin + valid_batches].cpu().float()
    if transpose:
        values = values.transpose(1, 2)
    expected[1 : 1 + valid_batches, : values.shape[1], : values.shape[2]] = values
    identity = torch.eye(target_cols, dtype=torch_dtype, device="npu")
    poison = torch.full((5, target_rows, target_cols), -7.0, dtype=torch_dtype, device="npu")
    actual = kernel(source, identity, poison)
    torch.npu.synchronize()
    # Check the full allocation: uncopied M/N regions and batches stay untouched.
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
