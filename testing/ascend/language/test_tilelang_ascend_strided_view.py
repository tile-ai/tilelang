"""Key ``T.view`` coverage for strided tensors on Ascend."""

import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

BATCH, ROWS = 2, 12
COLS_I8 = 448
PITCHES_I8 = (512, 520)
COLS_I32 = COLS_I8 // 4


@tilelang.jit(out_idx=None, target="ascend")
def rank_changed_scalar_store_kernel():
    stride0 = T.dynamic("stride0", dtype=T.int64)
    stride1 = T.dynamic("stride1", dtype=T.int64)

    @T.prim_func
    def main(out: T.StridedTensor((BATCH, ROWS, COLS_I8), (stride0, stride1, 1), T.int8)) -> None:
        # The caller guarantees stride0 == ROWS * stride1 and stride1 % 4 == 0.
        out_i32 = T.view(
            out,
            shape=(BATCH * ROWS, COLS_I32),
            dtype=T.int32,
            strides=(stride1 // 4, 1),
        )
        with T.Kernel(1):
            out_i32[BATCH * ROWS - 1, COLS_I32 - 1] = 1

    return main


@tilelang.jit(out_idx=None, target="ascend")
def strided_copy_kernel():
    stride0 = T.dynamic("stride0", dtype=T.int64)

    @T.prim_func
    def main(
        src: T.Tensor((ROWS, COLS_I8), T.int8),
        out: T.StridedTensor((ROWS, COLS_I8), (stride0, 1), T.int8),
    ) -> None:
        src_i32 = T.view(src, shape=(ROWS, COLS_I32), dtype=T.int32)
        out_i32 = T.view(out, shape=(ROWS, COLS_I32), dtype=T.int32, strides=(stride0 // 4, 1))
        with T.Kernel(1):
            staging = T.alloc_shared((ROWS, COLS_I32), T.int32)
            T.copy(src_i32, staging)
            T.copy(staging, out_i32)

    return main


def test_explicit_strided_view_rank_change():
    device = torch.device("npu")
    kernel = rank_changed_scalar_store_kernel()
    for pitch_i8 in PITCHES_I8:
        storage = torch.zeros((BATCH, ROWS, pitch_i8), dtype=torch.int8, device=device)
        view = storage[:, :, :COLS_I8]
        assert view.stride(0) == ROWS * view.stride(1)
        assert view.stride(1) % 4 == 0

        kernel(view)
        torch.npu.synchronize()

        view_i32 = view.view(torch.int32).reshape(BATCH * ROWS, COLS_I32)
        assert view_i32[BATCH * ROWS - 1, COLS_I32 - 1].item() == 1
        assert int(torch.count_nonzero(storage)) == 1


def test_strided_view_copy():
    device = torch.device("npu")
    src = torch.randint(1, 127, (ROWS, COLS_I8), dtype=torch.int8, device=device)
    kernel = strided_copy_kernel()
    for pitch_i8 in PITCHES_I8:
        storage = torch.zeros((ROWS, pitch_i8), dtype=torch.int8, device=device)
        view = storage[:, :COLS_I8]

        kernel(src, view)
        torch.npu.synchronize()

        torch.testing.assert_close(view, src)
        assert int(torch.count_nonzero(storage[:, COLS_I8:])) == 0


if __name__ == "__main__":
    tilelang.testing.main()
