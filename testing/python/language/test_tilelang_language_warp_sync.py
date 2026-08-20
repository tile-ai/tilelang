import pytest

import tilelang
import tilelang.language as T
import torch
from tvm import tirx
import tilelang.testing


@tilelang.jit
def kernel_with_warp_sync():
    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                tirx.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


@tilelang.testing.requires_cuda
def test_warp_sync():
    a = torch.empty((1), device="cuda", dtype=torch.int32)
    b = torch.empty((1), device="cuda", dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    assert "__syncwarp" in kernel.get_kernel_source()
    kernel(a, b)
    assert b[0] == -1


@tilelang.jit
def kernel_with_shfl_sync():
    @T.prim_func
    def main(
        A: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_sync(val, 31)
            A[tx] = broadcast

    return main


@tilelang.testing.requires_cuda
def test_shfl_sync():
    a = torch.empty((32), device="cuda", dtype=torch.int32)
    kernel = kernel_with_shfl_sync()
    assert "__shfl_sync" in kernel.get_kernel_source()
    kernel(a)
    assert torch.all(a == 310)


def _shift(lane, delta):
    """Return the source lane, or the current lane when the shift is out of range."""
    shifted = lane + delta
    return shifted if 0 <= shifted < 32 else lane


# Builtin each shuffle lowers to and the lane its result comes from, in the row
# order shfl_all_ops_kernel writes. The tables differ so the two shapes cannot fold.
SHFL_TEMP_OPS = (
    ("__shfl_sync", lambda lane: 31),
    ("__shfl_xor_sync", lambda lane: lane ^ 1),
    ("__shfl_down_sync", lambda lane: _shift(lane, 1)),
    ("__shfl_up_sync", lambda lane: _shift(lane, -1)),
)
SHFL_INLINE_OPS = (
    ("__shfl_sync", lambda lane: 0),
    ("__shfl_xor_sync", lambda lane: lane ^ 2),
    ("__shfl_down_sync", lambda lane: _shift(lane, 2)),
    ("__shfl_up_sync", lambda lane: _shift(lane, -2)),
)

# One distinct value per lane so a wrong source lane cannot match by accident, on
# float8_e5m2's coarse grid. Lane 0's negative zero needs the byte comparison.
_MANTISSAS = (1.0, 1.25, 1.5, 1.75)
_GRID = [sign * mant * 2.0**exp for exp in range(4) for mant in _MANTISSAS for sign in (1, -1)]
LANE_VALUES = [-0.0, *_GRID[1:]]


def shfl_all_ops_kernel(dtype):
    @T.prim_func
    def main(
        A: T.Tensor((32,), dtype),
        B: T.Tensor((len(SHFL_TEMP_OPS) + len(SHFL_INLINE_OPS), 32), dtype),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            # Rows 0-3 copy-initialize the result, rows 4-7 store it inline.
            broadcast = T.shfl_sync(A[tx], 31)
            swapped = T.shfl_xor(A[tx], 1)
            shifted_down = T.shfl_down(A[tx], 1)
            shifted_up = T.shfl_up(A[tx], 1)
            B[0, tx] = broadcast
            B[1, tx] = swapped
            B[2, tx] = shifted_down
            B[3, tx] = shifted_up
            B[4, tx] = T.shfl_sync(A[tx], 0)
            B[5, tx] = T.shfl_xor(A[tx], 2)
            B[6, tx] = T.shfl_down(A[tx], 2)
            B[7, tx] = T.shfl_up(A[tx], 2)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float8_e4m3, T.float8_e5m2])
def test_shfl_narrow_float_dtypes(dtype):
    kernel = tilelang.compile(shfl_all_ops_kernel(dtype))
    source = kernel.get_kernel_source()

    values = torch.tensor(LANE_VALUES, device="cuda", dtype=torch.float32)
    torch_dtype = dtype.as_torch()
    rows = len(SHFL_TEMP_OPS) + len(SHFL_INLINE_OPS)
    # out_idx would rebuild a torch tensor from float8 DLPack, unsupported on some versions.
    b = torch.empty(rows, 32, device="cuda", dtype=torch_dtype)
    kernel(values.to(torch_dtype), b)

    # Both shapes must still reach the raw builtin, or these overloads stop being
    # covered. Count in the kernel body, not the preamble.
    body = source[source.rindex("__global__") :]
    for builtin, _ in SHFL_TEMP_OPS:
        assert body.count(f"{builtin}(") == 2

    for base, ops, form in ((0, SHFL_TEMP_OPS, "temporary"), (len(SHFL_TEMP_OPS), SHFL_INLINE_OPS, "inline")):
        for offset, (builtin, src_lane) in enumerate(ops):
            ref = values[[src_lane(lane) for lane in range(32)]].to(torch_dtype)
            # Compare bytes: a float comparison would not separate -0.0 from +0.0.
            got_bytes = b[base + offset].view(torch.uint8)
            ref_bytes = ref.view(torch.uint8)
            assert torch.equal(got_bytes, ref_bytes), (
                f"{builtin} ({form} form) shuffled the wrong lane: got {got_bytes.tolist()}, expected {ref_bytes.tolist()}"
            )


if __name__ == "__main__":
    tilelang.testing.main()
