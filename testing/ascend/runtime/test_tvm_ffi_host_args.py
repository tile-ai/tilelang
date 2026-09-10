"""Exercise the ACL host-argument ABI used by the tvm-ffi Ascend runtime."""

from __future__ import annotations

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


pytest.importorskip("torch_npu")

ELEMENTS = 128
ROWS = T.dynamic("rows")
WIDTH = 128


@T.prim_func
def scalar_args_kernel(
    output: T.Tensor((ELEMENTS,), T.int64),
    left: T.int32,
    wide: T.int64,
    right: T.int32,
):
    with T.Kernel(1), T.SimtVF(threads=ELEMENTS):
        for i in T.Parallel(ELEMENTS):
            output[i] = left + wide + right


@T.prim_func
def dynamic_ubuf_kernel(
    source: T.Tensor((ROWS, WIDTH), T.bfloat16),
    destination: T.Tensor((ROWS, WIDTH), T.bfloat16),
):
    with T.Kernel(ROWS) as row:
        tile = T.alloc_shared((WIDTH,), T.bfloat16)
        T.copy(source[row, :], tile)
        T.copy(tile, destination[row, :])


def require_npu() -> None:
    if not torch.npu.is_available():
        pytest.skip("an available Ascend NPU is required")
    torch.npu.set_device(0)


def test_tvm_ffi_packs_mixed_width_scalars() -> None:
    require_npu()
    kernel = tilelang.compile(
        scalar_args_kernel,
        out_idx=0,
        target="ascend",
        execution_backend="tvm_ffi",
    )
    assert kernel.adapter._ffi_callee_allocated_output_abi
    assert "TVMFFIEnvTensorAlloc" in kernel.get_host_source()

    left = 17
    wide = 1 << 40
    right = -9
    output = kernel(left, wide, right)
    torch.npu.synchronize()

    expected = torch.full_like(output, left + wide + right)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("rows", [1, 8])
def test_tvm_ffi_dynamic_grid_and_ubuf(rows: int) -> None:
    require_npu()
    kernel = tilelang.compile(
        dynamic_ubuf_kernel,
        out_idx=-1,
        target="ascend",
        execution_backend="tvm_ffi",
    )
    assert kernel.adapter._ffi_callee_allocated_output_abi
    assert "TVMFFIEnvTensorAlloc" in kernel.get_host_source()

    source = torch.randn(
        (rows, WIDTH),
        dtype=torch.bfloat16,
        device="npu:0",
    )
    destination = kernel(source)
    torch.npu.synchronize()

    torch.testing.assert_close(destination, source)


if __name__ == "__main__":
    tilelang.testing.main()
