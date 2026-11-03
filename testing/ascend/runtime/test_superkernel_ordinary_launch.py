"""SK generation must preserve normal TVM-FFI execution on an Ascend device."""

import os

import pytest
import torch

import tilelang
from tilelang.ascend import language as T
from tilelang.contrib.bisheng import normalize_options

pytest.importorskip("torch_npu")
ROWS = T.dynamic("rows")


@T.prim_func
def add_bias(A: T.Tensor((ROWS,), T.int32), B: T.Tensor((ROWS,), T.int32), bias: T.int32):
    with T.Kernel(T.ceildiv(ROWS, 32)) as bx:
        count = T.Cast("int32", T.call_extern("int64", "asc_get_block_num"))
        with T.SimtVF(threads=32):
            for i in T.Parallel(32):
                idx = bx * 32 + i
                if idx < ROWS:
                    B[idx] = A[idx] + bias + count


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, 125, 256])
def test_ordinary_launch(enabled, rows):
    if not torch.npu.is_available():
        pytest.skip("an available Ascend NPU is required")
    kernel = tilelang.compile(
        add_bias,
        target="ascend",
        execution_backend="tvm_ffi",
        out_idx=1,
        pass_configs={
            "tl.enable_auto_schedule": False,
            "tl.ascend_enable_superkernel": enabled,
            "tl.device_compile_flags": normalize_options(os.getenv("TILELANG_TEST_ASCEND_COMPILE_FLAGS")),
        },
    )
    host_source = torch.arange(rows, dtype=torch.int32)
    source = host_source.to("npu")
    result = kernel(source, 7)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), host_source + 7 + (rows + 31) // 32)
