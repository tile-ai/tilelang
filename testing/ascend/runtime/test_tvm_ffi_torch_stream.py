"""Verify Torch producer work and TVM-FFI kernels share one NPU stream."""

from __future__ import annotations

import pytest
import torch

import tilelang
from tilelang.ascend import language as T


pytest.importorskip("torch_npu")

PROBE_ELEMENTS = 256


@tilelang.jit(
    target="ascend",
    execution_backend="tvm_ffi",
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    compile_flags=["-O2"],
)
def build_copy_kernel():
    @T.prim_func
    def copy_kernel(
        source: T.Tensor((PROBE_ELEMENTS,), T.bfloat16),
        destination: T.Tensor((PROBE_ELEMENTS,), T.bfloat16),
    ):
        with T.Kernel(1), T.SimtVF(threads=128):
            for index in T.Parallel(PROBE_ELEMENTS):
                destination[index] = source[index]

    return copy_kernel


def test_tvm_ffi_uses_torch_current_npu_stream() -> None:
    if not torch.npu.is_available():
        pytest.skip("an available Ascend NPU is required")
    torch.npu.set_device(0)

    copy_kernel = build_copy_kernel()
    from tilelang.ascend.torch_exchange import (
        is_torch_npu_stream_exchange_installed,
    )

    size = 1024
    source = torch.full(
        (size, size),
        7,
        dtype=torch.bfloat16,
        device="npu:0",
    ).T
    prepared = torch.zeros(
        (size, size),
        dtype=torch.bfloat16,
        device="npu:0",
    )
    observed = torch.empty(
        (PROBE_ELEMENTS,),
        dtype=torch.bfloat16,
        device="npu:0",
    )
    work = torch.randn(
        (size, size),
        dtype=torch.bfloat16,
        device="npu:0",
    )
    scratch = torch.empty_like(work)

    prepared.copy_(source)
    torch.npu.synchronize()
    copy_kernel(prepared.view(-1)[:PROBE_ELEMENTS], observed)
    assert is_torch_npu_stream_exchange_installed()
    torch.npu.synchronize()

    prepared.zero_()
    observed.fill_(-1)
    torch.npu.synchronize()
    for _ in range(64):
        torch.mm(work, work, out=scratch)
    prepared.copy_(source)
    copy_kernel(prepared.view(-1)[:PROBE_ELEMENTS], observed)
    torch.npu.synchronize()

    assert torch.count_nonzero(observed != 7).item() == 0
