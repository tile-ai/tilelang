"""Exercise the ACL host-argument ABI used by the tvm-ffi Ascend runtime."""

from __future__ import annotations

import platform

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


if tilelang.tvm.ffi.get_global_func("target.build.tilelang_ascend", allow_missing=True) is None:
    pytest.skip("Ascend codegen is not built", allow_module_level=True)

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
def dynamic_grid_kernel(
    source: T.Tensor((ROWS, WIDTH), T.bfloat16),
    destination: T.Tensor((ROWS, WIDTH), T.bfloat16),
):
    with T.Kernel(ROWS) as row:
        tile = T.alloc_shared((WIDTH,), T.bfloat16)
        T.copy(source[row, :], tile)
        T.copy(tile, destination[row, :])


def require_npu() -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("an available Ascend NPU is required")


def test_tvm_ffi_packs_mixed_width_scalars() -> None:
    require_npu()
    kernel = tilelang.compile(
        scalar_args_kernel,
        out_idx=0,
        target="ascend",
    )

    left = 17
    wide = 1 << 40
    right = -9
    output = kernel(left, wide, right)
    torch.npu.synchronize()

    expected = torch.full_like(output, left + wide + right)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("rows", [1, 8])
def test_tvm_ffi_dynamic_grid(rows: int, tmp_path) -> None:
    require_npu()
    kernel = tilelang.compile(
        dynamic_grid_kernel,
        out_idx=-1,
        target="ascend",
    )

    source = torch.randn(
        (rows, WIDTH),
        dtype=torch.bfloat16,
        device="npu",
    )
    destination = kernel(source)
    torch.npu.synchronize()

    torch.testing.assert_close(destination, source)
    if kernel.target_host.kind.name == "mir":
        path = str(tmp_path / "ascend.tlm")
        kernel.export_library(path)
        loaded = tilelang.tvm.runtime.load_module(path)
        anchor = torch.empty(0, dtype=source.dtype, device=source.device)
        restored = loaded(source, anchor)
        assert restored.data_ptr() != source.data_ptr()
        torch.testing.assert_close(restored, source)


@pytest.mark.parametrize("func,out_idx", [(scalar_args_kernel, 0), (dynamic_grid_kernel, -1)])
def test_mir_host_lowering_without_npu(func, out_idx, tmp_path):
    from tilelang import tvm
    from tilelang.backend import create_backend_context
    from tilelang.engine.lower import host_codegen, lower_to_host_device_ir
    from tilelang.jit.abi import prepare_tvm_ffi_callee_allocated_outputs

    if platform.machine().lower() not in ("x86_64", "amd64", "aarch64", "arm64"):
        pytest.skip("MIR host execution requires x86-64 or AArch64")
    if tvm.ffi.get_global_func("target.build.tilelang_mir", allow_missing=True) is None:
        pytest.skip("MIR host compiler is not built")
    context = create_backend_context("ascend")
    assert context.target_host.kind.name == "mir"
    func, _ = prepare_tvm_ffi_callee_allocated_outputs(func, out_idx, supports_callee_allocated_outputs=True)
    with context.target, tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        host, _, *_ = lower_to_host_device_ir(func, context)
        module = host_codegen(host, context)
    path = tmp_path / "host.tlm"
    path.write_bytes(tvm.ffi.get_global_func("tilelang.ModuleSerialize")(module))
    assert tvm.runtime.load_module(str(path)).kind == module.kind


if __name__ == "__main__":
    tilelang.testing.main()
