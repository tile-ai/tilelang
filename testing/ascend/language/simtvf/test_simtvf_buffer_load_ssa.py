import warnings
from typing import Any

warnings.filterwarnings("ignore", message="Permission mismatch.*", module="torch_npu.utils._path_manager")
warnings.filterwarnings("ignore", message="Warning: The .* owner does not match the current owner\\.", module="torch_npu.utils.collect_env")

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


PASS_CONFIGS = {tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def mutable_buffer_load_cast_kernel() -> Any:
    @T.prim_func
    def main(dst: T.Tensor((4,), T.float16)) -> None:
        with T.Kernel(1):
            out = T.alloc_shared((4,), T.float16)
            with T.SimtVF(threads=1):
                x = T.alloc_local((1,), "float32x2")
                x[0] = T.Broadcast(T.float32(1.0), 2)
                cast_1 = T.cast(x[0], "float16x2")
                out[T.Ramp(0, 1, 2)] = cast_1
                x[0] = T.Broadcast(T.float32(2.0), 2)
                cast_2 = T.cast(x[0], "float16x2")
                out[T.Ramp(2, 1, 2)] = cast_2
            T.copy(out, dst)

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def mutable_scalarized_store_kernel() -> Any:
    @T.prim_func
    def main(dst: T.Tensor((6,), T.float32)) -> None:
        with T.Kernel(1):
            out = T.alloc_shared((6,), T.float32)
            with T.SimtVF(threads=1):
                x = T.alloc_local((1,), "float32x2")
                x[0] = T.Broadcast(T.float32(1.0), 2)
                out[T.Ramp(1, 1, 2)] = x[0]
                x[0] = T.Broadcast(T.float32(2.0), 2)
                out[T.Ramp(3, 1, 2)] = x[0]
            T.copy(out, dst)

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def mutable_nested_buffer_load_kernel() -> Any:
    @T.prim_func
    def main(dst: T.Tensor((6,), T.float32)) -> None:
        with T.Kernel(1):
            out = T.alloc_shared((6,), T.float32)
            with T.SimtVF(threads=1):
                x = T.alloc_local((1,), "float32x2")
                y = T.alloc_local((1,), "float32x2")
                x[0] = T.Broadcast(T.float32(1.0), 2)
                y[0] = T.Broadcast(T.float32(10.0), 2)
                out[T.Ramp(1, 1, 2)] = x[0] + y[0]
                x[0] = T.Broadcast(T.float32(2.0), 2)
                out[T.Ramp(3, 1, 2)] = x[0] + y[0]
            T.copy(out, dst)

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def overlapping_scalarized_store_kernel() -> Any:
    @T.prim_func
    def main(dst: T.Tensor((6,), T.float32)) -> None:
        with T.Kernel(1):
            x = T.alloc_shared((6,), T.float32)
            with T.SimtVF(threads=1):
                x[0] = 1.0
                x[1] = 2.0
                x[2] = 3.0
                x[3] = 4.0
                x[4] = 5.0
                x[5] = 6.0
                x[T.Ramp(1, 1, 2)] = x[T.Ramp(0, 1, 2)]
            T.copy(x, dst)

    return main


def test_cast_uses_operation_local_buffer_load_ssa_value():
    dst = torch.empty((4,), dtype=torch.float16, device="npu")
    kernel = mutable_buffer_load_cast_kernel()

    assert kernel.get_kernel_source().count(" = x[0];") == 2
    kernel(dst)
    torch.npu.synchronize()
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float16)
    assert torch.equal(dst.cpu(), expected)


def test_scalarized_store_uses_operation_local_buffer_load_ssa_value():
    dst = torch.empty((6,), dtype=torch.float32, device="npu")
    kernel = mutable_scalarized_store_kernel()

    assert kernel.get_kernel_source().count(" = x[0];") == 2
    kernel(dst)
    torch.npu.synchronize()
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32)
    assert torch.equal(dst[1:5].cpu(), expected)


def test_nested_buffer_load_expression_uses_operation_local_ssa_value():
    dst = torch.empty((6,), dtype=torch.float32, device="npu")
    kernel = mutable_nested_buffer_load_kernel()

    assert kernel.get_kernel_source().count("x[0] + y[0]") == 2
    kernel(dst)
    torch.npu.synchronize()
    expected = torch.tensor([11.0, 11.0, 12.0, 12.0], dtype=torch.float32)
    assert torch.equal(dst[1:5].cpu(), expected)


def test_scalarized_store_materializes_overlapping_buffer_load_once():
    dst = torch.empty((6,), dtype=torch.float32, device="npu")
    kernel = overlapping_scalarized_store_kernel()

    kernel(dst)
    torch.npu.synchronize()
    expected = torch.tensor([1.0, 1.0, 2.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    assert torch.equal(dst.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
