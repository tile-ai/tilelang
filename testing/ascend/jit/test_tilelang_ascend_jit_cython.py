from types import SimpleNamespace

from tilelang import tvm as tvm
import tilelang.ascend.language as T
import pytest
import torch

from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang_cython_wrapper import CythonKernelWrapper


PTO_TARGET = SimpleNamespace(kind=SimpleNamespace(name="ascend"), keys=("ascend", "pto"))
CUDA_TARGET = SimpleNamespace(kind=SimpleNamespace(name="cuda"), keys=("cuda",))


def test_cython_adapter_scales_explicit_outer_int4_stride_to_storage_units():
    """Packed int4 tensors expose byte-storage strides to the Cython wrapper."""

    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (512, 1), T.int4]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = PTO_TARGET
    static_shapes, static_strides, _ = adapter._process_static_buffer_infos()

    assert list(static_shapes.values())[0][1] == [(0, 4), (1, 128)]
    assert list(static_strides.values())[0][1] == [(0, 256), (1, 1)]

    wrapper = CythonKernelWrapper([], [], None)
    wrapper.set_static_strides_map(static_strides)
    wrapper._check_static_strides([torch.empty_strided((4, 128), (256, 1), dtype=torch.int8)])
    with pytest.raises(ValueError, match="Static stride mismatch"):
        wrapper._check_static_strides([torch.empty_strided((4, 128), (512, 1), dtype=torch.int8)])


def test_cython_adapter_rejects_dynamic_outer_packed_stride():
    outer_stride = T.dynamic("outer_stride")

    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (outer_stride, 1), T.int4]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = PTO_TARGET
    with pytest.raises(ValueError, match="do not support dynamic outer strides"):
        adapter._process_static_buffer_infos()


@pytest.mark.parametrize("dtype", [T.int4, T.dtype("uint4")])
def test_cython_adapter_rejects_non_unit_innermost_packed_stride(dtype):
    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (512, 2), dtype]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = PTO_TARGET
    with pytest.raises(ValueError, match="require a static innermost stride of 1"):
        adapter._process_static_buffer_infos()


def test_cython_adapter_leaves_non_pto_int4_abi_unchanged():
    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (512, 1), T.int4]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = CUDA_TARGET
    static_shapes, static_strides, _ = adapter._process_static_buffer_infos()

    assert list(static_shapes.values())[0][1] == [(0, 4), (1, 256)]
    assert list(static_strides.values())[0][1] == [(0, 512), (1, 1)]
