from types import SimpleNamespace

import pytest
import torch

from tilelang import tvm
import tilelang.ascend.language as T
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
    static_shapes, static_strides, _, _ = adapter._process_static_buffer_infos()

    assert list(static_shapes.values())[0][1] == [(0, 4), (1, 128)]
    assert list(static_strides.values())[0][1] == [(0, 256), (1, 1)]

    wrapper = CythonKernelWrapper([], [], None)
    wrapper.set_static_strides_map(static_strides)
    wrapper._check_static_strides([torch.empty_strided((4, 128), (256, 1), dtype=torch.int8)])
    with pytest.raises(ValueError, match="Static stride mismatch"):
        wrapper._check_static_strides([torch.empty_strided((4, 128), (512, 1), dtype=torch.int8)])


def test_cython_adapter_tracks_dynamic_outer_packed_stride():
    outer_stride = T.dynamic("outer_stride")

    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (outer_stride, 1), T.int4]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = PTO_TARGET
    _, _, _, dynamic_strides = adapter._process_static_buffer_infos()

    buffer_index, strides = dynamic_strides["A"]
    assert buffer_index == 0
    assert strides == [(0, outer_stride, 2)]

    wrapper = CythonKernelWrapper([], [], None)
    wrapper.set_dynamic_strides_map(dynamic_strides)
    wrapper._check_dynamic_strides(
        [torch.empty_strided((4, 128), (256, 1), dtype=torch.int8)],
        {outer_stride: 512},
    )
    with pytest.raises(ValueError, match="Dynamic packed stride mismatch"):
        wrapper._check_dynamic_strides(
            [torch.empty_strided((4, 128), (512, 1), dtype=torch.int8)],
            {outer_stride: 512},
        )


def test_cython_adapter_resolves_explicit_scalar_param_in_packed_stride():
    """Packed strides referencing an explicit int kernel arg must resolve."""

    @T.prim_func
    def main(A: T.handle, s: T.int32, scale: T.float32):
        A = T.match_buffer(A, (4, 256), "int4", strides=(s, 1))
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = PTO_TARGET
    adapter.result_idx = []
    _, _, _, dynamic_strides = adapter._process_static_buffer_infos()

    stride_idx, stride_expr, packing_factor = dynamic_strides["A"][1][0]
    assert (stride_idx, packing_factor) == (0, 2)
    assert stride_expr.same_as(main.params[1])

    scalar_param_vars = adapter._process_scalar_param_vars()
    assert scalar_param_vars[0] is None
    assert scalar_param_vars[1].same_as(main.params[1])
    assert scalar_param_vars[2].same_as(main.params[2])

    wrapper = CythonKernelWrapper([], [], None)
    wrapper.set_dynamic_symbolic_map(adapter._process_dynamic_symbolic())
    wrapper.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
    wrapper.set_dynamic_strides_map(dynamic_strides)
    wrapper.set_scalar_param_vars(scalar_param_vars)

    # s is the logical (unpacked) stride; storage stride 256 * packing 2 = 512.
    # The float scalar is collected but not merged: IntImm substitution is
    # integer-only.
    tensor_list = [torch.empty_strided((4, 128), (256, 1), dtype=torch.int8), 512, 1.0]
    dynamic_values = wrapper._resolve_dynamic_values(tensor_list)
    assert dynamic_values[main.params[1]] == 512
    assert main.params[2] not in dynamic_values
    wrapper._check_dynamic_strides(tensor_list, dynamic_values)

    tensor_list = [torch.empty_strided((4, 128), (512, 1), dtype=torch.int8), 1024, 1.0]
    wrapper._check_dynamic_strides(tensor_list, wrapper._resolve_dynamic_values(tensor_list))
    with pytest.raises(ValueError, match="Dynamic packed stride mismatch"):
        tensor_list = [torch.empty_strided((4, 128), (512, 1), dtype=torch.int8), 512, 1.0]
        wrapper._check_dynamic_strides(tensor_list, wrapper._resolve_dynamic_values(tensor_list))


def test_cython_adapter_accepts_zero_sized_packed_tensor():
    """Zero-sized tensors have unconstrained strides and must be accepted."""
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.StridedTensor((4, n), (n, 1), T.int4)):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = PTO_TARGET
    adapter.result_idx = []
    _, _, _, dynamic_strides = adapter._process_static_buffer_infos()

    wrapper = CythonKernelWrapper([], [], None)
    wrapper.set_dynamic_symbolic_map(adapter._process_dynamic_symbolic())
    wrapper.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
    wrapper.set_dynamic_strides_map(dynamic_strides)

    for tensor in (
        torch.empty((4, 0), dtype=torch.int8),
        torch.empty_strided((4, 0), (1, 1), dtype=torch.int8),
    ):
        assert tensor.numel() == 0
        dynamic_values = wrapper._resolve_dynamic_values([tensor])
        assert dynamic_values[n] == 0
        wrapper._check_dynamic_strides([tensor], dynamic_values)

    # Non-empty tensors with a genuine stride mismatch are still rejected.
    tensor = torch.empty_strided((4, 128), (512, 1), dtype=torch.int8)
    with pytest.raises(ValueError, match="Dynamic packed stride mismatch"):
        wrapper._check_dynamic_strides([tensor], wrapper._resolve_dynamic_values([tensor]))


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


def test_cython_adapter_leaves_non_packed_int4_abi_unchanged():
    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (512, 1), T.int4]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = CUDA_TARGET
    static_shapes, static_strides, _, _ = adapter._process_static_buffer_infos()

    assert list(static_shapes.values())[0][1] == [(0, 4), (1, 256)]
    assert list(static_strides.values())[0][1] == [(0, 512), (1, 1)]
