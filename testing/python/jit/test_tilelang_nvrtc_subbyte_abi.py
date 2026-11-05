"""NVRTC byte carriers retain their source ABI and actual storage bounds."""

from functools import lru_cache

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _adapter(function, outputs=()):
    from tilelang.engine.lower import extrac_params
    from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter

    adapter = NVRTCKernelAdapter.__new__(NVRTCKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({function.attrs["global_symbol"]: function})
    adapter.params = extrac_params(function)
    adapter.result_idx = list(outputs)
    adapter._initialize_native_arguments()
    return adapter


def _signed_nibbles(raw):
    values = raw.cpu().view(torch.uint8).reshape(-1).tolist()
    return [((value >> shift & 15) ^ 8) - 8 for value in values for shift in (0, 4)]


@lru_cache(None)
def _static_unpack(element_offset=0):
    @T.prim_func
    def main(A: T.Buffer((8,), "int4", elem_offset=element_offset), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(8):
                B[i] = T.cast(A[i], "int32")

    return main


@lru_cache(None)
def _dynamic_unpack(explicit_stride):
    length = T.dynamic("length", "int64")
    stride = T.dynamic("stride", "int64")
    annotation = T.StridedTensor((length,), (stride,), "int4") if explicit_stride else T.Tensor((length,), "int4")

    @T.prim_func
    def main(A: annotation, B: T.Tensor((length,), "int32")):
        if length > 0:
            with T.Kernel(T.ceildiv(length, 32), threads=32) as block:
                for lane in T.Parallel(32):
                    index = block * 32 + lane
                    if index < length:
                        B[index] = T.cast(A[index], "int32")

    return main


@lru_cache(None)
def _pack(dynamic):
    length = T.dynamic("length", "int64") if dynamic else 8

    @T.prim_func
    def main(A: T.Tensor((length,), "int32"), B: T.Tensor((length,), "int4")):
        # One writer owns every storage byte; the test is about the ABI and
        # never relies on racing nibble read/modify/write operations.
        with T.Kernel(1, threads=1):
            for index in T.serial(length):
                B[index] = T.cast(A[index], "int4")

    return main


def _explicit_shape_unpack():
    # Attach the already-used extent identity as an explicit scalar. This
    # exercises the public IR extension contract as well as the unchanged DSL
    # body; an unrelated same-name scalar must not replace this identity.
    function = _dynamic_unpack(False)
    if hasattr(function, "buffer_map"):
        extent = function.buffer_map[function.params[0]].shape[0]
        return type(function)(
            [*function.params, extent],
            function.body,
            function.ret_type,
            buffer_map=function.buffer_map,
            attrs=function.attrs,
        )
    extent = function.params[0].ty.shape[0]
    return type(function)([*function.params, extent], function.body, function.ret_type, function.attrs)


def _affine_unpack(explicit_stride):
    extent = T.dynamic("extent", "int64")
    stride = T.dynamic("stride", "int64")
    annotation = T.StridedTensor((extent * 2,), (stride * 2 + 1,), "int4") if explicit_stride else T.Tensor((extent * 2,), "int4")

    @T.prim_func
    def main(A: annotation, B: T.Tensor((extent,), "int32")):
        with T.Kernel(1, threads=32):
            for index in T.Parallel(32):
                if index < extent:
                    B[index] = T.cast(A[index * 2], "int32")

    # Old and native IR constructors differ, while this DSL body and its
    # byte/nibble oracle run unchanged in both isolated environments.
    scalars = []
    if hasattr(main, "buffer_map"):
        tirx = tilelang.tvm.tirx

        scalars.append(main.buffer_map[main.params[1]].shape[0])
        if explicit_stride:
            tirx.stmt_functor.post_order_visit(
                main.buffer_map[main.params[0]].strides[0], lambda value: scalars.append(value) if isinstance(value, tirx.Var) else None
            )
        return type(main)([*main.params, *scalars], main.body, main.ret_type, buffer_map=main.buffer_map, attrs=main.attrs)
    import tvm_ffi
    from tvm_tirx import ir

    scalars.append(main.params[1].ty.shape[0])
    if explicit_stride:
        tvm_ffi.structural_walk(main.params[0].ty.effective_layout().shard[0].stride, (ir.Var, scalars.append), order="pre")
    return type(main)([*main.params, *scalars], main.body, main.ret_type, main.attrs)


def _compile(function, tmp_path):
    kernel = tilelang.compile(function, out_idx=1, target="cuda", execution_backend="nvrtc")
    (tmp_path / "host.py").write_text(kernel.adapter.get_host_source())
    (tmp_path / "device.cu").write_text(kernel.get_kernel_source())
    return kernel


@pytest.mark.parametrize("cells", [4, 5, 8])
def test_static_subbyte_uses_storage_capacity_without_requiring_equal_shapes(cells):
    adapter = _adapter(_static_unpack(), (1,))
    source = torch.empty(cells, dtype=torch.int8)
    arguments, implicit = adapter._prepare_native_arguments((source,), device="cpu")
    assert arguments[0] is source
    assert arguments[1].shape == (8,)
    assert implicit == []


@pytest.mark.parametrize("element_offset,cells", [(0, 3), (1, 4), (2, 4)])
def test_static_subbyte_rejects_insufficient_underlying_bytes(element_offset, cells):
    adapter = _adapter(_static_unpack(element_offset), (1,))
    with pytest.raises(ValueError, match="needs byte range"):
        adapter._prepare_native_arguments((torch.empty(cells, dtype=torch.int8),), device="cpu")


@pytest.mark.parametrize("cells", [0, 7, 8])
def test_dynamic_subbyte_output_allocates_original_shape(cells):
    adapter = _adapter(_pack(True), (1,))
    source = torch.empty(cells, dtype=torch.int32)
    arguments, implicit = adapter._prepare_native_arguments((source,), device="cpu")
    assert arguments[1].dtype == torch.int8
    assert arguments[1].shape == (cells,)
    assert implicit == [cells]


def test_static_subbyte_output_keeps_extra_storage():
    adapter = _adapter(_pack(False), (1,))
    arguments, implicit = adapter._prepare_native_arguments((torch.empty(8, dtype=torch.int32),), device="cpu")
    assert arguments[1].dtype == torch.int8
    assert arguments[1].shape == (8,)
    assert implicit == []


def test_dynamic_subbyte_strides_scale_on_both_axes():
    rows, columns = T.dynamic("rows"), T.dynamic("columns")
    row_stride, column_stride = T.dynamic("row_stride"), T.dynamic("column_stride")

    @T.prim_func
    def main(A: T.StridedTensor((rows, columns), (row_stride, column_stride), "int4")):
        T.evaluate(0)

    adapter = _adapter(main)
    source = torch.empty_strided((2, 7), (30, 3), dtype=torch.int8)
    arguments, implicit = adapter._prepare_native_arguments((source,), device="cpu")
    assert arguments == [source]
    assert implicit == [2, 7, 60, 6]


def test_nvrtc_byte_policy_does_not_change_native_packed_bindings():
    from tilelang.jit.adapter.abi import SourceABIPlan

    plan = SourceABIPlan(_dynamic_unpack(True))
    data = torch.empty_strided((7,), (3,), dtype=torch.int8)
    native = plan.bind_arguments([data, None], packed_storage=True)
    source = plan.bind_arguments([data, None], nvrtc_storage=True)
    variables = {variable.name: variable for variable in plan.implicit_symbols}
    assert native.evaluate(variables["length"], "length") == 14
    assert native.evaluate(variables["stride"], "stride") == 3
    assert source.evaluate(variables["length"], "length") == 7
    assert source.evaluate(variables["stride"], "stride") == 6


def test_offset_capacity_is_relative_to_the_logical_pointer_once():
    adapter = _adapter(_static_unpack(1), (1,))
    source = torch.empty(8, dtype=torch.int8)[3:]
    arguments, implicit = adapter._prepare_native_arguments((source,), device="cpu")
    assert source.storage_offset() == 3
    assert arguments[0].data_ptr() == source.data_ptr()
    assert implicit == []
    too_short = torch.empty(8, dtype=torch.int8)[4:]
    with pytest.raises(ValueError, match="storage provides"):
        adapter._prepare_native_arguments((too_short,), device="cpu")


def test_same_name_offset_symbols_remain_distinct():
    offset = T.dynamic("offset", "int64")

    @T.prim_func
    def main(A: T.Buffer((8,), "int4", elem_offset=offset), offset: T.int64):
        T.evaluate(0)

    adapter = _adapter(main)
    arguments, implicit = adapter._prepare_native_arguments((torch.empty(5, dtype=torch.int8), 2), device="cpu")
    assert arguments[1] == 2
    # The annotation captured one Var; the scalar declaration created another.
    # Names must never merge their identities when the native graph is bound.
    assert implicit == [0]
    from tilelang.jit.adapter.abi import source_abi_plan

    assert not source_abi_plan(adapter).implicit_symbols[0].same_as(main.params[1])


def test_explicit_element_offset_scalar_is_not_bound_to_zero():
    from tvm_tirx import ir, tirx
    from tvm_tirx.tirx.layout import S

    offset = ir.Var("offset", "int64")
    tensor = ir.Var("A", tirx.TensorType([8], "int4", layout=tirx.TileLayout(S[[8] : [1]] + offset)))
    function = tirx.Function([tensor, offset], [], ir.PrimType("void"), ir.DictAttrs({"global_symbol": "main"}))
    adapter = _adapter(function)
    arguments, implicit = adapter._prepare_native_arguments((torch.empty(5, dtype=torch.int8), 2), device="cpu")
    assert arguments[1] == 2
    assert implicit == []
    with pytest.raises(ValueError, match="needs byte range"):
        adapter._prepare_native_arguments((torch.empty(5, dtype=torch.int8), 4), device="cpu")


@pytest.mark.parametrize("extent", [0, 7, 14])
def test_explicit_shape_scalar_precedes_subbyte_carrier_shape(extent):
    adapter = _adapter(_explicit_shape_unpack(), (1,))
    source = torch.empty(7, dtype=torch.int8)
    arguments, implicit = adapter._prepare_native_arguments((source, extent), device="cpu")
    assert arguments[0] is source
    assert arguments[1].shape == (extent,)
    assert arguments[2] == extent
    assert implicit == []


def test_explicit_shape_scalar_still_checks_subbyte_storage_capacity():
    adapter = _adapter(_explicit_shape_unpack(), (1,))
    with pytest.raises(ValueError, match="needs byte range"):
        adapter._prepare_native_arguments((torch.empty(7, dtype=torch.int8), 15), device="cpu")


def test_explicit_stride_scalar_is_a_logical_subbyte_stride():
    from tvm_tirx import ir, tirx
    from tvm_tirx.tirx.layout import S

    extent, stride = ir.Var("extent", "int64"), ir.Var("stride", "int64")
    tensor = ir.Var("A", tirx.TensorType([extent], "int4", layout=tirx.TileLayout(S[[extent] : [stride]])))
    function = tirx.Function([tensor, extent, stride], [], ir.PrimType("void"), ir.DictAttrs({"global_symbol": "main"}))
    adapter = _adapter(function)
    source = torch.empty(7, dtype=torch.int8)
    arguments, implicit = adapter._prepare_native_arguments((source, 4, 3), device="cpu")
    assert arguments == [source, 4, 3]
    assert implicit == []
    with pytest.raises(ValueError, match="needs byte range"):
        adapter._prepare_native_arguments((source, 5, 4), device="cpu")


@pytest.mark.parametrize("explicit_stride", [False, True])
def test_composite_subbyte_layout_uses_explicit_scalars(explicit_stride):
    adapter = _adapter(_affine_unpack(explicit_stride), (1,))
    # A[2*N] has logical stride 2*S+1 when S is supplied. Its byte
    # carrier need not have either the logical shape or logical stride.
    cells = 7 if not explicit_stride else 20
    scalars = (7,) if not explicit_stride else (7, 1)
    source = torch.empty(cells, dtype=torch.int8)
    arguments, implicit = adapter._prepare_native_arguments((source, *scalars), device="cpu")
    assert arguments[0] is source
    assert arguments[1].shape == (7,)
    assert arguments[2:] == list(scalars)
    assert implicit == []
    with pytest.raises(ValueError, match="needs byte range"):
        adapter._prepare_native_arguments((torch.empty(cells - 1, dtype=torch.int8), *scalars), device="cpu")


def test_composite_subbyte_layout_uses_later_bare_metadata_carriers():
    extent, stride = T.dynamic("extent", "int64"), T.dynamic("stride", "int64")

    @T.prim_func
    def main(
        A: T.StridedTensor((extent * 2,), (stride * 2 + 1,), "int4"),
        R: T.StridedTensor((extent,), (stride,), "int32"),
        B: T.Tensor((extent,), "int32"),
    ):
        T.evaluate(0)

    adapter = _adapter(main, (2,))
    source = torch.empty(33, dtype=torch.int8)
    carrier = torch.empty_strided((7,), (2,), dtype=torch.int32)
    arguments, implicit = adapter._prepare_native_arguments((source, carrier), device="cpu")
    assert arguments[0] is source and arguments[1] is carrier
    assert arguments[2].shape == (7,)
    assert implicit == [7, 2]
    with pytest.raises(ValueError, match="needs byte range"):
        adapter._prepare_native_arguments((torch.empty(32, dtype=torch.int8), carrier), device="cpu")


@pytest.mark.parametrize("field", ["shape", "stride"])
def test_composite_subbyte_layout_does_not_invent_an_unbound_symbol(field):
    unknown = T.dynamic("unknown", "int64")
    annotation = T.Tensor((unknown * 2,), "int4") if field == "shape" else T.StridedTensor((4,), (unknown * 2,), "int4")

    @T.prim_func
    def main(A: annotation):
        T.evaluate(0)

    source = torch.empty(14, dtype=torch.int8) if field == "shape" else torch.empty_strided((4,), (3,), dtype=torch.int8)
    # The native packed decoder could obtain unknown=7/3 from these carrier
    # metadata, but old NVRTC only reads a shape or stride that is a bare Var.
    with pytest.raises(ValueError, match=f"Cannot resolve parameter 0 {field}"):
        _adapter(main)._prepare_native_arguments((source,), device="cpu")


def test_numpy_view_capacity_uses_its_owner_allocation():
    from tilelang.jit.adapter.abi import SourceABIPlan

    plan = SourceABIPlan(_static_unpack(1))
    backing = np.empty(8, dtype="int8")
    plan.bind_arguments([backing[3:4], None], nvrtc_storage=True)
    with pytest.raises(ValueError, match="storage provides"):
        plan.bind_arguments([backing[4:5], None], nvrtc_storage=True)


@tilelang.testing.requires_cuda
def test_nvrtc_static_minimum_packed_input_numeric(tmp_path):
    kernel = _compile(_static_unpack(), tmp_path)
    source = torch.tensor([0xF1, 0x27, 0x80, 0xD3], dtype=torch.uint8, device="cuda").view(torch.int8)
    expected = torch.tensor(_signed_nibbles(source), dtype=torch.int32, device="cuda")
    torch.testing.assert_close(kernel(source), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_nvrtc_explicit_shape_scalar_numeric(tmp_path):
    kernel = _compile(_explicit_shape_unpack(), tmp_path)
    source = torch.tensor([0xF1, 0x27, 0x80, 0xD3, 0x5A, 0x69, 0xBC], dtype=torch.uint8, device="cuda").view(torch.int8)
    expected = torch.tensor(_signed_nibbles(source), dtype=torch.int32, device="cuda")
    actual = kernel(source, 14)
    assert actual.shape == (14,)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("explicit_stride", [False, True])
def test_nvrtc_composite_shape_and_stride_scalars_numeric(explicit_stride, tmp_path):
    kernel = _compile(_affine_unpack(explicit_stride), tmp_path)
    cells = 7 if not explicit_stride else 20
    source = (torch.arange(cells, dtype=torch.uint8, device="cuda") * 13 + 7).view(torch.int8)
    scalars = (7,) if not explicit_stride else (7, 1)
    logical_stride = 1 if not explicit_stride else 3
    nibbles = _signed_nibbles(source)
    expected = torch.tensor([nibbles[index * 2 * logical_stride] for index in range(7)], dtype=torch.int32, device="cuda")
    actual = kernel(source, *scalars)
    assert actual.shape == (7,)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("explicit_stride", [False, True])
def test_nvrtc_dynamic_shape_stride_and_view_offset_numeric(explicit_stride, tmp_path):
    kernel = _compile(_dynamic_unpack(explicit_stride), tmp_path)
    step = 3 if explicit_stride else 1
    storage = torch.arange(5 + 7 * step, dtype=torch.uint8, device="cuda") * 13
    source = storage[5::step].view(torch.int8)
    assert source.shape == (7,)
    assert source.storage_offset() == 5
    expected_values = (
        [((value & 15) ^ 8) - 8 for value in source.cpu().to(torch.int32).tolist()] if explicit_stride else _signed_nibbles(source)[:7]
    )
    expected = torch.tensor(expected_values, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    actual = kernel(source, stream=stream.cuda_stream)
    stream.synchronize()
    assert actual.shape == (7,)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_nvrtc_layout_and_storage_offsets_are_each_applied_once(tmp_path):
    kernel = _compile(_static_unpack(1), tmp_path)
    source = torch.tensor([1, 2, 3, 0xF1, 0x27, 0x80, 0xD3, 0x5A], dtype=torch.uint8, device="cuda")[3:].view(torch.int8)
    assert source.storage_offset() == 3
    expected = torch.tensor(_signed_nibbles(source)[1:9], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(kernel(source), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dynamic,cells", [(False, 8), (True, 7), (True, 8)])
def test_nvrtc_subbyte_output_shape_and_written_nibbles_numeric(dynamic, cells, tmp_path):
    kernel = _compile(_pack(dynamic), tmp_path)
    source = torch.arange(cells, dtype=torch.int32, device="cuda") - 4
    output = kernel(source)
    assert output.dtype == torch.int8
    assert output.shape == (cells,)
    # Unaddressed output bytes and the high nibble after an odd extent are
    # deliberately unspecified. Assert every logical element actually written.
    assert _signed_nibbles(output)[:cells] == source.cpu().tolist()
