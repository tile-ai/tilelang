"""Ascend source hints and MX bindings survive native tensor construction."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang.ascend import language as T
from tilelang.testing.ir import collect_nodes


def calls(function, name):
    return [node for node in collect_nodes(function, ir.Call) if node.op.name == name]


def annotations(call):
    operand_count = int(call.attrs.get("operand_count", 0))
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[operand_count], call.attrs["annotation_schema"])


@pytest.mark.parametrize("l0_scope", ["shared.l0a", "shared.l0b"])
@pytest.mark.parametrize("pitch", [2, T.int32(2)])
def test_ascend_l0_copy_keeps_native_hint_payload(l0_scope, pitch):
    @T.prim_func
    def kernel(source: T.Tensor((16, 32), "float16", scope="shared.l1")):
        destination = T.alloc_buffer((16, 32), "float16", scope=l0_scope)
        T.copy(source, destination, l0_pitch=pitch, coalesced_width=4, annotations={"unit_flag_ctrl": 3})

    (copy,) = calls(kernel, "tl.tileop.ascend_copy")
    assert int(copy.attrs["operand_count"]) == 2
    assert len(copy.args) == 3
    metadata = annotations(copy)
    assert int(metadata["l0_pitch"]) == 2
    assert int(metadata["coalesced_width"]) == 4
    assert int(metadata["unit_flag_ctrl"]) == 3
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("data_select", [False, True])
def test_ascend_pad_write_precedes_copy_and_keeps_dtype(data_select):
    @T.prim_func
    def kernel(source: T.Tensor((16,), "float16")):
        destination = T.alloc_shared((16,), "float16")
        if data_select:
            T.ascend_set_copy_pad_value(T.float16(-2.5))
            T.copy(source, destination, data_select=True)
        else:
            T.copy(source, destination, pad_value=-2.5)

    top_calls = [node.value for node in kernel.body.seq if isinstance(node, ir.Evaluate)]
    pad = [call for call in top_calls if call.op.name == "tl.ascend_set_copy_pad_value"]
    copy = [call for call in top_calls if call.op.name == "tl.tileop.ascend_copy"]
    assert len(pad) == len(copy) == 1
    assert top_calls.index(pad[0]) < top_calls.index(copy[0])
    assert str(pad[0].args[0].ty.dtype) == "float16"
    value = sym.Analyzer().simplify(pad[0].args[0])
    assert isinstance(value, ir.FloatImm)
    assert float(value.value) == -2.5
    assert int(annotations(copy[0])["data_select"]) == 1
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("leading", [(), (3,), (2, 3)])
@pytest.mark.parametrize("suffix", ["a", "b"])
def test_ascend_mx_scale_handle_keeps_typed_binding(leading, suffix):
    alloc_data = T.alloc_l0a if suffix == "a" else T.alloc_l0b
    alloc_scale = T.alloc_l0a_sf if suffix == "a" else T.alloc_l0b_sf

    @T.prim_func
    def kernel():
        data = alloc_data((*leading, 32, 128), "float8_e4m3fn")
        scale = alloc_scale(data)  # noqa: F841 - the tensor binding is inspected below
        T.evaluate(0)

    allocations = [binding.var for binding in collect_nodes(kernel, ir.Bind) if isinstance(binding.var.ty, tirx.TensorType)]
    data, scale = allocations
    assert str(data.ty.scope) == "shared.l0" + suffix
    assert str(scale.ty.scope) == "shared.l0" + suffix + ".sf"
    assert [int(extent) for extent in scale.ty.shape] == [*leading, 32, 2]
    (annotation,) = calls(kernel, "tl.annotation")
    bindings = annotations(annotation)["tl.l0_sf_bindings"]
    assert len(bindings) == 1
    assert next(iter(bindings)).same_as(scale)
    assert bindings[scale].same_as(data)
    assert tirx.analysis.check_well_formed(kernel)


def test_ascend_blockscaled_call_has_sixteen_data_operands_and_one_payload():
    @T.prim_func
    def kernel():
        a = T.alloc_l0a((16, 128), "float8_e4m3fn")
        b = T.alloc_l0b((32, 128), "float8_e4m3fn")
        c = T.alloc_l0c((16, 32), "float32", layout=False)
        sa = T.alloc_l0a_sf(a)
        sb = T.alloc_l0b_sf(b)
        T.gemm_blockscaled(a, b, c, sa, sb, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)

    (gemm,) = calls(kernel, "tl.tileop.gemm_blockscaled")
    assert int(gemm.attrs["operand_count"]) == 16
    assert len(gemm.args) == 17
    assert isinstance(gemm.args[13], ir.TensorRegion)
    assert isinstance(gemm.args[14], ir.TensorRegion)
    assert str(gemm.args[13].source.ty.scope) == "shared.l0a.sf"
    assert str(gemm.args[14].source.ty.scope) == "shared.l0b.sf"
    metadata = annotations(gemm)
    assert int(metadata["unit_flag_ctrl"]) == 3
    assert int(metadata["sf_a_granularity_k"]) == int(metadata["sf_b_granularity_k"]) == 32
    assert tirx.analysis.check_well_formed(kernel)


def test_ascend_copy_rejects_mismatched_elements_and_unsupported_pitch():
    source = ir.Var("source", tirx.TensorType((16,), "float16"))
    destination = ir.Var("destination", tirx.TensorType((32,), "float16"))
    with pytest.raises(ValueError, match="element count mismatch"):
        T.copy(source, destination)
    with pytest.raises(ValueError, match="L1"):
        T.copy(source, source, l0_pitch=2)


def test_ascend_mx_scale_rejects_wrong_data_scope_and_shape():
    wrong_scope = ir.Var("wrong", tirx.TensorType((16, 128), "float8_e4m3fn", scope="shared.l1"))
    with pytest.raises(ValueError, match="shared.l0a"):
        T.alloc_l0a_sf(wrong_scope)
    wrong_k = ir.Var("wrong", tirx.TensorType((16, 65), "float8_e4m3fn", scope="shared.l0a"))
    with pytest.raises(ValueError, match="not divisible"):
        T.alloc_l0a_sf(wrong_k)


@pytest.mark.parametrize("roundtrip", ["direct", "json", "renew"])
def test_ascend_root_allocation_metadata_keeps_native_tensor_identity(roundtrip):
    @T.prim_func
    def kernel():
        a = T.alloc_l0a((16, 128), "float8_e4m3fn")
        _scale = T.alloc_l0a_sf(a)
        _accumulator = T.alloc_l0c((16, 32), "float32")

    if roundtrip == "json":
        kernel = ir.load_json(ir.save_json(kernel))
    elif roundtrip == "renew":
        kernel = tirx.renew_def(kernel)
    assert tirx.analysis.check_well_formed(kernel)
    tensors = {bind.var for bind in collect_nodes(kernel, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)}
    records = [annotations(call) for call in calls(kernel, "tl.annotation")]
    assert len(records) == 2
    assert all("tl.source_block_annotation" not in record for record in records)
    (scales,) = [record["tl.l0_sf_bindings"] for record in records if "tl.l0_sf_bindings" in record]
    (layouts,) = [record["layout_map"] for record in records if "layout_map" in record]
    assert len(scales) == len(layouts) == 1
    scale, data = next(iter(scales.items()))
    accumulator = next(iter(layouts))
    assert all(var in tensors for var in (scale, data, accumulator))
    assert str(scale.ty.scope) == "shared.l0a.sf"
    assert str(data.ty.scope) == "shared.l0a"
    assert str(accumulator.ty.scope) == "shared.l0c"


def test_ascend_allocation_metadata_keeps_explicit_owner_conflict_checks():
    with pytest.raises(ValueError, match="Try to merge two annotations with different values"):

        @T.prim_func
        def invalid():
            with T.sblock("owner"):
                a = T.alloc_l0a((16, 128), "float8_e4m3fn")
                other = T.alloc_l0a((16, 128), "float8_e4m3fn")
                scale = T.alloc_l0a_sf(a)
                T.sblock_attr({"tl.l0_sf_bindings": {scale: other}})
