"""Ascend copy producers keep native views, physical strides and DMA units.

The slice address tests execute through TileLang's C packed host. Ascend tests verify lowering and CCE
source generation without claiming CANN compilation or NPU execution.
"""

import re

import numpy as np
import pytest
import tvm_ffi
import tilelang
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.layout import (
    Layout,
    make_ascend_major_k_layout,
    make_strided_slice,
)

from testing.ascend._ir import substitute

from test_native_cuda_copy import calls, lower
from test_native_reducer_contracts import allocate, annotation, integer, objects, region, run_pass, tensor, tileop


def strided_tensor(name, shape, strides, origin=0, dtype="float16", scope="global"):
    memory = tirx.TileLayout(tirx.layout.S[tuple(shape) : tuple(strides)] + origin)
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=memory, align=32))


def subregion(value, starts, extents):
    return ir.TensorRegion(value, [ir.Range(start, start + extent) for start, extent in zip(starts, extents)], ir.TensorRegionType())


def function(params, body):
    return (
        tirx.Function(params, body, ir.PrimType("void"))
        .with_attr("target", tvm_tirx.target.Target("ascend"))
        .with_attr("global_symbol", "main")
    )


def constant(value):
    return int(tvm_tirx.sym.Analyzer().simplify(value))


def address_offset(access):
    """Evaluate the address contract independently of DMA parameter derivation."""
    if access.op.name == "tl.ptr_access":
        return constant(access.args[1])
    assert access.op.name == "tl.access_ptr"
    load = access.args[0]
    assert isinstance(load, ir.TensorLoad)
    memory = load.source.ty.layout
    if memory is None:
        memory = tirx.TileLayout(tirx.layout.S[tuple(load.source.ty.shape)])
    return constant(memory.apply(*load.indices, shape=load.source.ty.shape)["m"])


def emit_source(value, tmp_path):
    value = run_pass("LowerAccessPtr", value)
    value = tvm_ffi.get_global_func("tl.transform.LowerSourceAnnotations")()(ir.IRModule({"main": value}))["main"]
    params, declarations = [], []
    for var in value.params:
        if isinstance(var.ty, tirx.TensorType):
            pointer = ir.Var(var.name + "_ptr", ir.PtrType(var.ty.dtype, var.ty.scope))
            params.append(pointer)
            declarations.append(ir.Bind(var, ir.Call("tirx.decl_tensor", [pointer], ty_args=[var.ty])))
        else:
            params.append(var)
    value = tirx.Function(params, [*declarations, value.body], value.ret_type, value.attrs)
    module = ir.IRModule({"main": value})
    for transform in (tirx.transform.LowerTensorLayouts, tirx.transform.FlattenTensor, tirx.transform.StmtSimplify):
        module = transform()(module)
    value = module["main"].with_attr("calling_conv", 2)
    assert tirx.analysis.check_well_formed(value)
    build = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")
    code = build(ir.IRModule({"main": value}), value.attrs["target"]).inspect_source("asc")
    (tmp_path / "lowered.json").write_text(ir.save_json(value))
    (tmp_path / "kernel.asc").write_text(code)
    assert "tl.tileop" not in code
    return code


@pytest.mark.parametrize("dtype", ["float16", "float32", "int8"])
@pytest.mark.parametrize("nested", [False, True])
def test_strided_slice_executes_with_original_origin_and_physical_stride(dtype, nested):
    storage = tensor("storage", [4096], dtype, "global")
    physical = strided_tensor("physical", [4, 2, 16, 16], [1024, 320, 16, 1], 13, dtype)
    logical = tensor("logical", [32, 64], "float16", "global")
    layout = make_ascend_major_k_layout(logical)
    declaration = make_strided_slice(physical, layout, [ir.Range(16, 32), ir.Range(16, 48)])
    assert isinstance(declaration, ir.Call) and declaration.op.name == "tirx.decl_tensor"
    assert declaration.args[0].op.name == "tirx.tensor_data_ptr"
    assert declaration.args[0].args[0].same_as(physical)
    assert [int(dim) for dim in declaration.ty.shape] == [2, 1, 16, 16]
    assert declaration.ty.data_alignment == 32
    assert declaration.ty.scope.same_as(physical.ty.scope)
    view = ir.Var("slice", declaration.ty)
    body = [
        ir.Bind(physical, ir.Call("tirx.decl_tensor", [storage.data], ty_args=[physical.ty])),
        ir.Bind(view, declaration),
    ]
    indices = [1, 0, 3, 5]
    if nested:
        identity = Layout([2, 1, 16, 16], lambda *idx: list(idx))
        second = make_strided_slice(view, identity, [ir.Range(1, 2), ir.Range(0, 1), ir.Range(2, 5), ir.Range(3, 7)])
        view = ir.Var("nested", second.ty)
        body.append(ir.Bind(view, second))
        indices = [0, 0, 1, 2]
    expected_offset = 13 + 2 * 1024 + 320 + 3 * 16 + 5
    assert constant(view.ty.layout.apply(*indices, shape=view.ty.shape)["m"]) == expected_offset
    body.append(ir.TensorStore(view, indices, ir.const(7, dtype)))
    value = tirx.Function([storage], body, ir.PrimType("void")).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(value)
    executable = tilelang.compile(value, target="c", target_host="c", execution_backend="tvm_ffi")
    actual = tvm_tirx.runtime.tensor(np.zeros(4096, dtype=dtype))
    executable(actual)
    expected = np.zeros(4096, dtype=dtype)
    expected[expected_offset] = 7
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_slice_transpose_keeps_nonunit_innermost_stride():
    physical = strided_tensor("physical", [7, 3], [9, 2], 5)
    layout = Layout([3, 7], lambda i, j: [j, i])
    value = make_strided_slice(physical, layout, [ir.Range(1, 3), ir.Range(2, 5)])
    assert [int(dim) for dim in value.ty.shape] == [3, 2]
    assert constant(value.ty.layout.apply(integer(1), integer(1), shape=value.ty.shape)["m"]) == 36


def test_slice_inside_native_builder_defines_its_view():
    from tilelang import language as T

    layout = Layout([4, 8], lambda i, j: [i, j])

    @T.prim_func
    def kernel(data: T.Tensor((4, 8), "float32")):
        view = make_strided_slice(data, layout, [ir.Range(1, 3), ir.Range(2, 6)])
        view[1, 2] = T.float32(5)

    declarations = [bind for bind in objects(kernel, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)]
    assert len(declarations) == 1
    assert declarations[0].value.op.name == "tirx.decl_tensor"
    assert objects(kernel, ir.TensorStore)[0].dest.same_as(declarations[0].var)
    assert tirx.analysis.check_well_formed(kernel)
    assert not tirx.analysis.undefined_vars(kernel)


@pytest.mark.parametrize("dtype,element_bytes", [("float16", 2), ("float32", 4), ("int8", 1)])
@pytest.mark.parametrize("store", [False, True])
def test_dma_preserves_strided_rows_origins_and_element_addresses(dtype, element_bytes, store, tmp_path):
    gm = strided_tensor("gm", [4, 9], [32, 1], 7, dtype)
    ub = strided_tensor("ub", [4, 16], [40, 1], 3, dtype, "shared.dyn")
    gm_region = subregion(gm, [1, 2], [2, 5])
    ub_region = subregion(ub, [1, 4], [2, 5])
    source, destination = (ub_region, gm_region) if store else (gm_region, ub_region)
    value = function([gm], [allocate(ub), tileop("ascend_copy", source, destination)])
    result = lower(value, "ascend-strided-" + dtype + str(store))
    name = "tl.ascend_copy_ubuf_to_gm" if store else "tl.ascend_copy_gm_to_ubuf"
    (copy,) = calls(result, name)
    assert [address_offset(arg) for arg in copy.args[:2]] == ([41, 47] if store else [47, 41])
    assert [constant(arg) for arg in copy.args[2:5]] == [0, 2, 5 * element_bytes]
    if store:
        assert [constant(arg) for arg in copy.args[5:]] == [4, 32 * element_bytes, 40 * element_bytes]
    else:
        assert [constant(arg) for arg in copy.args[8:]] == [0, 32 * element_bytes, 40 * element_bytes]
    code = emit_source(result, tmp_path)
    assert ("asc_copy_ub2gm_align" if store else "asc_copy_gm2ub_align") in code
    assert re.search(r"\b41\b", code) and re.search(r"\b47\b", code)


def test_subblock_padding_uses_all_rows_and_allocation_pitch(tmp_path):
    gm = strided_tensor("gm", [2, 5], [9, 1], 0)
    ub = strided_tensor("ub", [2, 32], [40, 1], 0, scope="shared.dyn")
    value = function(
        [gm],
        [
            allocate(ub),
            ir.Evaluate(ir.Call("tl.ascend_set_copy_pad_value", [ir.const(-2.5, "float16")], ty_args=[ir.PrimType("void")])),
            tileop("ascend_copy", region(gm), subregion(ub, [0, 0], [2, 5]), metadata={"data_select": 1}),
        ],
    )
    result = lower(value, "ascend-pad-rows")
    (copy,) = calls(result, "tl.ascend_copy_gm_to_ubuf")
    assert [constant(arg) for arg in copy.args[3:]] == [2, 10, 0, 11, 1, 0, 18, 80]
    code = emit_source(result, tmp_path)
    assert code.index("asc_set_copy_pad_val") < code.index("asc_copy_gm2ub_align")


@pytest.mark.parametrize("scope,intrinsic", [("shared.l0a", "tl.ascend_load_cbuf_to_ca"), ("shared.l0b", "tl.ascend_load_cbuf_to_cb")])
def test_l0_pitch_retains_origin_and_full_pipeline_plane_stride(scope, intrinsic, tmp_path):
    source = strided_tensor("source", [2, 64, 128], [8192, 128, 1], 17, scope="shared.l1")
    destination = strided_tensor("destination", [2, 64, 128], [8192, 128, 1], 29, scope=scope)
    value = function(
        [],
        [
            allocate(source),
            allocate(destination),
            annotation({"layout_map": {source: make_ascend_major_k_layout(source), destination: make_ascend_major_k_layout(destination)}}),
            tileop(
                "ascend_copy",
                subregion(source, [1, 16, 32], [1, 32, 64]),
                subregion(destination, [1, 16, 32], [1, 32, 64]),
                metadata={"l0_pitch": 3},
            ),
        ],
    )
    result = lower(value, "ascend-l0-pitch-" + scope)
    (copy,) = calls(result, intrinsic)
    assert address_offset(copy.args[0]) == 29 + 8192 + 2 * (3 * 256) + 256
    assert address_offset(copy.args[1]) == 17 + 8192
    assert [constant(arg) for arg in copy.args[2:]] == [1, 2, 2, 4, 4, 3, 0]
    code = emit_source(result, tmp_path)
    assert "asc_copy_l12l0" in code
    # Raw destination access lowers to char-pointer arithmetic (bytes), while
    # the source tensor address remains half-pointer arithmetic (elements).
    assert re.search(r"\b20026\b", code) and re.search(r"\b8209\b", code)


def test_empty_dma_does_not_issue_a_transfer():
    gm, ub = tensor("gm", [4, 8], "float16", "global"), tensor("ub", [4, 8], "float16", "shared.dyn")
    value = function([gm], [allocate(ub), tileop("ascend_copy", subregion(gm, [4, 0], [0, 8]), subregion(ub, [0, 0], [0, 8]))])
    result = lower(value, "ascend-empty-dma")
    assert not calls(result, "tl.ascend_copy_gm_to_ubuf")


def test_dtype_reinterpreted_view_keeps_pointer_binding_and_byte_origin(tmp_path):
    raw = ir.Var("raw", ir.PtrType(ir.PrimType("uint8"), "global"))
    typed = ir.Var("typed", ir.PtrType(ir.PrimType("int32"), "global"))
    pointer = ir.Call("tirx.reinterpret", [raw], ty=typed.ty)
    pointer = tirx.op.ptr_byte_offset(pointer, integer(4), ty=typed.ty)
    gm = strided_tensor("view", [3, 8], [16, 1], 3, "int32")
    ub = tensor("ub", [1, 4], "int32", "shared.dyn")
    value = function(
        [raw],
        [
            ir.Bind(typed, pointer),
            ir.Bind(gm, ir.Call("tirx.decl_tensor", [typed], ty_args=[gm.ty])),
            allocate(ub),
            tileop("ascend_copy", subregion(gm, [1, 2], [1, 4]), region(ub)),
        ],
    )
    result = lower(value, "ascend-reinterpreted-view")
    (copy,) = calls(result, "tl.ascend_copy_gm_to_ubuf")
    assert address_offset(copy.args[1]) == 21
    (declaration,) = [bind for bind in objects(result, ir.Bind) if bind.var.same_as(gm)]
    assert declaration.value.args[0].same_as(typed)
    code = emit_source(result, tmp_path)
    assert "asc_copy_gm2ub_align" in code and "__gm__ char" in code
    assert re.search(r"\+\s*4\b", code) and re.search(r"\b21\b", code)


@pytest.mark.parametrize("extra_alias", [False, True])
def test_fp4_byte_backed_view_preserves_alias_chain_and_uses_packed_offsets(extra_alias):
    raw = ir.Var("raw", ir.PtrType(ir.PrimType("uint8"), "global"))
    typed = ir.Var("typed", ir.PtrType(ir.PrimType("float4_e2m1fn"), "global"))
    pointer = ir.Call("tirx.reinterpret", [raw], ty=typed.ty)
    pointer = tirx.op.ptr_byte_offset(pointer, integer(5), ty=typed.ty)
    declarations = [ir.Bind(typed, pointer)]
    backing = typed
    if extra_alias:
        backing = ir.Var("second_alias", typed.ty)
        declarations.append(ir.Bind(backing, typed))
    gm = strided_tensor("fp4_view", [4, 128], [160, 1], 6, "float4_e2m1fn")
    l1 = tensor("l1", [16, 64], "float4_e2m1fn", "shared.l1")
    value = function(
        [raw],
        [
            *declarations,
            ir.Bind(gm, ir.Call("tirx.decl_tensor", [backing], ty_args=[gm.ty])),
            allocate(l1),
            annotation({"layout_map": {l1: make_ascend_major_k_layout(l1)}}),
            tileop("ascend_copy", subregion(gm, [1, 16], [2, 32]), subregion(l1, [0, 0], [2, 32])),
        ],
    )
    result = lower(value, "ascend-fp4-byte-view-" + str(extra_alias))
    (copy,) = calls(result, "tl.ascend_copy_gm_to_cbuf")
    access = copy.args[1]
    assert access.op.name == "tl.ptr_access"
    ir.assert_structural_equal(access.ty, ir.PtrType(ir.PrimType("int8"), "global"))
    # The origin and row stride are FP4 elements; only the final DMA address
    # is in packed bytes. The separate five-byte backing origin stays bound.
    assert address_offset(access) == (6 + 160 + 16) // 2
    assert constant(access.args[2]) == 2 * 32 // 2
    assert access.args[0].op.name == "tirx.reinterpret"
    assert access.args[0].args[0].op.name == "tirx.tensor_data_ptr"
    assert access.args[0].args[0].args[0].same_as(gm)
    bindings = {bind.var: bind.value for bind in objects(result, ir.Bind)}
    assert bindings[gm].args[0].same_as(backing)
    ir.assert_structural_equal(bindings[typed], pointer)
    if extra_alias:
        assert bindings[backing].same_as(typed)
    assert [constant(copy.args[i]) for i in (3, 5, 6)] == [80, 2, 16]
    assert copy.args[-1].value == "int8_t"


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("cols", [3, 4])
def test_packed_fp4_dma_requires_whole_byte_rows(store, cols):
    gm = tensor("gm", [4, 64], "float4_e2m1fn", "global")
    ub = tensor("ub", [4, 64], "float4_e2m1fn", "shared.dyn")
    source, destination = (ub, gm) if store else (gm, ub)
    value = function(
        [gm], [allocate(ub), tileop("ascend_copy", subregion(source, [0, 0], [4, cols]), subregion(destination, [0, 0], [4, cols]))]
    )
    if cols % 2:
        with pytest.raises(Exception, match="packed sub-byte row must be byte-aligned"):
            lower(value, "ascend-fp4-invalid-row")
    else:
        result = lower(value, "ascend-fp4-whole-byte-" + str(store))
        (copy,) = calls(result, "tl.ascend_copy_ubuf_to_gm" if store else "tl.ascend_copy_gm_to_ubuf")
        assert [constant(arg) for arg in copy.args[3:5]] == [4, 2]


def test_dynamic_empty_dma_keeps_guard_and_fixed_axis_stride(tmp_path):
    rows = ir.Var("rows", "int32")
    gm = tensor("gm", [4, 2, 32], "float32", "global")
    ub = tensor("ub", [4, 32], "float32", "shared.dyn")
    value = function(
        [rows, gm],
        [allocate(ub), tileop("ascend_copy", subregion(gm, [0, 1, 0], [rows, 1, 32]), subregion(ub, [0, 0], [rows, 32]))],
    )
    result = lower(value, "ascend-dynamic-empty-dma")
    (copy,) = calls(result, "tl.ascend_copy_gm_to_ubuf")
    assert constant(copy.args[4]) == 128
    assert [constant(copy.args[i]) for i in (9, 10)] == [256, 128]
    assert address_offset(copy.args[1]) == 32
    (guard,) = objects(result, ir.If)
    for count in (0, 1, 4):
        analyzer = tvm_tirx.sym.Analyzer()
        assert bool(int(analyzer.simplify(substitute(guard.condition, {rows: count})))) == (count > 0)
    code = emit_source(result, tmp_path)
    assert "if (" in code and "asc_copy_gm2ub_align" in code


@pytest.mark.parametrize("pitch", [0, -1, 1, 5, ir.FloatImm("float32", 2.5)])
def test_invalid_l0_pitch_fails_before_emission(pitch):
    source = tensor("source", [64, 128], "float16", "shared.l1")
    destination = tensor("destination", [64, 128], "float16", "shared.l0a")
    value = function(
        [],
        [
            allocate(source),
            allocate(destination),
            annotation({"layout_map": {source: make_ascend_major_k_layout(source), destination: make_ascend_major_k_layout(destination)}}),
            tileop(
                "ascend_copy", subregion(source, [0, 0], [32, 64]), subregion(destination, [0, 0], [32, 64]), metadata={"l0_pitch": pitch}
            ),
        ],
    )
    with pytest.raises(ValueError, match="l0_pitch"):
        lower(value, "ascend-invalid-l0-pitch")
