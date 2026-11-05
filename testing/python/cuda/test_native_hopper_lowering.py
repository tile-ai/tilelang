"""Native TMA setup lives on the host and is captured through real IR uses."""

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, sym, tirx
from tvm_tirx.tirx.layout import S, TileLayout


def _nodes(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="pre")
    return result


def _calls(value, name):
    return [call for call in _nodes(value, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == name]


def _tensor(name="A", *, size=128, dtype="float16", offset=0):
    count = size * 64
    return ir.Var(name, tirx.TensorType((count,), dtype, layout=TileLayout(S[count:1] + offset)))


def _descriptor(pointer, size, *, im2col=False):
    name = "tl.create_tma_im2col_descriptor" if im2col else "tl.create_tma_descriptor"
    values = (
        [6, 3, pointer, 64, size, 1, 2, 128, size * 128, 1, 1, 1, -1, 1, 16, 16, 0, 0, 0, 0]
        if im2col
        else [6, 2, pointer, size, 64, 2, size * 2, 16, 16, 1, 1, 0, 0, 0, 0]
    )
    values = [value if isinstance(value, ir.Expr) else ir.IntImm("int32", value) for value in values]
    return ir.Call(name, values, ty_args=[ir.PtrType(ir.PrimType("void"))])


def _launch(body, *, thread=None, tag="threadIdx.x", extent=32):
    thread = ir.Var("thread", "int32") if thread is None else thread
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), ir.IntImm("int32", extent)], [thread], {}, body)


def _function(params, body, *, attrs=None, ret_type="void"):
    fields = {"global_symbol": "main", "target": tvm.target.Target({"kind": "cuda", "arch": "sm_90a"}, host="c")}
    fields.update(attrs or {})
    result = tirx.Function(params, body, ir.PrimType(ret_type), attrs=ir.DictAttrs(fields))
    assert tirx.analysis.check_well_formed(result)
    return result


def _apply(function, tmp_path, *, disable_elect=False):
    before = ir.save_json(function)
    with tvm.transform.PassContext(config={"tl.disable_shuffle_elect": disable_elect}):
        result = tvm_ffi.get_global_func("tl.cuda.transform.LowerHopperIntrin")()(tvm.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    (tmp_path / "before.json").write_text(before)
    (tmp_path / "after.json").write_text(ir.save_json(result))
    (tmp_path / "after.py").write_text(result.script())
    return result


@pytest.mark.parametrize("disable_elect", [False, True])
@pytest.mark.parametrize("im2col", [False, True])
def test_descriptor_typed_setup_dynamic_args_and_prefetch_capture(disable_elect, im2col, tmp_path):
    size = ir.Var("size", "int64")
    source = _tensor(size=size)
    original = _descriptor(source.data, size, im2col=im2col)
    thread = ir.Var("tx", "int32")
    before = _function([size, source], [_launch([ir.Evaluate(original)], thread=thread)])
    result = _apply(before, tmp_path, disable_elect=disable_elect)
    assert len(result.body.seq) == 3
    binding, initialization, launch = result.body.seq
    assert isinstance(binding, ir.Bind)
    assert isinstance(binding.var.ty, ir.PtrType) and isinstance(binding.var.ty.element_type, tirx.TensorMapType)
    assert binding.value.op.name == "tirx.stack_alloca"
    assert binding.value.args[0].value == "tensormap" and int(binding.value.args[1]) == 1
    call = initialization.value
    assert call.op.name == "tirx.call_packed"
    assert call.args[0].value == ("__tvm_tensormap_create_im2col" if im2col else "__tvm_tensormap_create_tiled")
    assert call.args[1].same_as(binding.var)
    assert tvm_ffi.structural_equal(list(call.args[2:]), list(original.args))
    assert tvm_ffi.structural_equal(result.attrs["tma_descriptor_args"][binding.var], call.args)
    assert not _calls(result.body, original.op.name)
    (prefetch,) = _calls(launch, "tl.prefetch_tma_descriptor")
    assert prefetch.args[0].same_as(binding.var)
    guard = launch.body.seq[0]
    assert isinstance(guard, ir.If)
    if disable_elect:
        assert tvm_ffi.structural_equal(guard.condition, thread == 0)
    else:
        assert guard.condition.op.name == "tl.tl_shuffle_elect"
    assert any(value.same_as(binding.var) for value in tirx.analysis.undefined_vars(launch))


def test_local_global_allocation_dominates_descriptor_setup(tmp_path):
    source = _tensor()
    allocation = ir.Bind(source, ir.Call("tirx.alloc_tensor", [], ty_args=[source.ty]))
    descriptor = _descriptor(source.data, ir.IntImm("int32", 128))
    before = _function([], [allocation, _launch([ir.Evaluate(descriptor)])])
    result = _apply(before, tmp_path)
    assert result.body.seq[0].var.same_as(source)
    descriptor_binding = result.body.seq[1]
    assert isinstance(descriptor_binding.var.ty.element_type, tirx.TensorMapType)
    assert result.body.seq[2].value.args[4].args[0].same_as(source)
    assert isinstance(result.body.seq[3], ir.RegionStmt)


def test_descriptor_dedup_is_local_to_each_host_launch_scope(tmp_path):
    source = _tensor()
    condition = ir.Var("condition", "bool")
    descriptor = _descriptor(source.data, ir.IntImm("int32", 128))

    def make_body():
        return [_launch([ir.Evaluate(descriptor), ir.Evaluate(descriptor)])]

    before = _function([source, condition], [ir.If(condition, make_body(), make_body())])
    result = _apply(before, tmp_path)
    branch = result.body.seq[0]
    assert isinstance(branch, ir.If)
    assert len(branch.then_case.seq) == len(branch.else_case.seq) == 3
    first, second = branch.then_case.seq[0].var, branch.else_case.seq[0].var
    assert not first.same_as(second)
    assert len(_calls(result.body, "tirx.call_packed")) == 2
    assert len(_calls(result.body, "tl.prefetch_tma_descriptor")) == 2
    assert len(result.attrs["tma_descriptor_args"]) == 2


def test_device_pure_aliases_expand_to_host_operands(tmp_path):
    size = ir.Var("size", "int64")
    source = _tensor(size=size)
    local_size = ir.Var("local_size", "int64")
    local_pointer = ir.Var("local_pointer", source.data.ty)
    descriptor = _descriptor(local_pointer, local_size)
    before = _function(
        [size, source],
        [_launch([ir.Bind(local_size, size + 16), ir.Bind(local_pointer, source.data), ir.Evaluate(descriptor)])],
    )
    result = _apply(before, tmp_path)
    (initialize,) = _calls(result.body, "tirx.call_packed")
    assert tvm_ffi.structural_equal(initialize.args[4], source.data)
    assert tvm_ffi.structural_equal(initialize.args[5], size + 16)
    free = tirx.analysis.undefined_vars(initialize)
    assert not any(value.same_as(local_size) or value.same_as(local_pointer) for value in free)


def test_descriptor_bind_alias_keeps_one_native_identity(tmp_path):
    source = _tensor()
    descriptor = _descriptor(source.data, ir.IntImm("int32", 128))
    named = ir.Var("named_descriptor", descriptor.ty)
    before = _function([source], [ir.Bind(named, descriptor), _launch([ir.Evaluate(named)])])
    result = _apply(before, tmp_path)
    (allocation,) = _calls(result.body, "tirx.stack_alloca")
    bindings = [binding for binding in _nodes(result.body, ir.Bind) if binding.value.same_as(allocation)]
    assert len(bindings) == 1
    (prefetch,) = _calls(result.body, "tl.prefetch_tma_descriptor")
    assert prefetch.args[0].same_as(bindings[0].var)
    assert len(result.attrs["tma_descriptor_args"]) == 1


def test_device_thread_index_cannot_escape_into_host_descriptor(tmp_path):
    source = _tensor()
    thread = ir.Var("thread", "int32")
    descriptor = _descriptor(source.data, thread + 32)
    before = _function([source], [_launch([ir.Evaluate(descriptor)], thread=thread)])
    with pytest.raises(ValueError, match="depends on a device-local value"):
        _apply(before, tmp_path)


def test_device_memory_read_cannot_be_hoisted_to_host(tmp_path):
    source = _tensor(dtype="int32")
    extent = ir.Var("extent", "int32")
    descriptor = _descriptor(source.data, extent)
    before = _function([source], [_launch([ir.Bind(extent, ir.TensorLoad(source, [ir.IntImm("int32", 0)])), ir.Evaluate(descriptor)])])
    with pytest.raises(ValueError, match="depends on a device-local value"):
        _apply(before, tmp_path)


def test_host_split_captures_tensormap_without_descriptor_metadata(tmp_path):
    size = ir.Var("size", "int64")
    source = _tensor(size=size)
    descriptor = _descriptor(source.data, size)
    before = _function([size, source], [_launch([_launch([ir.Evaluate(descriptor)])], tag="blockIdx.x", extent=1)])
    result = _apply(before, tmp_path)
    # The projection is not an input to closure conversion. Erasing it must
    # leave the exact same actual descriptor and dynamic inputs available.
    without_projection = result.without_attr("tma_descriptor_args")
    split = tirx.transform.SplitHostDevice()(tvm.IRModule({"main": without_projection}))
    assert len(split.functions) == 2
    assert tirx.analysis.check_well_formed(split)
    (tmp_path / "split.py").write_text(split.script())
    device = next(function for _, function in split.functions.items() if function.attrs.get("calling_conv", 0) == 2)
    host = split["main"]
    assert len(device.params) == 1 and isinstance(device.params[0].ty.element_type, tirx.TensorMapType)
    (initializer,) = _calls(host.body, "tirx.call_packed")
    (launch,) = _calls(host.body, "tirx.call_ffi_kernel")
    assert initializer.args[1].same_as(launch.args[1])
    assert len(_calls(device.body, "tirx.call_packed")) == 0
    assert len(_calls(device.body, "tl.prefetch_tma_descriptor")) == 1
    source_module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        tvm.IRModule({"main_kernel": device}), device.attrs["target"]
    )
    generated = source_module.inspect_source("cuda")
    (tmp_path / "kernel.cu").write_text(generated)
    assert "__grid_constant__ const CUtensorMap" in generated
    assert "tl::prefetch_tma_descriptor(" in generated


@pytest.mark.parametrize("dtype,offset,byte_offset", [("float16", 3, 6), ("float32", 7, 28), ("uint8", 11, 11)])
def test_l2_policy_pointer_includes_native_tensor_origin(dtype, offset, byte_offset, tmp_path):
    source = _tensor(dtype=dtype, offset=offset)
    attrs = {"l2_persistent_map": {"A": [ir.FloatImm("float64", 0.75), ir.IntImm("int64", 256)]}}
    before = _function([source], [_launch([])], attrs=attrs)
    result = _apply(before, tmp_path)
    calls = _calls(result.body, "tirx.call_packed")
    assert [call.args[0].value for call in calls] == [
        "__tvm_cuda_stream_set_access_policy_window",
        "__tvm_cuda_stream_reset_access_policy_window",
    ]
    pointer = calls[0].args[1]
    assert pointer.op.name == "tirx.handle_add_byte_offset"
    assert pointer.ty == ir.PtrType(ir.PrimType("void"))
    assert int(sym.Analyzer().simplify(pointer.args[1])) == byte_offset
    assert int(calls[0].args[2]) == 256 and float(calls[0].args[3].value) == 0.75
    assert not _nodes(calls[0], ir.TensorLoad)


def test_l2_reset_precedes_host_early_return(tmp_path):
    source = _tensor()
    condition = ir.Var("condition", "bool")
    attrs = {"l2_persistent_map": {"A": [ir.FloatImm("float64", 0.5), ir.IntImm("int64", 256)]}}
    before = _function([source, condition], [ir.If(condition, [ir.Return(ir.IntImm("int32", 0))], None)], attrs=attrs, ret_type="int32")
    result = _apply(before, tmp_path)
    branch = next(statement for statement in result.body.seq if isinstance(statement, ir.If))
    assert branch.then_case.seq[0].value.args[0].value == "__tvm_cuda_stream_reset_access_policy_window"
    assert isinstance(branch.then_case.seq[1], ir.Return)


def test_repeated_lowering_preserves_descriptor_and_policy_counts(tmp_path):
    source = _tensor()
    descriptor = _descriptor(source.data, ir.IntImm("int32", 128))
    attrs = {"l2_persistent_map": {"A": [ir.FloatImm("float64", 0.5), ir.IntImm("int64", 256)]}}
    before = _function([source], [_launch([ir.Evaluate(descriptor)])], attrs=attrs)
    first = _apply(before, tmp_path)
    second = _apply(first, tmp_path)
    assert len(_calls(second.body, "tirx.stack_alloca")) == 1
    assert len(_calls(second.body, "tl.prefetch_tma_descriptor")) == 1
    names = [call.args[0].value for call in _calls(second.body, "tirx.call_packed")]
    assert names == [
        "__tvm_cuda_stream_set_access_policy_window",
        "__tvm_tensormap_create_tiled",
        "__tvm_cuda_stream_reset_access_policy_window",
    ]
    assert tvm_ffi.structural_equal(first, second)


@pytest.mark.parametrize("address_op", ["tirx.address_of", "tl.access_ptr"])
def test_address_lvalue_is_not_a_device_memory_read(address_op, tmp_path):
    source = _tensor(dtype="int32")
    load = ir.TensorLoad(source, [ir.IntImm("int32", 16)])
    operands = [load] if address_op == "tirx.address_of" else [load, ir.IntImm("int32", 32), ir.IntImm("int32", 1)]
    pointer = ir.Call(address_op, operands)
    local_pointer = ir.Var("local_pointer", pointer.ty)
    descriptor = _descriptor(local_pointer, ir.IntImm("int32", 128))
    before = _function([source], [_launch([ir.Bind(local_pointer, pointer), ir.Evaluate(descriptor)])])
    result = _apply(before, tmp_path)
    (initialize,) = _calls(result.body, "tirx.call_packed")
    assert tvm_ffi.structural_equal(initialize.args[4], pointer)
    assert not any(variable.same_as(local_pointer) for variable in tirx.analysis.undefined_vars(initialize))


@pytest.mark.parametrize("address_kind", ["data", "address_of", "access_ptr"])
def test_parameter_backed_device_view_uses_a_typed_host_alias(address_kind, tmp_path):
    size = ir.Var("size", "int64")
    source = _tensor(size=size)
    local_size = ir.Var("local_size", "int64")
    alias = _tensor("view", size=local_size, offset=11)
    declare = ir.Call("tirx.decl_tensor", [source.data], ty_args=[alias.ty])
    load = ir.TensorLoad(alias, [ir.IntImm("int32", 7)])
    if address_kind == "data":
        pointer = alias.data
    elif address_kind == "address_of":
        pointer = ir.Call("tirx.address_of", [load])
    else:
        pointer = ir.Call("tl.access_ptr", [load, ir.IntImm("int32", 8), ir.IntImm("int32", 1)])
    descriptor = _descriptor(pointer, size)
    before = _function([size, source], [_launch([ir.Bind(local_size, size + 16), ir.Bind(alias, declare), ir.Evaluate(descriptor)])])
    result = _apply(before, tmp_path)
    host_alias, desc_bind, initialize, launch = result.body.seq
    assert host_alias.value.op.name == "tirx.decl_tensor"
    assert tvm_ffi.structural_equal(host_alias.value.args[0], source.data)
    assert tvm_ffi.structural_equal(host_alias.var.ty.shape[0], (size + 16) * 64)
    assert not host_alias.var.same_as(alias)
    assert tvm_ffi.structural_equal(host_alias.var.ty.layout, TileLayout(S[(size + 16) * 64 : 1] + 11))
    assert isinstance(desc_bind.var.ty.element_type, tirx.TensorMapType)
    assert any(variable.same_as(host_alias.var) for variable in tirx.analysis.undefined_vars(initialize))
    assert any(binding.var.same_as(alias) for binding in _nodes(launch, ir.Bind))
    assert len(result.attrs["tma_descriptor_args"]) == 1


def test_unused_device_view_does_not_create_a_host_binding(tmp_path):
    source = _tensor()
    alias = _tensor("alias")
    declare = ir.Call("tirx.decl_tensor", [source.data], ty_args=[alias.ty])
    before = _function([source], [_launch([ir.Bind(alias, declare)])])
    result = _apply(before, tmp_path)
    assert tvm_ffi.structural_equal(result.body, before.body)
