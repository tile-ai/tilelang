"""WebGPU workgroup normalization preserves typed storage and lexical scopes.

These tests inspect native IR and real WGSL source. They do not execute a
browser or a WebGPU device.
"""

import importlib.util
from pathlib import Path
import re

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
from tilelang import transform
from tilelang.webgpu.codegen import _lower_tilelang_scopes, build_webgpu
from tilelang.webgpu.pipeline import _lower_workgroup_storage


def _objects(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="pre")
    return result


def _tensor(name, shape, dtype="float32", scope="shared.dyn", **kwargs):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, **kwargs))


def _alloc(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _scope(body, explicit=True):
    attrs = {"explicit_alloc_scope": True} if explicit else {}
    return ir.RegionStmt("tl.lexical_scope", [], [], attrs, body)


def _function(body, params=(), target="webgpu", *, device=True):
    # Device codegen consumes pointer parameters with explicit tensor views.
    # Host functions keep tensor parameters until the ordinary ABI passes run.
    if device:
        lowered_params, prefix = [], []
        for param in params:
            if isinstance(param.ty, tirx.TensorType):
                pointer = ir.Var(param.name + "_ptr", ir.PtrType(param.ty.dtype, scope=param.ty.scope))
                lowered_params.append(pointer)
                prefix.append(ir.Bind(param, ir.Call("tirx.decl_tensor", [pointer], ty_args=[param.ty])))
            else:
                lowered_params.append(param)
        params, body = lowered_params, [*prefix, *body]
    result = tirx.Function(list(params), body, ir.PrimType("void")).with_attr("global_symbol", "main")
    if target is not None:
        result = result.with_attr("target", tvm_tirx.target.Target(target))
    if device:
        result = result.with_attr("calling_conv", 2).with_attr("tirx.is_global_func", True)
    return result


def _normalize(function):
    before = ir.save_json(function)
    try:
        return _lower_workgroup_storage(ir.IRModule({"main": function}))["main"]
    finally:
        assert ir.save_json(function) == before


def _build(function, tmp_path, *, normalize=True, target=None):
    function = _normalize(function) if normalize else function
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    module = ir.IRModule({"main": function})
    target = tvm_tirx.target.Target(target) if target is not None else function.attrs["target"]
    source = build_webgpu(module, target).inspect_source("wgsl")
    (tmp_path / "device.json").write_text(ir.save_json(module))
    (tmp_path / "kernel.wgsl").write_text(source)
    return source


def test_normalize_preserves_shapes_layouts_locations_and_alias_identity():
    location = ir.SourceLoc(ir.SourceName("workgroup-contract"), 3, 2, 9, 11)
    layout = tirx.TileLayout(tirx.layout.S[(2, 3) : (7, 2)] + 1)
    storage = _tensor("scratch", [32], data_alignment=64, loc=location)
    pointer = ir.Var("pointer", ir.PtrType("float32", scope="shared.dyn", loc=location))
    view = _tensor("view", [2, 3], layout=layout, data_alignment=32, loc=location)
    second_view = _tensor("second_view", [2, 3], layout=layout, data_alignment=32, loc=location)
    local = _tensor("local", [1], scope="local", data_alignment=16)
    shadow = _tensor("scratch", [8], "float16", data_alignment=16)
    function = _function(
        [
            _alloc(storage),
            ir.Bind(pointer, storage.data),
            ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
            ir.Bind(second_view, ir.Call("tirx.decl_tensor", [view.data], ty_args=[second_view.ty])),
            _alloc(local),
            _scope(
                [
                    _alloc(shadow),
                    ir.TensorStore(shadow, [0], ir.FloatImm("float16", 7)),
                    ir.TensorStore(second_view, [1, 2], ir.TensorLoad(storage, [4])),
                ]
            ),
            ir.TensorStore(storage, [0], ir.TensorLoad(second_view, [1, 2])),
            ir.Evaluate(second_view.data),
        ]
    )
    assert tirx.analysis.check_well_formed(function)
    normalized = _normalize(function)
    assert tirx.analysis.check_well_formed(normalized)
    assert not tirx.analysis.undefined_vars(normalized)
    bindings = _objects(normalized, ir.Bind)
    actual_storage, actual_pointer, actual_view, actual_second, actual_local, actual_shadow = [item.var for item in bindings]
    originals = [storage, pointer, view, second_view, local, shadow]
    for original, binding in zip(originals, bindings):
        actual = binding.var
        assert actual.name == original.name
        assert str(actual.ty.scope) == ("local" if original.same_as(local) else "shared")
        ir.assert_structural_equal(actual.ty.loc, original.ty.loc)
        if isinstance(original.ty, tirx.TensorType):
            ir.assert_structural_equal(actual.ty.shape, original.ty.shape)
            ir.assert_structural_equal(actual.ty.dtype, original.ty.dtype)
            ir.assert_structural_equal(actual.ty.layout, original.ty.layout)
            assert actual.ty.data_alignment == original.ty.data_alignment
        else:
            ir.assert_structural_equal(actual.ty.element_type, original.ty.element_type)
        ir.assert_structural_equal(binding.value.ty, actual.ty)
    assert actual_storage.name == actual_shadow.name
    assert not actual_storage.same_as(actual_shadow)
    assert bindings[1].value.args[0].same_as(actual_storage)
    assert bindings[2].value.args[0].same_as(actual_pointer)
    assert bindings[3].value.args[0].args[0].same_as(actual_view)
    stores = _objects(normalized, ir.TensorStore)
    assert stores[0].dest.same_as(actual_shadow)
    assert stores[1].dest.same_as(actual_second)
    assert stores[1].value.source.same_as(actual_storage)
    assert stores[2].dest.same_as(actual_storage)
    assert stores[2].value.source.same_as(actual_second)
    assert actual_local.ty.scope.name == "local"
    assert all(ty.scope.name != "shared.dyn" for cls in (tirx.TensorType, ir.PtrType) for ty in _objects(normalized, cls))
    ir.assert_structural_equal(_normalize(normalized), normalized)


@pytest.mark.parametrize("target", [None, {"kind": "cuda", "arch": "sm_90"}, {"kind": "rocm", "mcpu": "gfx942"}, "metal", "llvm"])
def test_normalize_keeps_other_targets_unchanged(target):
    shared = _tensor("shared", [8])
    function = _function(
        [
            _alloc(shared),
            ir.TensorStore(shared, [0], ir.FloatImm("float32", 2)),
            ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared.dyn")])),
        ],
        target=target,
    )
    result = _normalize(function)
    assert result.same_as(function)
    assert ir.save_json(result) == ir.save_json(function)


def test_normalize_updates_pointer_parameters_and_function_type():
    pointer = ir.Var("pointer", ir.PtrType("int32", scope="shared.dyn"))
    view = _tensor("view", [4], "int32")
    function = _function(
        [
            ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
            ir.TensorStore(view, [0], ir.IntImm("int32", 9)),
        ],
        [pointer],
    )
    result = _normalize(function)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    assert result.params[0].ty.scope.name == "shared"
    assert result.body.seq[0].value.args[0].same_as(result.params[0])
    ir.assert_structural_equal(result.ty.arg_types[0], result.params[0].ty)


@pytest.mark.parametrize("explicit", [False, True])
def test_normalize_preserves_rejection_of_lexically_escaped_definition(explicit):
    private = _tensor("private", [4])
    function = _function(
        [
            _scope([_alloc(private)], explicit),
            ir.TensorStore(private, [0], ir.FloatImm("float32", 3)),
        ]
    )
    assert not tirx.analysis.check_well_formed(function, assert_mode=False)
    normalized = _normalize(function)
    assert not tirx.analysis.check_well_formed(normalized, assert_mode=False)
    lowered = _lower_tilelang_scopes(ir.IRModule({"main": normalized}))["main"]
    assert not tirx.analysis.check_well_formed(lowered, assert_mode=False)


@pytest.mark.parametrize("dtype,wgsl", [("float16", "f16"), ("float32", "f32"), ("int32", "i32"), ("uint32", "u32")])
def test_multiple_typed_workgroup_arrays_have_no_dynamic_launch_operand(dtype, wgsl, tmp_path):
    a, b = _tensor("first", [8], dtype), _tensor("second", [16], dtype)
    output = _tensor("output", [1], dtype, "global")
    thread, block = ir.Var("thread", "int32"), ir.Var("block", "int32")
    constant = ir.FloatImm(dtype, 2) if dtype.startswith("float") else ir.IntImm(dtype, 2)
    body = [
        _alloc(a),
        _alloc(b),
        ir.TensorStore(a, [0], constant),
        ir.TensorStore(b, [0], ir.TensorLoad(a, [0])),
        ir.TensorStore(output, [0], ir.TensorLoad(b, [0])),
    ]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.IntImm("int32", 8)], [thread], {}, body)
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), ir.IntImm("int32", 1)], [block], {}, [launch])
    target = tvm_tirx.target.Target("webgpu").with_host(tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"}))
    function = _function([launch], [output], target, device=False)
    module = ir.IRModule({"main": function})
    module = tirx.transform.AnnotateEntryFunc()(module)
    module = transform.AnnotateDeviceRegions()(module)
    module = transform.OutlineDeviceRegions()(module)
    module = _lower_workgroup_storage(module)
    module = transform.LowerDeviceKernelLaunch()(module)
    module = transform.MakePackedAPI()(module)
    assert tirx.analysis.check_well_formed(module)
    (tmp_path / "launch.json").write_text(ir.save_json(module))
    devices = [value for value in module.functions.values() if value.attrs.get("calling_conv") == 2]
    assert len(devices) == 1
    device = devices[0]
    assert list(device.attrs["tirx.kernel_launch_params"]) == ["blockIdx.x", "threadIdx.x"]
    assert len(device.params) == 1
    assert all(ty.scope.name != "shared.dyn" for ty in _objects(device, tirx.TensorType))
    (tmp_path / "launch.json").write_text(ir.save_json(module))
    source = _build(device, tmp_path, normalize=False)
    assert re.search(r"var<workgroup>\s+first\s*:\s*array<" + wgsl + r",\s*8>;", source)
    assert re.search(r"var<workgroup>\s+second\s*:\s*array<" + wgsl + r",\s*16>;", source)
    assert "@workgroup_size(8, 1, 1)" in source
    assert "dyn_shared_memory_buf" not in source


@pytest.mark.parametrize("explicit", [False, True])
def test_codegen_preserves_nested_lexical_scopes_captures_and_aliases(explicit, tmp_path):
    source_tensor = _tensor("source", [4], scope="global")
    output = _tensor("output", [4], scope="global")
    outer = _tensor("scratch", [1], scope="local")
    first = _tensor("scratch", [1], scope="local")
    second = _tensor("scratch", [1], scope="local")
    work = _tensor("work", [4])
    view = _tensor("work_view", [4])
    view2 = _tensor("work_view2", [4])
    capture = ir.TensorLoad(source_tensor, [0]) + ir.TensorLoad(outer, [0])
    function = _function(
        [
            _alloc(outer),
            ir.TensorStore(outer, [0], ir.FloatImm("float32", 2)),
            _scope(
                [
                    _alloc(first),
                    ir.TensorStore(first, [0], capture),
                    _scope(
                        [
                            _alloc(work),
                            ir.Bind(view, ir.Call("tirx.decl_tensor", [work.data], ty_args=[view.ty])),
                            ir.Bind(view2, ir.Call("tirx.decl_tensor", [view.data], ty_args=[view2.ty])),
                            ir.TensorStore(view2, [0], ir.TensorLoad(first, [0]) + ir.FloatImm("float32", 3)),
                            ir.TensorStore(output, [0], ir.TensorLoad(work, [0])),
                        ],
                        explicit,
                    ),
                ],
                explicit,
            ),
            _scope(
                [
                    _alloc(second),
                    ir.TensorStore(second, [0], capture),
                    ir.TensorStore(output, [1], ir.TensorLoad(second, [0]) + ir.TensorLoad(outer, [0])),
                ],
                explicit,
            ),
            ir.TensorStore(output, [2], ir.TensorLoad(outer, [0])),
        ],
        [source_tensor, output],
    )
    normalized = _normalize(function)
    regions = _objects(normalized, ir.RegionStmt)
    assert len(regions) == 3
    assert all(bool(region.attrs.get("explicit_alloc_scope", False)) == explicit for region in regions)
    before = ir.save_json(normalized)
    lowered = _lower_tilelang_scopes(ir.IRModule({"main": normalized}))["main"]
    assert ir.save_json(normalized) == before
    assert tirx.analysis.check_well_formed(lowered)
    assert not _objects(lowered, ir.RegionStmt)
    branches = _objects(lowered, ir.If)
    assert len(branches) == 3
    for region, branch in zip(regions, branches):
        assert bool(branch.condition.value)
        assert branch.else_case is None
        ir.assert_structural_equal(branch.loc, region.loc)
    # Lower only the region wrapper. Allocation definitions and all uses keep
    # their exact identities, including same-named local arrays and aliases.
    for old, new in zip(_objects(normalized, ir.Bind), _objects(lowered, ir.Bind)):
        assert new.var.same_as(old.var)
        assert new.value.same_as(old.value)
    for old, new in zip(_objects(normalized, ir.TensorStore), _objects(lowered, ir.TensorStore)):
        assert new.same_as(old)
    assert _lower_tilelang_scopes(ir.IRModule({"main": lowered}))["main"].same_as(lowered)
    source = _build(normalized, tmp_path, normalize=False)
    assert "var<storage, read> source_ptr" in source
    assert "var<storage, read_write> output_ptr" in source
    assert source.count("var<workgroup>") == 1
    locals_with_depth = []
    depth = 0
    for line in source.splitlines():
        depth -= line.count("}")
        match = re.search(r"\bvar (scratch\w*)\s*:\s*array<f32, 1>;", line)
        if match:
            locals_with_depth.append((match.group(1), depth))
        depth += line.count("{")
    assert depth == 0
    assert [depth for _, depth in locals_with_depth] == [1, 2, 2]
    assert len({name for name, _ in locals_with_depth}) == 3
    assert "work_view" not in source
    assert re.search(r"output_ptr\[2[ui]?\] = scratch\[0[ui]?\]", source)


def test_collector_sees_launch_dimensions_inside_lexical_scope(tmp_path):
    thread = ir.Var("thread", "int32")
    scratch = _tensor("scratch", [4])
    launch = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.IntImm("int32", 4)],
        [thread],
        {},
        [_alloc(scratch), ir.TensorStore(scratch, [thread], ir.FloatImm("float32", 1))],
    )
    source = _build(_function([_scope([launch])]), tmp_path)
    assert "@workgroup_size(4, 1, 1)" in source


@pytest.mark.parametrize("operation,args", [("tirx.compute_scope", ["compute"]), ("tirx.parallel_launch", [])])
def test_unknown_webgpu_regions_still_rejected(operation, args, tmp_path):
    args = [ir.StringImm(value) for value in args]
    region = ir.RegionStmt(operation, args, [], {}, [ir.Evaluate(ir.IntImm("int32", 0))])
    function = _function([_scope([region])])
    with pytest.raises(ValueError, match="Unsupported region op"):
        _build(function, tmp_path)


@pytest.mark.parametrize("sizes,limit,used", [([8], 32, 32), ([8, 1], 32, 48), ([1, 1], 16, 32), ([1, 1], 32, 32)])
def test_workgroup_limit_counts_all_lexical_declarations_and_alignment(sizes, limit, used, tmp_path):
    body = []
    for index, size in enumerate(sizes):
        array = _tensor("scratch" + str(index), [size])
        body.append(_scope([_alloc(array), ir.TensorStore(array, [0], ir.FloatImm("float32", 1))]))
    target = {"kind": "webgpu", "max_shared_memory_per_block": limit}
    function = _function(body, target=target)
    if used <= limit:
        source = _build(function, tmp_path)
        assert source.count("var<workgroup>") == len(sizes)
    else:
        with pytest.raises(RuntimeError, match=rf"WebGPU workgroup allocations use {used} bytes, .* supports only {limit} bytes"):
            _build(function, tmp_path)


@pytest.mark.parametrize("bound,limit", [(32, 128), (32, 112), (2147483647, 128)])
def test_bounded_symbolic_workgroup_size_is_validated_after_normalization(bound, limit, tmp_path):
    n = ir.Var("n", "int32")
    extent = ir.Var("extent", "int32")
    array = _tensor("scratch", [extent])
    target = {"kind": "webgpu", "max_shared_memory_per_block": limit}
    function = _function(
        [
            _scope(
                [
                    ir.Bind(extent, tirx.min(n, ir.IntImm("int32", bound))),
                    _alloc(array),
                    ir.TensorStore(array, [0], ir.FloatImm("float32", 1)),
                ]
            )
        ],
        [n],
        target,
    )
    if bound == 32 and limit == 128:
        assert "array<f32, 32>" in _build(function, tmp_path)
    else:
        message = "finite compile-time upper bound" if bound != 32 else "workgroup allocations use 128 bytes"
        with pytest.raises(RuntimeError, match=message):
            _build(function, tmp_path)


@pytest.mark.parametrize("extent", [0, -1, "unbounded"])
def test_invalid_workgroup_extents_still_rejected(extent, tmp_path):
    n = ir.Var("n", "int32")
    shape = [n] if extent == "unbounded" else [extent]
    array = _tensor("scratch", shape)
    function = _function([_scope([_alloc(array), ir.Evaluate(array.data)])], [n] if extent == "unbounded" else [])
    message = "finite compile-time upper bound" if extent == "unbounded" else "positive compile-time upper bound"
    with pytest.raises(RuntimeError, match=message):
        _build(function, tmp_path)


def test_workgroup_element_count_overflow_still_rejected(tmp_path):
    array = _tensor("scratch", [1 << 30, 1 << 30, 1 << 30], "uint8")
    function = _function([_scope([_alloc(array), ir.Evaluate(array.data)])])
    with pytest.raises(RuntimeError, match="element count is too large to represent"):
        _build(function, tmp_path)


@pytest.mark.parametrize("scope", ["global", "local.fragment"])
def test_unsupported_allocation_scopes_are_not_normalized(scope, tmp_path):
    array = _tensor("scratch", [4], scope=scope)
    function = _function([_scope([_alloc(array), ir.Evaluate(array.data)])])
    normalized = _normalize(function)
    assert _objects(normalized, ir.Bind)[0].var.ty.scope.name == scope
    with pytest.raises(RuntimeError, match="Do not support storage scope"):
        _build(normalized, tmp_path, normalize=False)


def test_full_workgroup_sync_normalization_preserves_call_metadata(tmp_path):
    location = ir.SourceLoc(ir.SourceName("workgroup-sync"), 2, 1, 2, 40)
    scope = ir.StringImm("shared.dyn", loc=location)
    call = ir.Call("tirx.gpu_storage_sync", [scope], attrs={"diagnostic_tag": "producer"}, loc=location, ty="void")
    function = _function([_scope([ir.Evaluate(call)])])
    result = _normalize(function)
    actual = _objects(result, ir.Call)[0]
    assert actual.op.same_as(call.op)
    assert actual.args[0].value == "shared"
    assert call.args[0].value == "shared.dyn"
    ir.assert_structural_equal(actual.attrs, call.attrs)
    ir.assert_structural_equal(actual.ty_args, call.ty_args)
    ir.assert_structural_equal(actual.ty, call.ty)
    ir.assert_structural_equal(actual.loc, call.loc)
    ir.assert_structural_equal(actual.args[0].loc, scope.loc)
    assert _build(result, tmp_path, normalize=False).count("workgroupBarrier();") == 1


@pytest.mark.parametrize("scope", ["warp", "global"])
def test_non_workgroup_sync_is_not_rewritten(scope, tmp_path):
    call = ir.Call("tirx.gpu_storage_sync", [ir.StringImm(scope)])
    function = _function([ir.Evaluate(call)])
    result = _normalize(function)
    assert result.same_as(function)
    actual = _objects(result, ir.Call)[0]
    assert actual.same_as(call)
    assert actual.args[0].value == scope
    if scope == "global":
        with pytest.raises(RuntimeError, match="global barrier not supported"):
            _build(result, tmp_path, normalize=False)


@pytest.mark.parametrize("scope", ["local", "shared.custom", "texture"])
def test_unsupported_sync_scope_is_rejected_before_codegen_can_drop_it(scope, tmp_path):
    call = ir.Call("tirx.gpu_storage_sync", [ir.StringImm(scope)])
    function = _function([ir.Evaluate(call)])
    with pytest.raises(ValueError, match="Unsupported WebGPU storage synchronization scope"):
        _build(function, tmp_path)


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("count", [None, 64])
def test_named_and_partial_barriers_keep_their_contract_until_tilelang_codegen(scope, count, tmp_path):
    args = [ir.StringImm(scope), ir.IntImm("int32", 1)]
    if count is not None:
        args.append(ir.IntImm("int32", count))
    partial = ir.Call("tl.storage_sync", args)
    thread = ir.Var("thread", "int32")
    launch = ir.RegionStmt(
        "tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.IntImm("int32", 128)], [thread], {}, [ir.Evaluate(partial)]
    )
    result = _normalize(_function([launch]))
    actual = _objects(result, ir.Call)[0]
    assert actual.same_as(partial)
    assert actual.op.name == "tl.storage_sync"
    ir.assert_structural_equal(actual.args, args)
    with pytest.raises((ValueError, RuntimeError), match="tl.storage_sync"):
        tvm_ffi.get_global_func("target.build.webgpu")(ir.IRModule({"main": result}), result.attrs["target"])
    # The raw target emitter remains unaware of TileLang operations. Only the
    # final TileLang wrapper selects its established whole-workgroup fallback.
    from tilelang.webgpu.codegen import build_webgpu

    source = build_webgpu(ir.IRModule({"main": result}), result.attrs["target"]).inspect_source("wgsl")
    assert source.count("workgroupBarrier();") == 1
    assert _objects(result, ir.Call)[0].same_as(partial)


def test_provisional_partial_native_sync_keeps_extra_arguments_and_fails_validation():
    args = [ir.StringImm("shared.dyn"), ir.IntImm("int32", 1), ir.IntImm("int32", 64)]
    partial = ir.Call("tirx.gpu_storage_sync", args, ty="void")
    result = _normalize(_function([ir.Evaluate(partial)]))
    actual = _objects(result, ir.Call)[0]
    assert len(actual.args) == 3
    assert actual.args[0].value == "shared"
    ir.assert_structural_equal(actual.args[1:], partial.args[1:])
    with pytest.raises((ValueError, TypeError), match="arg"):
        actual.validate()


def test_unrelated_scope_string_is_not_rewritten():
    value = ir.StringImm("shared.dyn")
    function = _function([ir.Evaluate(value)])
    result = _normalize(function)
    assert result.same_as(function)
    assert result.body.seq[0].value.same_as(value)


def test_original_gemm_emits_producer_and_reuse_barriers_inside_reduction_loop(tmp_path):
    path = Path(__file__).with_name("test_webgpu_codegen.py")
    spec = importlib.util.spec_from_file_location("original_webgpu_gemm_for_sync", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    function = original.matmul(1024, 1024, 1024, 16, 16, 16)
    before = ir.save_json(function)
    with tilelang.transform.PassContext(), tvm_tirx.target.Target("webgpu"):
        artifact = tilelang.lower(function, target="webgpu")
    assert ir.save_json(function) == before
    source = artifact.kernel_source
    (tmp_path / "original-function.json").write_text(before)
    (tmp_path / "device.json").write_text(ir.save_json(artifact.device_mod))
    (tmp_path / "kernel.wgsl").write_text(source)
    barriers = [match.start() for match in re.finditer(r"(?m)^ +workgroupBarrier\(\);$", source)]
    assert len(barriers) == source.count("workgroupBarrier();") == 2
    syncs = [
        call
        for device in artifact.device_mod.functions.values()
        for call in _objects(device.body, ir.Call)
        if isinstance(call.op, ir.Op) and call.op.name == "tirx.gpu_storage_sync"
    ]
    assert [call.args[0].value for call in syncs] == ["shared", "shared"]
    loop = source.index("for (var ko :")
    first_write = source.index("A_shared[", loop)
    second_write = source.index("B_shared[", first_write)
    accumulation = source.index("C_local[", second_write)
    assert loop < barriers[0] < first_write < second_write < barriers[1] < accumulation
    assert "var<workgroup> A_shared : array<f16, 256>;" in source
    assert "var<workgroup> B_shared : array<f16, 256>;" in source
