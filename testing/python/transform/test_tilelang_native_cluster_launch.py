"""Cluster ownership and final CUDA launch operands use native IR contracts."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx
from tvm_tirx.target import Target

import tilelang


def _nodes(value, cls, name=None):
    found = []
    tvm_ffi.structural_walk(value, (cls, lambda node: found.append(node) if name is None or getattr(node.op, "name", "") == name else None))
    return found


def _thread(tag, extent, body, variable=None):
    if isinstance(extent, int):
        extent = ir.const(extent, "int64")
    variable = variable if variable is not None else ir.Var(tag.replace(".", "_"), extent.ty)
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), extent], [variable], {}, body)


def _metadata(dimensions, body=None, two_cta=None):
    values = {"cluster_dims": dimensions, "pragma_cluster_test": 7}
    if two_cta is not None:
        values["use_2cta"] = two_cta
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": "tl.kernel", "value": values})
    return ir.RegionStmt(
        "tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body if body is not None else [ir.Evaluate(0)]
    )


def _source(clusters=((2, 1, 1), (1, 2, 1)), two_cta=None, extra=None):
    n = ir.Var("n", "int64")
    launches = []
    for index, dimensions in enumerate(clusters):
        kernel = _metadata(dimensions, extra, two_cta)
        launches.append(_thread("blockIdx.x", n + index, [_thread("threadIdx.x", 32 * (index + 1), [kernel])]))
    function = tirx.Function([n], launches, ret_type=ir.PrimType("void"))
    function = function.with_attr("global_symbol", "main").with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}, host="llvm"))
    return tvm_tirx.IRModule({"main": function})


def _owners(module):
    module = tilelang.transform.LowerTileOp()(module)
    return tilelang.cuda.transform.LowerSharedTmem()(module)


def _cluster(module):
    return tilelang.cuda.transform.LowerClusterLaunch()(module)


def _launch(module):
    return tilelang.transform.LowerDeviceKernelLaunch()(module)


def _eval(expression, variable, value):
    expression = tvm_ffi.structural_map(
        expression, (ir.Var, lambda item: ir.const(value, variable.ty) if item.same_as(variable) else item), order="pre"
    )
    return int(sym.Analyzer().simplify(expression))


def _fields(call):
    return dict(zip(call.attrs.launch_fields, call.args[1 + int(call.attrs.num_kernel_args) :]))


def test_cluster_metadata_becomes_per_kernel_native_operands(tmp_path):
    module = _owners(_source())
    before = ir.save_json(module)
    result = _cluster(module)
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(result)
    scopes = _nodes(result, ir.RegionStmt, "tirx.device_scope")
    assert len(scopes) == 2
    for scope, expected in zip(scopes, ((2, 1, 1), (1, 2, 1))):
        assert list(scope.attrs["cuda.launch_fields"]) == ["cluster.x", "cluster.y", "cluster.z"]
        assert tuple(map(int, scope.args)) == expected
        assert scope.attrs["tl.derive_launch_geometry"]
    for region in _nodes(result, ir.RegionStmt, "tl.attr"):
        decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
        assert "cluster_dims" not in decoded["value"]
        assert decoded["value"]["pragma_cluster_test"] == 7
    assert all("clusterCtaIdx" not in region.args[0].value for region in _nodes(result, ir.RegionStmt, "tirx.launch_thread"))
    ir.assert_structural_equal(_cluster(result), result)
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)
    (tmp_path / "cluster.json").write_text(ir.save_json(result))


def test_cluster_consumption_requires_successful_owners():
    source = _source(((2, 1, 1),), two_cta=1)
    before = ir.save_json(source)
    with pytest.raises(ValueError, match="after LowerTileOp and LowerSharedTmem"):
        _cluster(source)
    after_tile = tilelang.transform.LowerTileOp()(source)
    with pytest.raises(ValueError, match="after LowerTileOp and LowerSharedTmem"):
        _cluster(after_tile)
    assert ir.save_json(source) == before
    # No TMEM allocation is required merely to declare a two-CTA cluster.
    finished = _cluster(_owners(source))
    assert len(_nodes(finished, ir.RegionStmt, "tirx.device_scope")) == 1
    assert not _nodes(finished, ir.Call, "tirx.alloc_tensor")


def test_two_cta_tmem_instructions_survive_cluster_hint_consumption():
    scratch = ir.Var("physical_tmem", tirx.TensorType((128, 32), "float32", scope="shared.tmem", layout="default"))
    source = _source(
        ((2, 1, 1),),
        two_cta=1,
        extra=[ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty])), ir.Evaluate(ir.TensorLoad(scratch, [0, 0]))],
    )
    source = _owners(source)
    result = _cluster(source)
    for name in ("tl.ptx_init_tensor_memory", "tl.ptx_deallocate_tensor_memory"):
        (instruction,) = _nodes(result, ir.Call, name)
        assert int(instruction.attrs["use_2cta"]) == 1
    (scope,) = _nodes(result, ir.RegionStmt, "tirx.device_scope")
    assert tuple(map(int, scope.args)) == (2, 1, 1)


def test_new_tmem_after_owner_marker_is_rejected():
    source = _owners(_source(((2, 1, 1),), two_cta=1))
    scratch = ir.Var("unlowered_tmem", tirx.TensorType((128, 32), "float32", scope="shared.tmem", layout="default"))
    allocation = ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty]))

    def insert(region):
        if region.op.name == "tl.attr":
            return ir.RegionStmt(
                region.op, region.args, region.body_params, region.attrs, [allocation, *region.body.seq], region.result_vars, region.loc
            )
        return region

    changed = tvm_ffi.structural_map(source, (ir.RegionStmt, insert), order="pre")
    assert len(_nodes(changed, ir.Call, "tirx.alloc_tensor")) == 1
    with pytest.raises(ValueError, match="consume all shared.tmem"):
        _cluster(changed)


@pytest.mark.parametrize(
    "dimensions,message",
    [
        ((2, 1), "three dimensions"),
        ((0, 1, 1), "positive uint32"),
        ((2, -1, 1), "positive uint32"),
        ((2**32, 1, 1), "positive uint32"),
    ],
)
def test_invalid_cluster_dimensions_are_rejected(dimensions, message):
    source = _owners(_source(((2, 1, 1),)))
    # Feed malformed late IR directly to this consumer. Earlier owners already
    # reject some invalid dimensions before this pass can observe them.
    changed = tvm_ffi.structural_map(
        source, (ir.RegionStmt, lambda region: _metadata(dimensions, region.body) if region.op.name == "tl.attr" else region), order="pre"
    )
    with pytest.raises(ValueError, match=message):
        _cluster(changed)


def test_conflicting_clusters_in_one_physical_kernel_are_rejected():
    source = _source(((2, 1, 1),), extra=[_metadata((1, 2, 1))])
    with pytest.raises(ValueError, match="Different cluster_dims"):
        _cluster(_owners(source))


def test_two_cta_requirement_is_not_downgraded():
    with pytest.raises(ValueError, match=r"use_2cta requires cluster_dims \(2,1,1\) or \(1,2,1\)"):
        _cluster(_owners(_source(((1, 1, 1),), two_cta=1)))


@pytest.mark.parametrize("conflict", [False, True])
def test_existing_native_launch_configuration_is_preserved_or_rejected(conflict):
    module = _owners(_source(((2, 1, 1),)))
    function = module["main"]
    scope = ir.RegionStmt(
        "tirx.device_scope",
        [ir.const(5), ir.const(128), ir.const(4 if conflict else 2)],
        [],
        {"cuda.launch_fields": ["grid.x", "block.x", "cluster.x"], "cuda.kernel_attrs": {"required_block_size": 1}},
        function.body,
    )
    wrapped = tirx.Function(function.params, [scope], function.ret_type, function.attrs)
    wrapped = tvm_tirx.IRModule({"main": wrapped})
    if conflict:
        with pytest.raises(ValueError, match="disagrees with cluster_dims"):
            _cluster(wrapped)
        return
    result = _cluster(wrapped)
    (scope,) = _nodes(result, ir.RegionStmt, "tirx.device_scope")
    assert dict(zip(scope.attrs["cuda.launch_fields"], map(int, scope.args))) == {
        "grid.x": 5,
        "block.x": 128,
        "cluster.x": 2,
        "cluster.y": 1,
        "cluster.z": 1,
    }
    assert scope.attrs["cuda.kernel_attrs"]["required_block_size"] == 1
    assert "tl.derive_launch_geometry" not in scope.attrs


def test_dynamic_cluster_metadata_is_not_hoisted_as_a_host_expression():
    source = _owners(_source(((2, 1, 1),)))
    n = source["main"].params[0]
    source = tvm_ffi.structural_map(
        source, (ir.RegionStmt, lambda region: _metadata((n, 1, 1), region.body) if region.op.name == "tl.attr" else region), order="pre"
    )
    with pytest.raises(ValueError, match="constant integers"):
        _cluster(source)


def test_geometry_is_collected_after_outlining_and_late_thread_rewrite(tmp_path):
    result = _cluster(_owners(_source()))
    result = tilelang.transform.LowerSourceAnnotations()(result)
    result = tilelang.transform.OutlineDeviceRegions()(result)

    # Model an intervening device preparation pass changing physical threads.
    # Launch parameters are taken from these actual final native regions.
    def update(region):
        if region.op.name == "tirx.launch_thread" and region.args[0].value == "threadIdx.x":
            return ir.RegionStmt(
                region.op,
                [region.args[0], region.args[1] * 4],
                region.body_params,
                region.attrs,
                region.body,
                region.result_vars,
                region.loc,
            )
        return region

    result = tvm_ffi.structural_map(result, (ir.RegionStmt, update), order="pre")
    result = _launch(result)
    assert tirx.analysis.check_well_formed(result)
    host = result["main"]
    assert not tirx.analysis.undefined_vars(host)
    calls = _nodes(host, ir.Call, "tirx.call_ffi_kernel")
    assert len(calls) == 2
    for index, (call, cluster) in enumerate(zip(calls, ((2, 1, 1), (1, 2, 1)))):
        fields = _fields(call)
        assert int(fields["block.x"]) == 128 * (index + 1)
        assert tuple(int(fields["cluster." + axis]) for axis in "xyz") == cluster
        for n in (2, 8, 18):
            assert _eval(fields["grid.x"], host.params[0], n) == n + index
        device = next(
            function for function in result.functions.values() if (function.attrs or {}).get("global_symbol") == call.args[0].value
        )
        assert list(map(int, device.attrs["cuda.launch_dimensions"])) == [128 * (index + 1), 1, 1, *cluster]
        assert not list(call.attrs.launch_params)
        assert "tirx.kernel_launch_params" not in device.attrs
    ir.assert_structural_equal(_launch(result), result)
    (tmp_path / "final.json").write_text(ir.save_json(result))


def _configured(*, dtype="float32", flags=True, bad_extent=None, supplied_flags=None):
    n, count = ir.Var("device_n", "int64"), ir.Var("count", "int64")
    output = ir.Var("device_output", ir.PtrType("int32", scope="global"))
    scratch = ir.Var("scratch", tirx.TensorType((count,), dtype, scope="shared.dyn", layout="default"))
    block = ir.Var("block", "int64")
    extent = block + 1 if bad_extent == "local" else n if bad_extent == "dynamic" else ir.const(128, "int64")
    if bad_extent == "memory":
        data = ir.Var("data", tirx.TensorType((1,), "int32", scope="global", layout="default"))
        extent = ir.TensorLoad(data, [ir.const(0, "int64")])
        prefix = [ir.Bind(data, ir.Call("tirx.decl_tensor", [output], ty_args=[data.ty]))]
    else:
        prefix = []
    body = [ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty]))]
    device = tirx.Function(
        [output, n],
        [*prefix, ir.Bind(count, n + 2), _thread("blockIdx.x", n, [_thread("threadIdx.x", extent, body)], block)],
        ret_type=ir.PrimType("void"),
    ).with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}))
    if flags:
        device = device.with_attr("has_cuda_pdl_trigger", 1).with_attr("has_cuda_pdl_sync", 1).with_attr("use_cooperative_groups", 1)
    symbol = ir.GlobalVar("device")
    host_n = ir.Var("host_n", "int64")
    host_output = ir.Var("host_output", output.ty)
    stream = ir.Var("stream", ir.PtrType("void"))
    fields = ["cluster.x", "cluster.y", "cluster.z", "stream"]
    values = [ir.const(2), ir.const(1), ir.const(1), stream]
    for key, value in (supplied_flags or {}).items():
        fields.append(key)
        values.append(ir.const(value))
    pending = ir.RegionStmt(
        "tl.pending_cuda_launch",
        values,
        [],
        {"cuda.launch_fields": fields, "cuda.kernel_attrs": {}, "tl.derive_launch_geometry": True},
        [ir.Evaluate(ir.Call(symbol, [host_output, host_n + 1], ty="void"))],
    )
    host = tirx.Function([host_output, host_n, stream], [pending])
    host = host.with_attr("target", Target("llvm")).with_attr("global_symbol", "main")
    return tvm_tirx.IRModule({symbol: device, "main": host}), symbol


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("int4", 4), ("int4x4", 16)])
def test_native_launch_uses_final_storage_flags_and_host_stream(dtype, bits):
    source, symbol = _configured(dtype=dtype)
    before = ir.save_json(source)
    result = _launch(source)
    assert ir.save_json(source) == before
    assert tirx.analysis.check_well_formed(result)
    host, device = result["main"], result[symbol]
    assert not tirx.analysis.undefined_vars(host)
    (call,) = _nodes(host, ir.Call, "tirx.call_ffi_kernel")
    values = _fields(call)
    assert values["stream"].same_as(host.params[2])
    assert int(values["cooperative"]) == int(values["programmatic_stream_serialization"]) == 1
    assert int(values["block.x"]) == 128
    for n in (1, 4, 17, 256):
        assert _eval(values["grid.x"], host.params[1], n) == n + 1
        assert _eval(values["dynamic_smem_bytes"], host.params[1], n) == ((n + 3) * bits + 7) // 8
    assert list(device.attrs["cuda.launch_fields"]) == list(call.attrs.launch_fields)
    assert list(map(int, device.attrs["cuda.launch_dimensions"])) == [128, 1, 1, 2, 1, 1]
    assert "tirx.kernel_launch_params" not in device.attrs
    assert not list(call.attrs.launch_params)
    ir.assert_structural_equal(_launch(result), result)


@pytest.mark.parametrize(
    "case,message", [("local", "device-local variable"), ("memory", "device-local variable|cannot read device memory")]
)
def test_native_launch_rejects_device_dependent_geometry(case, message):
    source, _ = _configured(bad_extent=case)
    with pytest.raises(ValueError, match=message):
        _launch(source)


@pytest.mark.parametrize("field", ["cooperative", "programmatic_stream_serialization"])
def test_native_launch_cannot_disable_required_flags(field):
    source, _ = _configured(supplied_flags={field: 0})
    with pytest.raises(ValueError, match="requires CUDA launch field"):
        _launch(source)


def test_dynamic_block_dimension_stays_in_host_operands():
    source, symbol = _configured(bad_extent="dynamic", flags=False)
    result = _launch(source)
    assert tirx.analysis.check_well_formed(result)
    (call,) = _nodes(result["main"], ir.Call, "tirx.call_ffi_kernel")
    values = _fields(call)
    for n in (31, 127, 255):
        assert _eval(values["block.x"], result["main"].params[1], n) == n + 1
    # Zero is the native "dynamic" sentinel, never a fixed one-thread block.
    assert list(map(int, result[symbol].attrs["cuda.launch_dimensions"])) == [0, 1, 1, 2, 1, 1]


@pytest.mark.parametrize("backend", ["metal", "rocm", "ascend"])
def test_non_cuda_ordinary_launch_keeps_backend_metadata(backend):
    extent = ir.Var("device_extent", "int64")
    device = tirx.Function(
        [extent], [_thread("blockIdx.x", extent, [_thread("threadIdx.x", 32, [ir.Evaluate(0)])])], ret_type=ir.PrimType("void")
    )
    device = device.with_attr("target", Target(backend)).with_attr("global_symbol", "ordinary_device")
    symbol = ir.GlobalVar("ordinary_device")
    host_extent = ir.Var("host_extent", "int64")
    host = tirx.Function([host_extent], [ir.Evaluate(ir.Call(symbol, [host_extent + 1], ty="void"))])
    host = host.with_attr("target", Target("llvm")).with_attr("global_symbol", "main")
    module = tvm_tirx.IRModule({symbol: device, "main": host})
    before = ir.save_json(module)
    result = _launch(module)
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(result)
    (call,) = _nodes(result["main"], ir.Call, "tirx.call_ffi_kernel")
    assert not list(call.attrs.launch_fields)
    assert not list(call.attrs.kernel_attrs)
    assert "blockIdx.x" in call.attrs.launch_params
    assert not _nodes(result, ir.RegionStmt, "tl.pending_cuda_launch")
    assert all(not str(key).startswith("cuda.") for key in result[symbol].attrs.dict)
    for value in (0, 3, 17):
        assert _eval(call.args[2], result["main"].params[0], value) == value + 1
    ir.assert_structural_equal(_launch(result), result)


@pytest.mark.parametrize("backend", ["metal", "rocm", "ascend"])
def test_non_cuda_explicit_launch_configuration_is_rejected(backend):
    module, symbol = _configured(flags=False)
    module[symbol] = module[symbol].with_attr("target", Target(backend))
    before = ir.save_json(module)
    with pytest.raises(ValueError, match="Explicit native CUDA launch configuration requires a CUDA device kernel"):
        _launch(module)
    assert ir.save_json(module) == before


def _annotation_statement(values):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}, ty="void"))


def _as_annotation_statements(module, values=None):
    def replace(region):
        if region.op.name != "tl.attr":
            return region
        fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
        if fields["key"] != "tl.kernel":
            return region
        metadata = fields["value"] if values is None else values
        return ir.RegionStmt("tl.lexical_scope", [], [], {}, [_annotation_statement(metadata), *region.body.seq])

    return tvm_ffi.structural_map(module, (ir.RegionStmt, replace), order="post")


def test_annotation_statement_clusters_reach_each_native_launch():
    source = _as_annotation_statements(_owners(_source()))
    before = ir.save_json(source)
    result = _cluster(source)
    assert ir.save_json(source) == before
    assert tirx.analysis.check_well_formed(result)
    scopes = _nodes(result, ir.RegionStmt, "tirx.device_scope")
    assert len(scopes) == 2
    for scope, expected in zip(scopes, ((2, 1, 1), (1, 2, 1))):
        assert tuple(map(int, scope.args)) == expected
    for call in _nodes(result, ir.Call, "tl.annotation"):
        values = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
        assert "cluster_dims" not in values
        assert values["pragma_cluster_test"] == 7
    ir.assert_structural_equal(_cluster(result), result)
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)
    result = _launch(tilelang.transform.OutlineDeviceRegions()(tilelang.transform.LowerSourceAnnotations()(result)))
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result["main"])
    calls = _nodes(result["main"], ir.Call, "tirx.call_ffi_kernel")
    assert len(calls) == 2
    for index, (call, expected) in enumerate(zip(calls, ((2, 1, 1), (1, 2, 1)))):
        fields = _fields(call)
        assert tuple(int(fields["cluster." + axis]) for axis in "xyz") == expected
        assert int(fields["block.x"]) == 32 * (index + 1)
        for n in (2, 8, 18):
            assert _eval(fields["grid.x"], result["main"].params[0], n) == n + index


def test_annotation_statement_keeps_dynamic_operands_and_lexical_scope():
    source = _owners(_source(((2, 1, 1),)))
    local = ir.Var("local_hint", "int64")
    n = source["main"].params[0]

    def replace(region):
        if region.op.name != "tl.attr":
            return region
        body = [
            ir.Bind(local, n + 7),
            _annotation_statement({"cluster_dims": [2, 1, 1], "pragma_cluster_test": local}),
            ir.Evaluate(local),
        ]
        return ir.RegionStmt("tl.lexical_scope", [], [], {"explicit_alloc_scope": True}, body)

    source = tvm_ffi.structural_map(source, (ir.RegionStmt, replace), order="post")
    before = ir.save_json(source)
    result = _cluster(source)
    assert ir.save_json(source) == before
    (scope,) = _nodes(result, ir.RegionStmt, "tl.lexical_scope")
    assert scope.attrs["explicit_alloc_scope"]
    assert scope.body.seq[0].var.same_as(local)
    (call,) = _nodes(result, ir.Call, "tl.annotation")
    values = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
    assert set(values) == {"pragma_cluster_test"}
    assert values["pragma_cluster_test"].same_as(local)
    assert scope.body.seq[2].value.same_as(local)
    result = tilelang.transform.LowerSourceAnnotations()(result)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result["main"])


@pytest.mark.parametrize("conflict", [False, True])
def test_mixed_cluster_metadata_forms_agree_or_reject_conflicts(conflict):
    dimensions = (1, 2, 1) if conflict else (2, 1, 1)
    source = _owners(_source(((2, 1, 1),), extra=[_annotation_statement({"cluster_dims": dimensions})]))
    if conflict:
        with pytest.raises(ValueError, match="Different cluster_dims"):
            _cluster(source)
        return
    result = _cluster(source)
    (scope,) = _nodes(result, ir.RegionStmt, "tirx.device_scope")
    assert tuple(map(int, scope.args)) == (2, 1, 1)
    assert tirx.analysis.check_well_formed(tilelang.transform.LowerSourceAnnotations()(result))


@pytest.mark.parametrize("inside_kernel", [False, True])
def test_annotation_statement_cluster_requires_kernel_and_completed_owners(inside_kernel):
    source = _as_annotation_statements(_source(((2, 1, 1),)))
    if inside_kernel:
        message = "after LowerTileOp and LowerSharedTmem"
    else:
        function = source["main"]
        source = tvm_tirx.IRModule(
            {"main": tirx.Function(function.params, [_annotation_statement({"cluster_dims": [2, 1, 1]})], attrs=function.attrs)}
        )
        message = "enclosing physical kernel launch"
    with pytest.raises(ValueError, match=message):
        _cluster(source)


@pytest.mark.parametrize(
    "metadata,message",
    [
        ({"cluster_dims": [2, 1]}, "three dimensions"),
        ({"cluster_dims": [0, 1, 1]}, "positive uint32"),
        ({"cluster_dims": [2**32, 1, 1]}, "positive uint32"),
        ({"use_2cta": True}, "requires cluster_dims"),
        ({"cluster_dims": [2, 1, 1], "use_2cta": 2}, "boolean flag"),
        ({"cluster_dims": [1, 1, 1], "use_2cta": True}, r"requires cluster_dims \(2,1,1\) or \(1,2,1\)"),
    ],
)
def test_annotation_statement_cluster_validation(metadata, message):
    # Inject malformed late metadata after the owners, matching the consumer's
    # existing region-form tests without relying on earlier rejection.
    source = _as_annotation_statements(_owners(_source(((2, 1, 1),))), metadata)
    with pytest.raises(ValueError, match=message):
        _cluster(source)


def test_annotation_statement_keeps_unknown_metadata_for_its_owner():
    # A compiler-owned neighbor still requires its consumer. Generic external
    # user hints are intentionally ignored once all owning passes have run.
    source = _as_annotation_statements(_owners(_source(((2, 1, 1),))), {"cluster_dims": [2, 1, 1], "tl.unconsumed_cluster_neighbor": 7})
    result = _cluster(source)
    result = tilelang.transform.LowerSourceAnnotations()(result)
    with pytest.raises(ValueError, match="Unconsumed TileLang call `tl.annotation`"):
        tilelang.transform.VerifyLoweredAnnotations()(result)
