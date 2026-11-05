"""Final device resources and host arguments share the native launch contract."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx
from tvm_tirx.target import Target

import tilelang


def _calls(value, name):
    found = []
    tvm_ffi.structural_walk(value, (ir.Call, lambda call: found.append(call) if getattr(call.op, "name", "") == name else None))
    return found


def _thread(tag, extent, body, variable=None):
    variable = variable if variable is not None else ir.Var(tag.replace(".", "_"), "int64")
    if isinstance(extent, int):
        extent = ir.const(extent, "int64")
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), extent], [variable], {}, body)


def _module(*, dtype="float32", flags=False, bad_extent=None, extra_allocation=False):
    extent = ir.Var("device_extent", "int64")
    count = ir.Var("local_count", "int64")
    output = ir.Var("device_output", ir.PtrType("int32", scope="global"))
    shared = ir.Var("scratch", tirx.TensorType((count,), dtype, scope="shared.dyn", layout="default"))
    body = [ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty]))]
    if extra_allocation:
        other = ir.Var("scratch2", shared.ty)
        body.append(ir.Bind(other, ir.Call("tirx.alloc_tensor", [], ty_args=[other.ty])))
    if flags:
        body.append(ir.Evaluate(ir.Call("tl.sync_grid", [])))
    block = ir.Var("block", "int64")
    thread_extent = block + 1 if bad_extent == "local" else ir.const(32, "int64")
    body = [_thread("threadIdx.x", thread_extent, body)]
    if bad_extent == "conflict":
        body.append(_thread("threadIdx.x", 64, [ir.Evaluate(0)]))
    device = (
        tirx.Function(
            [output, extent],
            [ir.Bind(count, extent + 2), _thread("blockIdx.x", extent, body, block)],
            ret_type=ir.PrimType("void"),
        )
        .with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}))
        .with_attr("global_symbol", "real_device")
    )
    if flags:
        device = device.with_attr("has_cuda_pdl_trigger", 1)
    symbol = ir.GlobalVar("device_hint")
    host_extent = ir.Var("host_extent", "int64")
    host_output = ir.Var("host_output", output.ty)
    host = (
        tirx.Function(
            [host_output, host_extent],
            [ir.Evaluate(ir.Call(symbol, [host_output, host_extent + 1], ty="void"))],
        )
        .with_attr("target", Target("llvm"))
        .with_attr("global_symbol", "main")
    )
    return tvm_tirx.IRModule({symbol: device, "main": host}), symbol


def _lower(module):
    return tilelang.transform.LowerDeviceKernelLaunch()(module)


def _eval(expression, variable, value):
    substituted = tvm_ffi.structural_map(
        expression,
        (ir.Var, lambda item: ir.const(value, variable.ty) if item.same_as(variable) else item),
        order="pre",
    )
    return int(sym.Analyzer().simplify(substituted))


def _configured_device_scope(dtype):
    data = ir.Var("data", tirx.TensorType([8], "float32"))
    n = ir.Var("n", "int64")
    stream = ir.Var("host_stream", ir.PtrType("void"))
    scratch = ir.Var("scratch", tirx.TensorType([8], dtype, scope="shared.dyn"))
    body = [
        ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty])),
        ir.Evaluate(ir.TensorLoad(data, [0])),
    ]
    body = _thread("blockIdx.x", n, [_thread("threadIdx.x", 32, body)])
    scope = ir.RegionStmt(
        "tirx.device_scope",
        [ir.const(2), stream],
        [],
        {
            "target": Target("cuda"),
            "cuda.launch_fields": ["cluster.x", "stream"],
            "cuda.kernel_attrs": {},
            "tl.derive_launch_geometry": True,
        },
        [body],
    )
    function = tirx.Function(
        [data, n, stream],
        [scope],
        ret_type=ir.PrimType("void"),
        attrs=ir.DictAttrs({"global_symbol": "main", "target": Target("cuda", host="c")}),
    )
    return tvm_tirx.IRModule({"main": function})


def _regions(module, name):
    found = []
    tvm_ffi.structural_walk(
        module,
        (ir.RegionStmt, lambda node: found.append(node) if node.op.name == name else None),
    )
    return found


def _launch_field_values(call):
    start = 1 + int(call.attrs.num_kernel_args)
    return dict(zip(call.attrs.launch_fields, call.args[start:]))


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("int4", 4)])
def test_explicit_launch_waits_for_final_allocation_and_keeps_stream_on_host(dtype, bits):
    source = _configured_device_scope(dtype)
    original = ir.save_json(source)
    outlined = tilelang.transform.OutlineDeviceRegions()(source)
    assert not _calls(outlined, "tirx.call_ffi_kernel")
    (pending,) = _regions(outlined["main"], "tl.pending_cuda_launch")
    assert pending.args[1].same_as(outlined["main"].params[2])
    device = next(value for key, value in outlined.functions.items() if key.name_hint != "main")
    assert not any(param.name == "host_stream" for param in device.params)

    def grow_allocation(bind):
        if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor" and str(bind.var.ty.scope) == "shared.dyn":
            grown = ir.Var(bind.var.name, tirx.TensorType([64], dtype, scope="shared.dyn"))
            return ir.Bind(grown, ir.Call("tirx.alloc_tensor", [], ty_args=[grown.ty]))
        return bind

    grown = tvm_ffi.structural_map(outlined, (ir.Bind, grow_allocation), order="post")
    final = _lower(grown)
    assert not _regions(final, "tl.pending_cuda_launch")
    (call,) = _calls(final["main"], "tirx.call_ffi_kernel")
    fields = _launch_field_values(call)
    assert int(fields["dynamic_smem_bytes"]) == (64 * bits + 7) // 8
    assert fields["stream"].same_as(final["main"].params[2])
    assert int(fields["cluster.x"]) == 2 and int(fields["block.x"]) == 32
    assert not tirx.analysis.undefined_vars(final["main"])
    assert tirx.analysis.check_well_formed(final)
    assert ir.save_json(source) == original


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("int4", 4)])
def test_full_split_entry_point_consumes_pending_cuda_configuration(dtype, bits):
    source = _configured_device_scope(dtype)
    result = tilelang.transform.SplitHostDevice()(source)
    separate = _lower(tilelang.transform.OutlineDeviceRegions()(source))
    assert not _regions(result, "tl.pending_cuda_launch")
    (call,) = _calls(result["main"], "tirx.call_ffi_kernel")
    fields = _launch_field_values(call)
    assert int(fields["dynamic_smem_bytes"]) == (8 * bits + 7) // 8
    assert fields["stream"].same_as(result["main"].params[2])
    ir.assert_structural_equal(result, separate)
    assert tirx.analysis.check_well_formed(result)


def test_outline_preserves_targetless_declarations():
    declaration = tirx.Function([], [], ret_type=ir.PrimType("void"), attrs=ir.DictAttrs({"declaration": True}))
    main = tirx.Function([], [ir.Evaluate(0)], ret_type=ir.PrimType("void"))
    main = main.with_attr({"global_symbol": "main", "target": Target("c", host="c")})
    module = tvm_tirx.IRModule({"main": main, "external": declaration})
    result = tilelang.transform.OutlineDeviceRegions()(module)
    ir.assert_structural_equal(result["external"], declaration)
    # The native verifier requires definitions. Verify the unchanged declaration
    # above and the actual definition separately instead of treating the external
    # declaration as a malformed function body.
    assert tirx.analysis.check_well_formed(result["main"])


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("int4", 4), ("uint4", 4), ("int4x4", 16)])
def test_dynamic_storage_bytes_and_geometry_are_in_caller_coordinates(dtype, bits):
    module, symbol = _module(dtype=dtype)
    original = ir.save_json(module)
    result = _lower(module)
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(module) == original
    call = _calls(result["main"], "tirx.call_ffi_kernel")[0]
    assert call.args[0].value == "real_device"
    assert list(call.attrs.launch_fields) == ["grid.x", "block.x", "dynamic_smem_bytes"]
    assert not list(call.attrs.launch_params)
    assert list(result[symbol].attrs["cuda.launch_fields"]) == list(call.attrs.launch_fields)
    assert "tirx.kernel_launch_params" not in result[symbol].attrs
    assert int(call.attrs.num_kernel_args) == 2
    assert len(call.args) == 6
    host_extent = result["main"].params[1]
    for value in (1, 4, 17, 256):
        assert _eval(call.args[3], host_extent, value) == value + 1
        assert _eval(call.args[-1], host_extent, value) == ((value + 3) * bits + 7) // 8
    assert not tirx.analysis.undefined_vars(result["main"])
    # The finalized call is the runtime contract. Resource expressions stay
    # in the device body until launch lowering; no fork-only closed codec is
    # needed in Function attributes, including across JSON and SSA renewal.
    callee = result[symbol]
    assert "dyn_shared_memory_buf" not in callee.attrs
    renewed = tirx.transform.ConvertSSA()(result)
    assert tirx.analysis.check_well_formed(renewed)
    renewed_call = _calls(renewed["main"], "tirx.call_ffi_kernel")[0]
    assert _eval(renewed_call.args[-1], renewed["main"].params[1], 17) == ((17 + 3) * bits + 7) // 8
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)


@pytest.mark.parametrize("case,expected", [("local", "device-local variable"), ("conflict", "Incompatible launch extents")])
def test_launch_extents_cannot_depend_on_threads_or_conflict(case, expected):
    module, _ = _module(bad_extent=case)
    with pytest.raises(ValueError, match=expected):
        _lower(module)


def test_multiple_dynamic_allocations_must_be_merged_first():
    module, _ = _module(extra_allocation=True)
    with pytest.raises(ValueError, match="one final shared.dyn allocation"):
        _lower(module)


def test_launch_flags_become_explicit_values_and_dynamic_size_stays_last():
    module, symbol = _module(flags=True)
    marked = tilelang.cuda.transform.PersistThreadblock()(module)
    assert int(marked[symbol].attrs["use_cooperative_groups"]) == 1
    assert "use_cooperative_groups" not in (module[symbol].attrs or {})
    result = _lower(marked)
    call = _calls(result["main"], "tirx.call_ffi_kernel")[0]
    assert list(call.attrs.launch_fields) == [
        "grid.x",
        "block.x",
        "programmatic_stream_serialization",
        "cooperative",
        "dynamic_smem_bytes",
    ]
    assert not list(call.attrs.launch_params)
    assert len(call.args) == 8
    assert int(call.args[5]) == int(call.args[6]) == 1
    assert _eval(call.args[3], result["main"].params[1], 17) == 18
    assert _eval(call.args[-1], result["main"].params[1], 17) == 80
    assert list(result[symbol].attrs["cuda.launch_fields"]) == list(call.attrs.launch_fields)
    assert "tirx.kernel_launch_params" not in result[symbol].attrs
    ir.assert_structural_equal(_lower(result), result)


def test_persist_threadblock_does_not_mark_kernels_without_grid_sync():
    module, symbol = _module()
    result = tilelang.cuda.transform.PersistThreadblock()(module)
    assert "use_cooperative_groups" not in (result[symbol].attrs or {})
    ir.assert_structural_equal(result, module)
