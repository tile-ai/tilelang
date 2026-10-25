import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def transform(name, body, params=(), *arguments):
    function = tirx.Function(list(params), body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "metadata")
    result = tvm_ffi.get_global_func(f"tl.{name}")(*arguments)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def calls(function, name):
    result = []
    stmt_functor.post_order_visit(
        function, lambda node: result.append(node) if isinstance(node, ir.Call) and node.op.name == name else None
    )
    return result


@pytest.mark.parametrize("copy_name", ["copy", "async_copy", "tma_copy"])
@pytest.mark.parametrize("bound_pointer", [False, True])
def test_device_bound_tma_markers_follow_native_tensor_declarations(copy_name, bound_pointer):
    source = TensorView.create("source", [32], "float32")
    destination = TensorView.create("destination", [32], "float32")
    alias = TensorView.create("alias", [32], "float32")
    pointer = source.data
    body = []
    if bound_pointer:
        pointer = ir.Var("device_pointer", pointer.ty)
        body.append(ir.Bind(pointer, source.data))
    body.extend([alias.declare(pointer), ir.Evaluate(ir.Call(f"tl.tileop.{copy_name}", [alias.tensor, destination.tensor]))])
    function = transform("cuda.transform.AnnotateDeviceBoundTmaCopies", ir.SeqStmt(body), [source.tensor, destination.tensor])
    call = calls(function, f"tl.tileop.{copy_name}")[0]
    marked = call.attrs is not None and bool(call.attrs.get("tma_descriptor_base_is_device_bound", False))
    assert marked == bound_pointer


def test_cpu_atomic_marker_survives_native_function_roundtrip():
    output = TensorView.create("output", [1], "float32")
    pointer = tvm_ffi.get_global_func("tl.MakeAccessPtrFromBufferLoad")(output.load(0), 3)
    atomic = ir.Call("tl.atomic_add_elem_op", [pointer, ir.const(1.0), ir.const(0)], ty="float32")
    function = transform("cpu.transform.MarkCPUAtomics", ir.Evaluate(atomic), [output.tensor])
    assert function.attrs["tl.cpu_had_atomics"]
    assert not transform("cpu.transform.MarkCPUAtomics", output.store(0, 1), [output.tensor]).attrs.get("tl.cpu_had_atomics", False)


@pytest.mark.parametrize("operation,attribute", [("pdl_trigger", "has_cuda_pdl_trigger"), ("pdl_sync", "has_cuda_pdl_sync")])
def test_pdl_markers_and_unsupported_target_diagnostics(operation, attribute):
    body = ir.Evaluate(ir.Call(f"tl.{operation}", [], ty=ir.TupleType([])))
    function = transform("cuda.transform.MarkCudaSyncCalls", body, [], True)
    assert function.attrs[attribute]
    with pytest.raises(ValueError, match="PDL is not supported"):
        transform("cuda.transform.MarkCudaSyncCalls", body, [], False)


def test_pdl_sync_rejects_read_only_cache_loads():
    source = TensorView.create("source", [1], "float32")
    body = ir.SeqStmt(
        [
            ir.Evaluate(ir.Call("tl.pdl_sync", [], ty=ir.TupleType([]))),
            ir.Evaluate(ir.Call("tl.__ldg", [source.data], ty="float32")),
        ]
    )
    with pytest.raises(ValueError, match="Cannot invoke __ldg"):
        transform("cuda.transform.MarkCudaSyncCalls", body, [source.tensor], True)


def test_cooperative_launch_marker_tracks_grid_sync():
    function = transform("cuda.transform.PersistThreadblock", ir.Evaluate(ir.Call("tl.sync_grid", [], ty=ir.TupleType([]))))
    assert function.attrs["use_cooperative_groups"]


@pytest.mark.parametrize("kind", ["allocation", "operation", "multiplicity"])
def test_reducer_consumption_verifier_rejects_live_native_constructs(kind):
    tensor = TensorView.create("reducer", [4], "float32", scope="local.reducer")
    if kind == "allocation":
        body = tensor.allocate()
    elif kind == "operation":
        tensor = TensorView.create("ordinary", [4], "float32", scope="local.fragment")
        body = ir.SeqStmt(
            [tensor.allocate(), ir.Evaluate(ir.Call("tl.reducer_update", [tensor.load(0), ir.const(1.0)], ty=ir.TupleType([])))]
        )
    else:
        body = ir.RegionStmt("tl.parallel_multiplicity", [], [], None, ir.SeqStmt([]))
    with pytest.raises(RuntimeError, match="survived past"):
        transform("transform.VerifyReducerConsumed", body)


@pytest.mark.parametrize(
    "name,args,match",
    [
        ("__tvm_error_dtype_mismatch", ["kernel", "source", 0, 32, 1, 2, 16, 1], "dtype expected float16, but got int32"),
        ("__tvm_error_ndim_mismatch", ["kernel", "source", 2, 1], "ndim expected 2, but got 1"),
        ("__tvm_error_byte_offset_mismatch", ["kernel", "source", 0, 16], "byte_offset expected 0, but got 16"),
        ("__tvm_error_device_type_mismatch", ["kernel", "source", 1, 2], "device_type expected cpu, but got cuda"),
    ],
)
def test_standalone_runtime_error_helpers(name, args, match):
    with pytest.raises(RuntimeError, match=match):
        tvm_ffi.get_global_func(name)(*args)
