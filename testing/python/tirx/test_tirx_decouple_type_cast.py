import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.ir.prim import Cast, if_then_else
from tvm_tirx.target import Target

from tilelang._compat.stmt_functor import post_order_visit
from tilelang._compat.tensor import TensorView
from tilelang.transform import DecoupleTypeCast


def tensor(name, dtype="float32", scope="global", shape=(16,)):
    return TensorView.create(name, shape, dtype, scope=scope).tensor


def make_loop(statement, variable, *, begin=0, end=16, step=1, kind=ir.ForKind.VECTORIZED, loc=ir.UNKNOWN_LOC):
    return ir.For(variable, begin, end, step, kind, statement, loc=loc)


def transform(body, params=()):
    function = tirx.Function(params, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "main")
    return DecoupleTypeCast()(ir.IRModule({"main": function}))["main"]


def collect_nodes(body, node_type):
    result = []
    post_order_visit(body, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


@pytest.mark.parametrize("direction", ["to_memory", "from_memory"])
def test_decouple_native_tensors(direction):
    memory = tensor("memory", "float16")
    local = tensor("local", "float32", "local")
    source, destination = (local, memory) if direction == "to_memory" else (memory, local)
    index = ir.Var("index", "int32")
    store = ir.TensorStore(destination, [index], Cast(destination.ty.dtype, ir.TensorLoad(source, [index])))
    result = transform(make_loop(store, index), [memory, local])
    region = result.body.seq[0]
    assert region.op.name == "tirx.scope"
    allocation, first, second = region.body.seq
    assert isinstance(allocation, ir.Bind)
    assert allocation.value.op.name == "tirx.alloc_tensor"
    assert allocation.var.ty.scope.name == "local"
    assert allocation.var.ty.dtype == memory.ty.dtype
    assert int(allocation.var.ty.shape[0]) == 16
    assert first.kind == second.kind == ir.ForKind.VECTORIZED
    assert first.body.seq[0].dest.same_as(allocation.var)
    assert second.body.seq[0].dest.same_as(destination)


@pytest.mark.parametrize("begin", [2, "symbolic"])
def test_rmw_preserves_native_loop_domain(begin):
    memory = tensor("memory", "float16")
    index = ir.Var("index", "int32")
    begin = ir.Var("start", "int32") if begin == "symbolic" else ir.const(begin)
    value = Cast("float16", Cast("float32", ir.TensorLoad(memory, [index])) + ir.const(1.0, "float32"))
    original = make_loop(ir.TensorStore(memory, [index], value), index, begin=begin, end=begin + 4)
    result = transform(original, [memory, *([begin] if isinstance(begin, ir.Var) else [])])
    allocation, *loops = result.body.seq[0].body.seq
    assert int(allocation.var.ty.shape[0]) == 4
    assert len(loops) == 3
    analyzer = sym.Analyzer()
    for loop in loops:
        assert tvm_ffi.structural_equal(loop.begin, original.begin)
        assert tvm_ffi.structural_equal(loop.end, original.end)
        for access in collect_nodes(loop, (ir.TensorLoad, ir.TensorStore)):
            accessed = access.source if isinstance(access, ir.TensorLoad) else access.dest
            if accessed.same_as(allocation.var):
                assert int(analyzer.simplify(access.indices[0] - (loop.var - begin))) == 0


def test_distinct_indices_and_rmw_share_only_matching_staging():
    memory = tensor("memory", "float16", shape=(32,))
    index = ir.Var("index", "int32")
    value = Cast("float16", Cast("float32", ir.TensorLoad(memory, [index])) + Cast("float32", ir.TensorLoad(memory, [index + 16])))
    result = transform(make_loop(ir.TensorStore(memory, [index], value), index), [memory])
    allocations = [node for node in result.body.seq[0].body.seq if isinstance(node, ir.Bind)]
    loops = [node for node in result.body.seq[0].body.seq if isinstance(node, ir.For)]
    assert len(allocations) == 2
    assert len(loops) == 4
    compute = loops[-2]
    assert compute.body.seq[0].dest.same_as(allocations[0].var)
    assert all(not access.source.same_as(memory) for access in collect_nodes(compute, ir.TensorLoad))


@pytest.mark.parametrize("step", [2, -1])
def test_nonunit_step_is_unchanged(step):
    memory = tensor("memory")
    index = ir.Var("index", "int32")
    loop = make_loop(ir.TensorStore(memory, [index], Cast("float32", index)), index, step=step)
    assert tvm_ffi.structural_equal(transform(loop, [memory]).body.seq[0], loop)


@pytest.mark.parametrize("case", ["no_cast", "index_cast", "local", "compound", "if_else", "scalar_load"])
def test_ineligible_loop_is_unchanged(case):
    memory = tensor("memory")
    local = tensor("local", scope="local")
    index = ir.Var("index", "int32")
    if case == "no_cast":
        body = ir.TensorStore(memory, [index], ir.TensorLoad(local, [index]))
    elif case == "index_cast":
        body = ir.TensorStore(memory, [Cast("int64", index)], ir.TensorLoad(local, [index]))
    elif case == "local":
        body = ir.TensorStore(local, [index], Cast("float32", index))
    elif case == "scalar_load":
        body = ir.TensorStore(local, [index], Cast("float32", Cast("float64", ir.TensorLoad(memory, [0]))))
    else:
        store = ir.TensorStore(memory, [index], Cast("float32", index))
        body = ir.SeqStmt([store, store]) if case == "compound" else ir.If(index < 8, store, store)
    loop = make_loop(body, index)
    assert tvm_ffi.structural_equal(transform(loop, [memory, local]).body.seq[0], loop)


def test_flat_and_branch_local_bind_chains_are_normalized():
    memory = tensor("memory")
    index = ir.Var("index", "int32")
    first = ir.Var("first", "int32")
    second = ir.Var("second", "float32")
    body = [
        ir.Bind(first, index + 1),
        ir.If(index < 8, [ir.Bind(second, Cast("float32", first)), ir.TensorStore(memory, [index], second)], None),
    ]
    result = transform(make_loop(body, index), [memory])
    region = result.body.seq[0]
    assert isinstance(region, ir.RegionStmt)
    assert len(collect_nodes(region, ir.Bind)) == 1
    variables = collect_nodes(region, ir.Var)
    assert not any(variable.same_as(first) or variable.same_as(second) for variable in variables)


def test_nested_guards_keep_short_circuit_evaluation():
    memory = tensor("memory")
    index = ir.Var("index", "int32")
    store = ir.TensorStore(memory, [index], Cast("float32", index))
    result = transform(make_loop(ir.If(index != 0, ir.If(16 // index > 2, store, None), None), index), [memory])
    copy_condition = result.body.seq[0].body.seq[-1].body.seq[0].condition
    assert isinstance(copy_condition, ir.Call)
    assert copy_condition.op.name == "prim.if_then_else"
    assert int(copy_condition.args[2]) == 0


@pytest.mark.parametrize("kind", ["memory", "effectful", "effectful_bind"])
def test_effects_are_not_duplicated_or_moved(kind):
    memory = tensor("memory")
    index = ir.Var("index", "int32")
    store = ir.TensorStore(memory, [index], Cast("float32", index))
    if kind == "memory":
        body = ir.If(ir.TensorLoad(memory, [index]) > ir.const(0.0, "float32"), store, None)
    elif kind == "effectful":
        body = ir.If(tirx.call_extern("guard", index, ty="bool"), store, None)
    else:
        temporary = ir.Var("temporary", "float32")
        body = [ir.Bind(temporary, tirx.call_extern("value", index, ty="float32")), store]
    loop = make_loop(body, index)
    assert tvm_ffi.structural_equal(transform(loop, [memory]).body.seq[0], loop)


def test_conditional_load_does_not_become_an_eager_copy():
    memory = tensor("memory", "float16")
    index = ir.Var("index", "int32")
    value = if_then_else(index < 8, Cast("float32", ir.TensorLoad(memory, [index + 8])), ir.const(0.0, "float32"))
    store = ir.TensorStore(memory, [index], Cast("float16", value))
    result = transform(make_loop(store, index), [memory])
    allocation, compute, copy = result.body.seq[0].body.seq
    assert isinstance(allocation, ir.Bind)
    assert len(collect_nodes(compute, ir.TensorLoad)) == 1
    assert collect_nodes(compute, ir.TensorLoad)[0].source.same_as(memory)
    assert copy.body.seq[0].dest.same_as(memory)


def test_storage_rewrite_preserves_region_allocation_boundary():
    memory = tensor("memory")
    outer = ir.Var("outer", "int32")
    index = ir.Var("index", "int32")
    loop = make_loop(ir.TensorStore(memory, [index], Cast("float32", index)), index)
    function = transform(make_loop(loop, outer, end=2, kind=ir.ForKind.DEFAULT), [memory]).with_attr("target", Target("c"))
    module = tirx.transform.VectorizeLoop(enable_vectorize=False)(ir.IRModule({"main": function}))
    result = tirx.transform.StorageRewrite()(module)["main"]
    assert isinstance(result.body.seq[0], ir.For)
    region = result.body.seq[0].body.seq[0]
    assert region.op.name == "tirx.scope"
    assert isinstance(region.body.seq[0], ir.Bind)


@pytest.mark.parametrize("begin,extent", [(-3, 4), (2, 4), (2, 0), (2, 1)])
@pytest.mark.parametrize("index_type", ["int32", "int64"])
def test_native_cpu_execution_preserves_loop_domain(begin, extent, index_type):
    memory = tensor("memory", shape=(8,))
    index = ir.Var("index", index_type)
    start = ir.const(begin, index_type)
    loop = make_loop(
        ir.TensorStore(memory, [index - start], Cast("float32", index)),
        index,
        begin=start,
        end=ir.const(begin + extent, index_type),
        step=ir.const(1, index_type),
    )
    function = transform(loop, [memory])
    module = tirx.transform.VectorizeLoop(enable_vectorize=False)(ir.IRModule({"main": function}))
    executable = tirx.compile(module, target="c")
    actual = runtime.tensor(np.full(8, -999, dtype="float32"))
    executable(actual)
    expected = np.full(8, -999, dtype="float32")
    expected[:extent] = np.arange(begin, begin + extent, dtype="float32")
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
def test_decoupled_native_kernel_executes_on_cuda():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A TIRx CUDA runtime and CUDA device are required")
    memory = tensor("memory", shape=(128,))
    thread = ir.Var("thread", "int32")
    index = ir.Var("index", "int32")
    offset = thread * 4 + index
    value = Cast("float32", Cast("float64", ir.TensorLoad(memory, [offset])) + ir.const(0.25, "float64"))
    loop = make_loop(ir.TensorStore(memory, [offset], value), index, end=4)
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, loop)
    function = transform(launch, [memory])
    assert collect_nodes(function, ir.RegionStmt)[0].op.name == "tirx.scope"
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(function, target=target)
    values = np.arange(128, dtype="float32")
    actual = runtime.tensor(values, device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), values + 0.25)


def test_declaration_is_preserved():
    assert transform(None).body is None
