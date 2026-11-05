"""Native CUDA fences preserve observable control flow and pointer access modes."""

import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, dtype="uint8", size=16, scope="shared"):
    return ir.Var(name, tirx.TensorType((size,), dtype, scope=scope))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def call(name, *args, dtype="void", attrs=None):
    result = ir.Call(name, list(args), attrs=attrs, ty=ir.PrimType(dtype))
    result.validate()
    return result


def evaluate(name, *args, **kwargs):
    return ir.Evaluate(call(name, *args, **kwargs))


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def calls(value, name):
    return [item for item in objects(value, ir.Call) if isinstance(item.op, ir.Op) and item.op.name == name]


def op_names(body):
    return [item.op.name for item in objects(body, ir.Call) if isinstance(item.op, ir.Op)]


def apply(name, body, tmp_path, *, params=(), arch="sm_100a"):
    function = tirx.Function(list(params), body).with_attr("global_symbol", "fences")
    if arch is not None:
        function = function.with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    operation = tvm_ffi.get_global_func("tl.cuda.transform." + name)()
    module = operation(tvm_tirx.IRModule({"fences": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(operation(module), module)
    assert ir.save_json(function) == before
    (tmp_path / "before.json").write_text(before)
    (tmp_path / "after.json").write_text(ir.save_json(module["fences"]))
    return function, module["fences"]


def tma_store(shared, global_tensor):
    # Native 1D ABI: global destination, shared source, bytes, reduce, cache hint.
    return evaluate("tl.tma_store", global_tensor.data, shared.data, integer(16), integer(0), integer(0))


def tma_load(shared, global_tensor, barrier):
    # Native 1D ABI differs from a tensor-map load: the barrier is operand 2.
    return evaluate("tl.tma_load", shared.data, global_tensor.data, barrier, integer(16), integer(0))


def access(value, mask, form):
    if form == "tensor":
        return ir.Call("tl.access_ptr", [ir.TensorLoad(value, [0]), integer(16), mask])
    result = ir.Call("tl.ptr_access", [value.data, integer(0), integer(16), mask])
    if form == "offset":
        result = ir.Call("tirx.ptr_byte_offset", [result, integer(4)])
    if form == "reinterpret":
        result = ir.Call("tirx.reinterpret", [result], ty=result.ty)
    return result


@pytest.mark.parametrize("form", ["tensor", "pointer", "offset", "reinterpret"])
@pytest.mark.parametrize("mask,expected", [(0, 0), (1, 0), (2, 1), (3, 1)])
def test_opaque_pointer_access_modes(form, mask, expected, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    pointer = access(shared, integer(mask), form)
    pointer.validate()
    opaque = ir.Evaluate(tirx.call_extern("opaque_pointer_consumer", pointer, ty="void"))
    _, result = apply("InjectFenceProxy", [allocate(shared), opaque, tma_store(shared, global_tensor)], tmp_path, params=[global_tensor])
    assert len(calls(result, "tl.fence_proxy_async")) == expected


@pytest.mark.parametrize("binding", [False, True])
def test_raw_shared_pointer_and_value_returning_opaque_calls_are_writes(binding, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    expression = tirx.call_extern("opaque_pointer_consumer", shared.data, ty="int32")
    statement = ir.Bind(ir.Var("token", "int32"), expression) if binding else ir.Evaluate(expression)
    _, result = apply("InjectFenceProxy", [allocate(shared), statement, tma_store(shared, global_tensor)], tmp_path, params=[global_tensor])
    assert len(calls(result, "tl.fence_proxy_async")) == 1


@pytest.mark.parametrize("kind", ["load", "barrier_wait", "barrier_expect", "global_pointer", "no_pointer"])
def test_scalar_loads_and_barrier_values_do_not_claim_generic_pointer_writes(kind, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    barrier = tensor("barrier", "uint64", 1, "shared.barrier")
    value = ir.TensorLoad(barrier, [0])
    if kind == "load":
        statement = ir.Evaluate(tirx.call_extern("consume_value", ir.TensorLoad(shared, [0]), ty="void"))
    elif kind == "barrier_wait":
        statement = evaluate("tl.mbarrier_wait_parity", value, integer(0))
    elif kind == "barrier_expect":
        statement = evaluate("tl.mbarrier_expect_tx", value, integer(16))
    else:
        args = [global_tensor.data] if kind == "global_pointer" else []
        statement = ir.Evaluate(tirx.call_extern("opaque", *args, ty="void"))
    _, result = apply(
        "InjectFenceProxy",
        [allocate(shared), allocate(barrier), statement, tma_store(shared, global_tensor)],
        tmp_path,
        params=[global_tensor],
    )
    assert not calls(result, "tl.fence_proxy_async")


@pytest.mark.parametrize("write", [False, True])
def test_native_cp_async_preserves_its_generic_destination_contract(write, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    copy = evaluate("tl.ptx_cp_async_bytes", access(shared, integer(2 if write else 1), "tensor"), global_tensor.data, integer(16))
    _, result = apply("InjectFenceProxy", [allocate(shared), copy, tma_store(shared, global_tensor)], tmp_path, params=[global_tensor])
    assert len(calls(result, "tl.fence_proxy_async")) == int(write)


def eval_integer(expression, values):
    if isinstance(expression, ir.IntImm):
        return int(expression.value)
    if isinstance(expression, ir.Var):
        return values[expression]
    if isinstance(expression, ir.TensorLoad):
        return values[expression.source]
    operations = {
        ir.prim.Add: lambda a, b: a + b,
        ir.prim.Sub: lambda a, b: a - b,
        ir.prim.LT: lambda a, b: a < b,
        ir.prim.LE: lambda a, b: a <= b,
        ir.prim.GE: lambda a, b: a >= b,
        ir.prim.EQ: lambda a, b: a == b,
        ir.prim.NE: lambda a, b: a != b,
    }
    return operations[type(expression)](eval_integer(expression.a, values), eval_integer(expression.b, values))


class LoopJump(Exception):
    def __init__(self, kind):
        self.kind = kind


def execution_trace(statement, values):
    """Execute the test's finite control flow independently of the pass analysis."""
    events = []

    def visit(node):
        if isinstance(node, ir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, ir.If):
            selected = node.then_case if eval_integer(node.condition, values) else node.else_case
            if selected is not None:
                visit(selected)
        elif isinstance(node, ir.For):
            for index in range(*(eval_integer(value, values) for value in (node.begin, node.end, node.step))):
                values[node.var] = index
                try:
                    visit(node.body)
                except LoopJump as jump:
                    if jump.kind == "break":
                        break
        elif isinstance(node, ir.While):
            trips = 0
            while eval_integer(node.condition, values):
                trips += 1
                assert trips <= 8, "Test oracle encountered an unexpectedly unbounded loop"
                try:
                    visit(node.body)
                except LoopJump as jump:
                    if jump.kind == "break":
                        break
        elif isinstance(node, ir.Bind):
            if isinstance(node.var.ty, tirx.TensorType):
                values[node.var] = 0
        elif isinstance(node, ir.TensorStore):
            values[node.dest] = eval_integer(node.value, values)
            if node.dest.ty.scope.name.startswith("shared"):
                events.append("generic")
        elif isinstance(node, ir.Evaluate):
            if isinstance(node.value, ir.Call):
                events.extend(
                    {
                        "tl.tma_store": ["async"],
                        "tl.fence_proxy_async": ["fence"],
                        "tirx.gpu_storage_sync": ["sync"],
                    }.get(node.value.op.name, [])
                )
        elif isinstance(node, ir.Break):
            raise LoopJump("break")
        elif isinstance(node, ir.Continue):
            raise LoopJump("continue")
        else:
            raise AssertionError(f"Unsupported statement in trace oracle: {type(node)}")

    visit(statement)
    return events


@pytest.mark.parametrize(
    "pattern",
    [
        "branch_join",
        "both_async",
        "one_async",
        "branch_write",
        "loop_backedge",
        "pure_async_loop",
        "zero_trip",
        "nested",
        "while",
        "manual",
        "break",
        "continue",
        "nested_break",
        "while_break",
        "while_continue",
        "sync",
    ],
)
def test_proxy_control_flow_has_no_unfenced_generic_to_async_trace(pattern, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    cond, size = ir.Var("condition", "bool"), ir.Var("size", "int32")
    index = ir.Var("index", "int32")
    store = ir.TensorStore(shared, [0], integer(7, "uint8"))
    issue = tma_store(shared, global_tensor)
    fence = evaluate("tl.fence_proxy_async")

    def loop(body):
        return ir.For(index, integer(3), size + 3, integer(1), ir.ForKind.DEFAULT, body)

    bodies = {
        "branch_join": [ir.If(cond, [store], [issue]), issue],
        "both_async": [store, ir.If(cond, [issue], [issue])],
        "one_async": [store, ir.If(cond, [issue], None), issue],
        "branch_write": [ir.If(cond, [issue, store], [store, issue]), issue],
        "loop_backedge": [loop([issue, store]), issue],
        "pure_async_loop": [store, loop([issue, issue]), issue],
        "zero_trip": [store, loop([issue]), issue],
        "nested": [loop([ir.If(cond, [issue, store], [store, issue])]), issue],
        "manual": [store, fence, ir.If(cond, [issue], [issue]), store, fence, issue],
        "break": [loop([store, ir.If(cond, [ir.Break()], None), issue]), issue],
        "continue": [loop([issue, store, ir.If(cond, [ir.Continue()], None), fence]), issue],
        "sync": [store, ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")])), issue],
    }
    if pattern == "nested_break":
        inner = ir.For(
            ir.Var("inner", "int32"),
            integer(0),
            integer(2),
            integer(1),
            ir.ForKind.DEFAULT,
            [store, ir.If(cond, [ir.Break()], None), issue],
        )
        bodies[pattern] = [loop([inner, issue]), issue]
    if pattern in ("while", "while_break", "while_continue"):
        counter = tensor("counter", "int32", 1, "local")
        update = ir.TensorStore(counter, [0], ir.TensorLoad(counter, [0]) + 1)
        suffix = []
        if pattern != "while":
            jump = ir.Break() if pattern == "while_break" else ir.Continue()
            suffix = [ir.If(cond, [jump], None), fence]
        bodies[pattern] = [
            allocate(counter),
            ir.TensorStore(counter, [0], integer(0)),
            ir.While(ir.TensorLoad(counter, [0]) < size, [update, issue, store, *suffix]),
            issue,
        ]
    before, result = apply("InjectFenceProxy", [allocate(shared), *bodies[pattern]], tmp_path, params=[global_tensor, cond, size])
    records = []
    for flag, count in itertools.product([False, True], range(4)):
        state = {cond: flag, size: count}
        original_trace = execution_trace(before.body, state.copy())
        trace = execution_trace(result.body, state.copy())
        assert [event for event in trace if event != "fence"] == [event for event in original_trace if event != "fence"]
        dirty = False
        for event in trace:
            if event == "generic":
                dirty = True
            elif event == "fence":
                dirty = False
            elif event == "async":
                assert not dirty, {"pattern": pattern, "condition": flag, "size": count, "trace": trace}
        records.append({"condition": flag, "size": count, "before": original_trace, "after": trace})
    (tmp_path / "traces.json").write_text(json.dumps(records, indent=2))
    if pattern in ("both_async", "pure_async_loop"):
        assert len(calls(result, "tl.fence_proxy_async")) == 1


@pytest.mark.parametrize("arch", [None, "sm_80", "sm_89"])
def test_proxy_pass_respects_target_gating(arch, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    _, result = apply(
        "InjectFenceProxy",
        [allocate(shared), ir.TensorStore(shared, [0], integer(7, "uint8")), tma_store(shared, global_tensor)],
        tmp_path,
        params=[global_tensor],
        arch=arch,
    )
    assert not calls(result, "tl.fence_proxy_async")


@pytest.mark.parametrize("kind", ["one_dimensional", "tiled", "im2col", "loop", "nested_election", "adjacent_elections"])
def test_expect_tx_fuses_only_the_matching_local_arrival(kind, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    barrier = tensor("barrier", "uint64", 1, "shared.barrier")
    value = ir.TensorLoad(barrier, [0])
    descriptor = ir.Var("descriptor", ir.PtrType(ir.PrimType("uint8"), "global"))
    issue = tma_load(shared, global_tensor, value)
    if kind in ("tiled", "im2col"):
        name = "tl.tma_load_im2col" if kind == "im2col" else "tl.tma_load"
        issue = evaluate(name, descriptor, value, shared.data, integer(0), integer(0), integer(0))
    if kind == "loop":
        issue = ir.For(ir.Var("index", "int32"), integer(1), integer(3), integer(1), ir.ForKind.DEFAULT, [issue])
    body = [evaluate("tl.mbarrier_expect_tx", value, integer(16)), issue, evaluate("tl.ptx_arrive_barrier", value)]
    cond = ir.Var("elected", "bool")
    if kind == "nested_election":
        body = [ir.If(cond, [ir.If(cond, body, None)], None)]
    elif kind == "adjacent_elections":
        body = [ir.If(cond, [statement], None) for statement in body]
    _, result = apply(
        "FuseMBarrierArriveExpectTx", [allocate(shared), allocate(barrier), *body], tmp_path, params=[global_tensor, descriptor, cond]
    )
    assert len(calls(result, "tl.ptx_arrive_barrier_expect_tx")) == 1
    assert not calls(result, "tl.mbarrier_expect_tx")
    assert not calls(result, "tl.ptx_arrive_barrier")
    assert len(calls(result, "tl.tma_load") + calls(result, "tl.tma_load_im2col")) == 1


@pytest.mark.parametrize(
    "kind", ["remote_cta", "different_arrive", "different_load", "opaque_intervening", "no_load", "conditional_load", "effectful_bound"]
)
def test_expect_tx_nonfusible_sequences_keep_their_barrier_contract(kind, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    barrier = tensor("barrier", "uint64", 2, "shared.barrier")
    value, other = ir.TensorLoad(barrier, [0]), ir.TensorLoad(barrier, [1])
    load = tma_load(shared, global_tensor, other if kind == "different_load" else value)
    arrive = evaluate("tl.ptx_arrive_barrier", other if kind == "different_arrive" else value)
    if kind == "remote_cta":
        arrive = evaluate("tl.ptx_arrive_cluster_barrier", value, integer(1))
    middle = [load]
    cond = ir.Var("elected", "bool")
    if kind == "opaque_intervening":
        middle.append(ir.Evaluate(tirx.call_extern("opaque", ty="void")))
    elif kind == "no_load":
        middle = []
    elif kind == "conditional_load":
        middle = [ir.If(cond, [load], None)]
    elif kind == "effectful_bound":
        middle = [
            ir.For(ir.Var("index", "int32"), integer(0), tirx.call_extern("read_bound", ty="int32"), integer(1), ir.ForKind.DEFAULT, [load])
        ]
    _, result = apply(
        "FuseMBarrierArriveExpectTx",
        [allocate(shared), allocate(barrier), evaluate("tl.mbarrier_expect_tx", value, integer(16)), *middle, arrive],
        tmp_path,
        params=[global_tensor, cond],
    )
    assert not calls(result, "tl.ptx_arrive_barrier_expect_tx")
    assert len(calls(result, "tl.mbarrier_expect_tx")) == 1
    if kind == "remote_cta":
        (remote,) = calls(result, "tl.ptx_arrive_cluster_barrier")
        assert int(remote.args[1]) == 1
    else:
        assert len(calls(result, "tl.ptx_arrive_barrier")) == 1


def tmem_load(word, local):
    return evaluate("tl.tcgen05_ld", integer(32), integer(128), integer(0, "bool"), ir.TensorLoad(word, [0]), integer(0), local.data)


@pytest.mark.parametrize(
    "kind",
    [
        "shared_sync",
        "dynamic_shared_sync",
        "wait_arrive",
        "remote_arrive",
        "manual",
        "boundary",
        "tcgen_commit",
        "nested_branch",
        "loop",
        "cp",
    ],
)
def test_tcgen05_thread_handoffs_preserve_fence_order_and_manual_fences(kind, tmp_path):
    word, local = tensor("word", "uint32", 1), tensor("local", "float32", 32, "local")
    barrier, shared = tensor("barrier", "uint64", 1, "shared.barrier"), tensor("shared")
    value = ir.TensorLoad(barrier, [0])
    wait, arrive = evaluate("tl.mbarrier_wait_parity", value, integer(0)), evaluate("tl.ptx_arrive_barrier", value)
    use = tmem_load(word, local)
    before, after = evaluate("tl.tcgen05_before_thread_sync"), evaluate("tl.tcgen05_after_thread_sync")
    cond = ir.Var("elected", "bool")
    if kind in ("shared_sync", "dynamic_shared_sync"):
        body = [
            ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared.dyn" if kind == "dynamic_shared_sync" else "shared")])),
            use,
        ]
    elif kind == "manual":
        body = [wait, after, use, before, arrive]
    elif kind == "boundary":
        body = [wait, arrive, use]
    elif kind == "tcgen_commit":
        body = [use, evaluate("tl.tcgen05_mma_arrive", value)]
    else:
        if kind == "remote_arrive":
            arrive = evaluate("tl.ptx_arrive_cluster_barrier", value, integer(1))
        if kind == "cp":
            use = evaluate("tl.ptx_tcgen05_cp_warpx4", shared.data, word.data, integer(0))
        body = [wait, use, arrive]
        if kind == "nested_branch":
            body = [ir.If(cond, body, None)]
        elif kind == "loop":
            body = [ir.For(ir.Var("index", "int32"), integer(2), integer(6), integer(2), ir.ForKind.DEFAULT, body)]
    _, result = apply("InjectTcgen05Fence", [*(allocate(v) for v in (word, local, barrier, shared)), *body], tmp_path, params=[cond])
    count = 0 if kind in ("boundary", "tcgen_commit") else 1
    assert len(calls(result, "tl.tcgen05_before_thread_sync")) == count
    assert len(calls(result, "tl.tcgen05_after_thread_sync")) == count
    if count:
        names = op_names(result)
        if kind in ("shared_sync", "dynamic_shared_sync"):
            assert (
                names.index("tl.tcgen05_before_thread_sync")
                < names.index("tirx.gpu_storage_sync")
                < names.index("tl.tcgen05_after_thread_sync")
            )
        else:
            use_name = "tl.ptx_tcgen05_cp_warpx4" if kind == "cp" else "tl.tcgen05_ld"
            arrive_name = "tl.ptx_arrive_cluster_barrier" if kind == "remote_arrive" else "tl.ptx_arrive_barrier"
            assert names.index("tl.mbarrier_wait_parity") < names.index("tl.tcgen05_after_thread_sync") < names.index(use_name)
            assert names.index(use_name) < names.index("tl.tcgen05_before_thread_sync") < names.index(arrive_name)


@pytest.mark.parametrize("arch", [None, "sm_80", "sm_90a"])
def test_tcgen05_fences_are_gated_on_the_target_architecture(arch, tmp_path):
    word, local = tensor("word", "uint32", 1), tensor("local", "float32", 32, "local")
    _, result = apply(
        "InjectTcgen05Fence",
        [allocate(word), allocate(local), ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")])), tmem_load(word, local)],
        tmp_path,
        arch=arch,
    )
    assert not calls(result, "tl.tcgen05_before_thread_sync")
    assert not calls(result, "tl.tcgen05_after_thread_sync")


@pytest.mark.parametrize("kind", ["proxy", "fused_tma", "tcgen05"])
def test_generated_fence_sequence_compiles_with_the_cuda_sdk(kind, tmp_path):
    shared, global_tensor = tensor("shared"), tensor("global", scope="global")
    barrier = tensor("barrier", "uint64", 1, "shared.barrier")
    value = ir.TensorLoad(barrier, [0])
    if kind == "proxy":
        body = [allocate(shared), ir.TensorStore(shared, [0], integer(7, "uint8")), tma_store(shared, global_tensor)]
        operation, spelling = "InjectFenceProxy", "tl::fence_proxy_async"
    elif kind == "fused_tma":
        body = [
            allocate(shared),
            allocate(barrier),
            evaluate("tl.mbarrier_expect_tx", value, integer(16)),
            tma_load(shared, global_tensor, value),
            evaluate("tl.ptx_arrive_barrier", value),
        ]
        operation, spelling = "FuseMBarrierArriveExpectTx", ".arrive_and_expect_tx("
    else:
        word, local = tensor("word", "uint32", 1), tensor("local", "float32", 32, "local")
        body = [
            allocate(word),
            allocate(local),
            allocate(barrier),
            evaluate("tl.mbarrier_wait_parity", value, integer(0)),
            tmem_load(word, local),
            evaluate("tl.ptx_arrive_barrier", value),
        ]
        operation, spelling = "InjectTcgen05Fence", "tl::tcgen05_after_thread_sync"
    thread = ir.Var("thread", "int32")
    region = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(128)], [thread], {}, body)
    _, function = apply(operation, [region], tmp_path, params=[global_tensor])
    function = function.with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="c"))
    split = tirx.transform.SplitHostDevice()(tvm_tirx.IRModule({"fences": function}))
    kernels = {symbol: value for symbol, value in split.functions.items() if int(value.attrs.get("calling_conv", 0)) == 2}
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm_tirx.IRModule(kernels))
    assert tirx.analysis.check_well_formed(lowered)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        lowered, tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    source = module.inspect_source("cuda")
    source_path = tmp_path / "fences.cu"
    source_path.write_text(source)
    assert spelling in source
    nvcc = shutil.which("nvcc")
    root = Path(__file__).resolve().parents[3]
    cutlass = Path(os.environ.get("TILELANG_TEST_CUTLASS_INCLUDE", root / "3rdparty/cutlass/include"))
    if nvcc is None or not (cutlass / "cute").is_dir():
        pytest.skip("CUDA SDK and fixed CUTLASS headers are needed for device compilation")
    command = [
        nvcc,
        "-std=c++17",
        "--cubin",
        "-arch=sm_100a",
        f"-I{root / 'src'}",
        f"-I{cutlass}",
        str(source_path),
        "-o",
        str(tmp_path / "fences.cubin"),
    ]
    compilation = subprocess.run(command, text=True, capture_output=True, check=False)
    (tmp_path / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": compilation.returncode}, indent=2))
    (tmp_path / "nvcc.log").write_text(compilation.stdout + compilation.stderr)
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
