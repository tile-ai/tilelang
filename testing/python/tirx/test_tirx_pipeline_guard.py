import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def plan(body, parameters):
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([])).with_attr(
        {"global_symbol": "guarded_pipeline", "target": Target("c"), "tirx.noalias": True}
    )
    return tvm_ffi.get_global_func("tl.transform.PipelinePlanning")()(ir.IRModule({"main": function}))


def execute(module, *arguments, compile_tilelang):
    module = tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(module)
    module = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(module)
    assert ir.check_well_formed(module)
    compile_tilelang(module, target="c")(*arguments)


def planned_loop(module):
    loops = []
    stmt_functor.post_order_visit(
        module, lambda node: loops.append(node) if isinstance(node, ir.For) and "software_pipeline_stage" in node.annotations else None
    )
    assert len(loops) == 1
    return loops[0]


@pytest.mark.parametrize(
    "relation,bound", [("lt", -1), ("lt", 5), ("le", 4), ("eq", 2), ("ge", 0), ("ge", 3), ("gt", 1), ("gt", 9), ("and", 2)]
)
def test_pipeline_restricts_only_constant_contiguous_induction_guards(relation, bound, *, compile_tilelang):
    source = TensorView.create("source", [8], "int32")
    positions = TensorView.create("positions", [8], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    output = TensorView.create("output", [8], "int32")
    iteration = ir.Var("iteration", "int32")
    offset = ir.Var("offset", "int32")
    comparisons = {
        "lt": lambda value: value < bound,
        "le": lambda value: value <= bound,
        "eq": lambda value: value == bound,
        "ge": lambda value: value >= bound,
        "gt": lambda value: value > bound,
        "and": lambda value: (value >= bound) & (value < 6),
    }
    predicate = ir.prim.And(iteration >= bound, iteration < 6) if relation == "and" else comparisons[relation](iteration)
    loop = ir.For(
        iteration,
        0,
        8,
        1,
        ir.ForKind.DEFAULT,
        ir.If(
            predicate,
            [
                ir.Bind(offset, positions.load(iteration)),
                temporary.store(0, source.load(offset)),
                output.store(iteration, temporary.load(0) * 2),
            ],
            None,
        ),
        annotations={"num_stages": 3},
    )
    module = plan([temporary.allocate(), loop], [source.tensor, positions.tensor, output.tensor])
    scheduled = planned_loop(module)
    assert list(map(int, scheduled.annotations["software_pipeline_stage"])) == [0, 2]
    assert not any(isinstance(statement, ir.If) for statement in scheduled.body.seq)
    values = np.arange(8, dtype="int32") * 11
    indices = np.arange(7, -1, -1, dtype="int32")
    actual = runtime.tensor(np.full(8, -99, dtype="int32"))
    execute(module, runtime.tensor(values), runtime.tensor(indices), actual, compile_tilelang=compile_tilelang)
    expected = np.full(8, -99, dtype="int32")
    for index in range(8):
        if comparisons[relation](index):
            expected[index] = values[indices[index]] * 2
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_pipeline_does_not_split_a_mutable_guard(*, compile_tilelang):
    flags = TensorView.create("flags", [8], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    output = TensorView.create("output", [8], "int32")
    iteration = ir.Var("iteration", "int32")
    loop = ir.For(
        iteration,
        0,
        8,
        1,
        ir.ForKind.DEFAULT,
        ir.If(
            flags.load(iteration) > 0,
            [
                temporary.store(0, iteration),
                flags.store(iteration, 0),
                output.store(iteration, temporary.load(0) + 1),
            ],
            None,
        ),
        annotations={"num_stages": 3},
    )
    module = plan([temporary.allocate(), loop], [flags.tensor, output.tensor])
    scheduled = planned_loop(module)
    assert isinstance(scheduled.body.seq[0], ir.If)
    assert list(map(int, scheduled.annotations["software_pipeline_stage"])) == [2]
    values = np.arange(8, dtype="int32") % 2
    actual = runtime.tensor(np.full(8, -99, dtype="int32"))
    conditions = runtime.tensor(values)
    execute(module, conditions, actual, compile_tilelang=compile_tilelang)
    np.testing.assert_array_equal(actual.numpy(), np.where(values, np.arange(8) + 1, -99))
    np.testing.assert_array_equal(conditions.numpy(), np.zeros(8, dtype="int32"))


def test_pipeline_does_not_replay_effectful_guards(tilelang_builders, *, compile_tilelang):
    counter = TensorView.create("counter", [1], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    output = TensorView.create("output", [13], "int32")
    iteration = ir.Var("iteration", "int32")
    condition = ir.Call("tirx.call_extern", [ir.StringImm("next_pipeline_guard"), counter.data], ty="int32")
    loop = ir.For(
        iteration,
        0,
        13,
        1,
        ir.ForKind.DEFAULT,
        ir.If(
            condition != 0,
            [
                temporary.store(0, iteration),
                output.store(iteration, temporary.load(0) + 1),
            ],
            None,
        ),
        annotations={"num_stages": 3},
    )
    imported = '#ifdef __cplusplus\nextern "C"\n#endif\nint32_t next_pipeline_guard(int32_t *counter) { return ((*counter)++) % 2; }\n'
    body = ir.RegionStmt("tl.scope", [], [], {"pragma_import_c": imported}, [temporary.allocate(), loop])
    module = plan(body, [counter.tensor, output.tensor])
    assert isinstance(planned_loop(module).body.seq[0], ir.If)
    actual = runtime.tensor(np.full(13, -99, dtype="int32"))
    count = runtime.tensor(np.array([0], dtype="int32"))
    execute(module, count, actual, compile_tilelang=compile_tilelang)
    np.testing.assert_array_equal(actual.numpy(), [index + 1 if index % 2 else -99 for index in range(13)])
    np.testing.assert_array_equal(count.numpy(), [13])
