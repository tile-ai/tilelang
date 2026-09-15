import itertools

import pytest

import tilelang.language as language
import tilelang.testing
from tilelang import tvm
from tvm.script.ir_builder import IRBuilder


def _make_persistent(domain, wave_size, **options):
    worker = tvm.tirx.Var("worker", "int32")
    with IRBuilder() as builder, language.Persistent(domain, wave_size, worker, **options) as coordinates:
        if len(domain) == 1:
            coordinates = [coordinates]
        for coordinate in coordinates:
            language.evaluate(coordinate)

    bindings = {}
    loops = []

    def collect(statement):
        if isinstance(statement, tvm.tirx.Bind):
            bindings[statement.var] = statement.value
        elif isinstance(statement, tvm.tirx.For):
            loops.append(statement)

    tvm.tirx.stmt_functor.post_order_visit(builder.get(), collect)
    assert len(loops) == 1
    loop = loops[0]
    return loop, worker, tvm.tirx.stmt_functor.substitute(loop.body, bindings)


def _trace_persistent(loop, worker, body, wave_size, parameters):
    analyzer = tvm.arith.Analyzer()

    def evaluate(expression, values):
        result = analyzer.simplify(tvm.tirx.stmt_functor.substitute(expression, values))
        assert isinstance(result, tvm.tirx.IntImm)
        return int(result)

    def execute(statement, values, coordinates):
        if isinstance(statement, tvm.tirx.SeqStmt):
            return any(execute(child, values, coordinates) for child in statement.seq)
        if isinstance(statement, tvm.tirx.IfThenElse):
            branch = statement.then_case if evaluate(statement.condition, values) else statement.else_case
            return execute(branch, values, coordinates) if branch is not None else False
        assert isinstance(statement, tvm.tirx.Evaluate)
        if isinstance(statement.value, tvm.tirx.Call):
            assert statement.value.op.name == "tl.loop_break"
            return True
        coordinates.append(evaluate(statement.value, values))
        return False

    events = []
    for worker_index in range(wave_size):
        for wave_index in range(evaluate(loop.extent, parameters)):
            values = {**parameters, worker: worker_index, loop.loop_var: wave_index}
            coordinates = []
            if execute(body, values, coordinates):
                assert not coordinates
                break
            if coordinates:
                events.append((wave_index * wave_size + worker_index, tuple(coordinates)))
    return sorted(events)


@pytest.mark.parametrize("domain", [(11,), (2, 4), (3, 10), (2, 3, 10)])
@pytest.mark.parametrize("wave_size", [1, 3, 16, 128])
@pytest.mark.parametrize("group_size", [None, 0, 1, 3, 32])
@pytest.mark.parametrize("symbolic", [False, True])
def test_persistent_group_order_and_tail_coverage(domain, wave_size, group_size, symbolic):
    options = {} if group_size is None else {"group_size": group_size}
    parameters = {}
    if symbolic:
        extents = [tvm.tirx.Var(f"extent_{axis}", "int32") for axis in range(len(domain))]
        workers = tvm.tirx.Var("workers", "int32")
        parameters = dict(zip(extents, domain))
        parameters[workers] = wave_size
        if group_size is not None:
            group = tvm.tirx.Var("group", "int32")
            parameters[group] = group_size
            options["group_size"] = group
    else:
        extents, workers = list(domain), wave_size

    loop, worker, body = _make_persistent(extents, workers, **options)
    events = _trace_persistent(loop, worker, body, wave_size, parameters)

    effective_group = max(1, min(8 if group_size is None else group_size, domain[-1]))
    expected = []
    linear_index = 0
    for group_start in range(0, domain[-1], effective_group):
        for outer in itertools.product(*(range(extent) for extent in domain[:-1])):
            for offset in range(effective_group):
                if group_start + offset < domain[-1]:
                    expected.append((linear_index, (*outer, group_start + offset)))
                linear_index += 1

    assert events == expected
    assert sorted(coordinates for _, coordinates in events) == list(itertools.product(*(range(extent) for extent in domain)))


@pytest.mark.parametrize("domain", [(2, 4), (3, 10), (2, 3, 10)])
def test_persistent_full_last_dimension_group_preserves_row_major_order(domain):
    loop, worker, body = _make_persistent(list(domain), 3, group_size=domain[-1])
    events = _trace_persistent(loop, worker, body, 3, {})
    assert [coordinates for _, coordinates in events] == list(itertools.product(*(range(extent) for extent in domain)))


@pytest.mark.parametrize("num_stages", [0, 2, 6])
@pytest.mark.parametrize("annotated_stages", [None, 4])
def test_persistent_preserves_pipeline_annotations(num_stages, annotated_stages):
    annotations = {"persistent_test": 7}
    if annotated_stages is not None:
        annotations["num_stages"] = annotated_stages
    original = annotations.copy()
    loop, _, _ = _make_persistent([2, 9], 3, num_stages=num_stages, annotations=annotations)
    assert int(loop.annotations["persistent_test"]) == 7
    expected_stages = num_stages or annotated_stages
    if expected_stages is None:
        assert "num_stages" not in loop.annotations
    else:
        assert int(loop.annotations["num_stages"]) == expected_stages
    assert annotations == original


if __name__ == "__main__":
    tilelang.testing.main()
