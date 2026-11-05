import itertools

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang.language as language
import tilelang.testing
from tilelang.testing.ir import collect_nodes


def _make_persistent(domain, wave_size, **options):
    worker = ir.Var("worker", ir.PrimType("int32"))
    with IRBuilder() as builder, language.Persistent(domain, wave_size, worker, **options) as coordinates:
        if len(domain) == 1:
            coordinates = [coordinates]
        for coordinate in coordinates:
            language.evaluate(coordinate)

    loops = collect_nodes(builder.get(), ir.For)
    assert len(loops) == 1
    loop = loops[0]
    return loop, worker, loop.body


def _trace_persistent(loop, worker, body, wave_size, parameters):
    analyzer = sym.Analyzer()

    def evaluate(expression, values):
        replaced = tvm_ffi.structural_map(expression, (ir.Var, lambda var: ir.const(values[var], var.ty.dtype) if var in values else var))
        result = analyzer.simplify(replaced)
        assert isinstance(result, ir.IntImm)
        return int(result)

    def execute(statement, values, coordinates):
        if isinstance(statement, ir.SeqStmt):
            return any(execute(child, values, coordinates) for child in statement.seq)
        if isinstance(statement, ir.Bind):
            values[statement.var] = evaluate(statement.value, values)
            return False
        if isinstance(statement, ir.If):
            branch = statement.then_case if evaluate(statement.condition, values) else statement.else_case
            return execute(branch, values, coordinates) if branch is not None else False
        assert isinstance(statement, ir.Evaluate)
        if isinstance(statement.value, ir.Call):
            assert statement.value.op.name == "tl.loop_break"
            return True
        coordinates.append(evaluate(statement.value, values))
        return False

    events = []
    for worker_index in range(wave_size):
        for wave_index in range(evaluate(loop.end - loop.begin, parameters)):
            values = {**parameters, worker: worker_index, loop.var: wave_index}
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
        extents = [ir.Var(f"extent_{axis}", ir.PrimType("int32")) for axis in range(len(domain))]
        workers = ir.Var("workers", ir.PrimType("int32"))
        parameters = dict(zip(extents, domain))
        parameters[workers] = wave_size
        if group_size is not None:
            group = ir.Var("group", ir.PrimType("int32"))
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


@pytest.mark.parametrize("wave_size", [1, 5, 128, None])
def test_persistent_guards_body_without_loop_break(wave_size):
    workers = ir.Var("workers", ir.PrimType("int32")) if wave_size is None else wave_size
    _, _, body = _make_persistent([2, 3, 10], workers, group_size=3, num_stages=2)

    assert not collect_nodes(body, ir.Break)
    assert all(node.op.name != "tl.loop_break" for node in collect_nodes(body, ir.Call))
    assert isinstance(body.seq[-1], ir.If)
    assert body.seq[-1].else_case is None


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


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_persistent_coordinates_are_defined_inside_the_wave_loop(index_dtype):
    extent = ir.Var("extent", ir.PrimType(index_dtype))

    @language.prim_func
    def kernel(A: language.Tensor((extent, 10), "int32"), worker: language.int32):
        for row, column in language.Persistent([extent, 10], 3, worker, group_size=3):
            A[row, column] = row + column

    assert tirx.analysis.check_well_formed(kernel)
    assert not tirx.analysis.undefined_vars(kernel)
    loops = collect_nodes(kernel, ir.For)
    assert len(loops) == 1
    binds = collect_nodes(loops[0].body, ir.Bind)
    assert len(binds) >= 2
    assert str(binds[0].var.ty.dtype) == index_dtype
    assert str(binds[1].var.ty.dtype) == "int32"


@pytest.mark.parametrize("domain", [(0,), (0, 10), (4, 0)])
def test_persistent_empty_domain_has_no_work(domain):
    loop, worker, body = _make_persistent(list(domain), 3)
    assert _trace_persistent(loop, worker, body, 3, {}) == []


@pytest.mark.parametrize("domain,wave_size", [([], 1), ([1], 0), ([1], -1), ([-1, 4], 2)])
def test_persistent_rejects_invalid_static_domains(domain, wave_size):
    with IRBuilder(), pytest.raises(ValueError):
        language.Persistent(domain, wave_size, 0)


if __name__ == "__main__":
    tilelang.testing.main()
