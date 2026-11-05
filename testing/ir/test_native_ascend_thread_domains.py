"""SIMT footprints quantify each launch's threads without inventing dense writes."""

import itertools

import pytest
import tilelang
import tvm_ffi
import tvm_tirx
from tilelang.ascend import language as T
from tvm_tirx import ir, tirx


def _tensor(name, shape):
    return ir.Var(name, tirx.TensorType(shape, "uint32", scope="global"))


def _launch(variable, extent, body, tag="threadIdx.x"):
    if isinstance(extent, int):
        extent = ir.const(extent, str(variable.ty.dtype))
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), extent], [variable], {}, body)


def _memory(body, *, physical=True):
    statement = ir.SeqStmt(body)
    function = tirx.Function(tirx.analysis.undefined_vars(statement), [statement])
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    original = ir.save_json(statement)
    result = tvm_ffi.get_global_func("tl.testing.AnalyzeAscendMemory")(statement, ir.SeqStmt([]), physical)
    assert ir.save_json(statement) == original
    return result


def _points(region, bindings=()):
    analyzer = tvm_tirx.sym.Analyzer()

    def value(expression):
        if bindings:
            variables, values = zip(*bindings)
            expression = tirx.IndexMap(list(variables), [expression], None).map_indices(list(values))[0]
        return int(analyzer.simplify(expression))

    return set(itertools.product(*(range(value(axis.begin), value(axis.begin + axis.extent)) for axis in region.region)))


def _region(result, kind, tensor):
    matches = [item for item in result[kind] if item.source.same_as(tensor)]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("width", [1, 4, 128])
@pytest.mark.parametrize("sid", [0, 1])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_memory_regions_cover_all_independent_threads(physical, width, sid, dtype):
    source, output = _tensor("source", [128]), _tensor("output", [256])
    thread = ir.Var("thread", dtype)
    body = [_launch(thread, width, [ir.TensorStore(output, [thread + sid * width], ir.TensorLoad(source, [thread]))])]
    result = _memory(body, physical=physical)
    reads = {(tx,) for tx in range(width)}
    writes = {(tx + sid * width,) for tx in range(width)}
    assert _points(_region(result, "reads", source)) == reads
    assert _points(_region(result, "writes", output)) == writes
    if physical:
        assert _points(_region(result, "must_writes", output)) == writes
    else:
        assert not result["must_writes"]


@pytest.mark.parametrize("x_extent,y_extent", [(1, 1), (1, 3), (4, 1), (4, 3)])
def test_nested_thread_dimensions_have_distinct_variables(x_extent, y_extent):
    source, output = _tensor("source", [y_extent, x_extent]), _tensor("output", [x_extent * y_extent])
    x, y = ir.Var("thread", "int32"), ir.Var("thread", "int32")
    assert not x.same_as(y)
    store = ir.TensorStore(output, [x + x_extent * y], ir.TensorLoad(source, [y, x]))
    result = _memory([_launch(x, x_extent, [_launch(y, y_extent, [store], "threadIdx.y")])])
    assert _points(_region(result, "reads", source)) == set(itertools.product(range(y_extent), range(x_extent)))
    expected = {(tx + x_extent * ty,) for tx in range(x_extent) for ty in range(y_extent)}
    assert _points(_region(result, "writes", output)) == expected
    assert _points(_region(result, "must_writes", output)) == expected


def test_sibling_launches_restore_outer_domain_and_keep_variable_identity():
    first, second = _tensor("first", [8, 2]), _tensor("second", [8, 5])
    outer = ir.Var("thread", "int32")
    left, right = ir.Var("thread", "int32"), ir.Var("thread", "int32")
    body = [
        _launch(left, 2, [ir.TensorStore(first, [outer, left], ir.const(1, "uint32"))], "threadIdx.y"),
        _launch(right, 5, [ir.TensorStore(second, [outer, right], ir.const(2, "uint32"))], "threadIdx.y"),
    ]
    result = _memory([_launch(outer, 8, body)])
    for tensor, width in [(first, 2), (second, 5)]:
        expected = set(itertools.product(range(8), range(width)))
        assert _points(_region(result, "writes", tensor)) == expected
        assert _points(_region(result, "must_writes", tensor)) == expected


def test_symbolic_sibling_does_not_suppress_later_dense_write():
    first, second = _tensor("first", [16]), _tensor("second", [5])
    count = ir.Var("count", "int32")
    left, right = ir.Var("thread", "int32"), ir.Var("thread", "int32")
    result = _memory(
        [
            _launch(left, count, [ir.TensorStore(first, [left], ir.const(1, "uint32"))]),
            _launch(right, 5, [ir.TensorStore(second, [right], ir.const(2, "uint32"))]),
        ]
    )
    assert len(result["must_writes"]) == 1
    assert _points(_region(result, "must_writes", second)) == {(tx,) for tx in range(5)}


@pytest.mark.parametrize("extent", [0, -1])
def test_empty_launch_has_no_body_memory_effect(extent):
    source, output = _tensor("source", [4]), _tensor("output", [4])
    thread = ir.Var("thread", "int32")
    result = _memory([_launch(thread, extent, [ir.TensorStore(output, [thread], ir.TensorLoad(source, [thread]))])])
    assert not result["reads"] and not result["writes"] and not result["must_writes"]


def test_launch_extent_resolves_a_dominating_let():
    output = _tensor("output", [8])
    count, thread = ir.Var("count", "int32"), ir.Var("thread", "int32")
    result = _memory(
        [
            ir.Bind(count, ir.const(4)),
            _launch(thread, count, [ir.TensorStore(output, [thread], ir.const(2, "uint32"))]),
        ]
    )
    assert _points(_region(result, "writes", output)) == {(tx,) for tx in range(4)}
    assert _points(_region(result, "must_writes", output)) == {(tx,) for tx in range(4)}


@pytest.mark.parametrize("thread_indexed", [False, True])
def test_unknown_extent_can_be_zero_and_has_no_must_write(thread_indexed):
    output = _tensor("output", [16])
    count, thread = ir.Var("count", "int32"), ir.Var("thread", "int32")
    index = thread if thread_indexed else ir.const(0)
    result = _memory([_launch(thread, count, [ir.TensorStore(output, [index], ir.const(2, "uint32"))])])
    assert not result["must_writes"]
    region = _region(result, "writes", output)
    for n in (0, 1, 5, 16):
        actual = {(tx if thread_indexed else 0,) for tx in range(n)}
        assert actual <= _points(region, [(count, ir.const(n))])


@pytest.mark.parametrize("thread_indexed", [False, True])
def test_extent_memory_read_is_visited_before_the_body(thread_indexed):
    count = ir.Var("count", tirx.TensorType([1], "int32", scope="global"))
    output = _tensor("output", [16])
    thread = ir.Var("thread", "int32")
    extent = ir.TensorLoad(count, [ir.const(0)])
    index = thread if thread_indexed else ir.const(0)
    result = _memory([_launch(thread, extent, [ir.TensorStore(output, [index], ir.const(2, "uint32"))])])
    assert _points(_region(result, "reads", count)) == {(0,)}
    assert not result["must_writes"]
    actual = {(tx if thread_indexed else 0,) for tx in range(16)}
    assert actual <= _points(_region(result, "writes", output))


@pytest.mark.parametrize("pattern", ["strided", "diagonal", "conditional"])
def test_must_write_does_not_fill_holes_or_unexecuted_lanes(pattern):
    output = _tensor("output", [8, 8] if pattern == "diagonal" else [16])
    thread, enabled = ir.Var("thread", "int32"), ir.Var("enabled", "bool")
    indices = [thread, thread] if pattern == "diagonal" else [thread * 2 if pattern == "strided" else thread]
    body = [ir.TensorStore(output, indices, ir.const(2, "uint32"))]
    if pattern == "conditional":
        body = [ir.If(enabled, body, None)]
    result = _memory([_launch(thread, 8, body)])
    actual = {(tx, tx) if pattern == "diagonal" else (tx * 2 if pattern == "strided" else tx,) for tx in range(8)}
    assert actual <= _points(_region(result, "writes", output))
    assert not result["must_writes"]


def test_inner_extent_dependent_on_outer_thread_is_not_cartesian():
    output = _tensor("output", [4, 4])
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
    inner = _launch(y, x + 1, [ir.TensorStore(output, [x, y], ir.const(2, "uint32"))], "threadIdx.y")
    result = _memory([_launch(x, 4, [inner])])
    assert not result["must_writes"]
    assert {(tx, ty) for tx in range(4) for ty in range(tx + 1)} <= _points(_region(result, "writes", output))


@pytest.mark.parametrize("width", [1, 128])
@pytest.mark.parametrize("gap", [-1, 0, 1])
def test_real_simt_lowering_keeps_disjoint_and_overlapping_subcores_distinct(width, gap):
    stride = width + gap

    @T.prim_func
    def main(out: T.Tensor((2 * width + 1,), "uint32"), scalar_out: T.Tensor((2,), "uint32")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            words[sid] = T.uint32(1)
            scalar_out[sid] = words[sid]
            with T.SimtVF(threads=width):
                tx = T.get_thread_binding()
                out[tx + sid * stride] = T.uint32(sid + 2)

    addresses = [{tx + sid * stride for tx in range(width)} for sid in (0, 1)]
    overlap = addresses[0] & addresses[1]
    original = ir.save_json(main)
    if gap < 0:
        assert overlap == {width - 1}
        with pytest.raises(RuntimeError, match="uncovered AIV0.V -> AIV1.V dependency at logical distance 0"):
            tilelang.lower(main, target="ascend", enable_device_compile=False)
    else:
        assert not overlap
        result = tilelang.lower(main, target="ascend", enable_device_compile=False)
        assert "volatile __ssbuf__" in result.kernel_source
        assert "__simt_vf__" in result.kernel_source
    assert ir.save_json(main) == original
