"""Source block declarations retain their construction and consumer contracts."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.language._annotation_codec import emit_annotation as _emit_annotation
from tilelang.language._annotation_codec import encode_annotations


def _execute(function, padding, *extra):
    values = np.arange(4, dtype="int32") + 20
    expected = np.stack([np.concatenate([values, np.full(4, pad, dtype="int32")]) for pad in padding])
    assert tirx.analysis.check_well_formed(function)
    executable = tilelang.compile(function, target="c", execution_backend="tvm_ffi")
    output = tvm_tirx.runtime.tensor(np.zeros(expected.shape, dtype="int32"))
    executable(tvm_tirx.runtime.tensor(values), output, *extra)
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("position", ["before", "after", "empty_loop", "runtime_branch"])
@pytest.mark.parametrize("flag", [0, 1])
def test_source_safe_value_applies_to_complete_owner(position, flag):
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((2, 8), "int32"), condition: T.int32):
        with T.Kernel(1):
            if position == "before":
                T.annotate_safe_value({A: 9})
            T.copy(A[0:8], B[0, 0:8])
            if position == "after":
                T.annotate_safe_value({A: 9})
            if position == "empty_loop":
                for i in T.serial(0):  # noqa: B007 - the empty owner is the test input
                    T.annotate_safe_value({A: 9})
            if position == "runtime_branch":  # noqa: SIM102 - keep compile-time and runtime guards separate
                if condition > 0:
                    T.annotate_safe_value({A: 9})
            T.copy(A[0:8], B[1, 0:8])

    _execute(kernel, [9, 9], flag)


def test_adjacent_and_nested_source_owners_do_not_leak_safe_values():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4, 8), "int32")):
        with T.Kernel(1):
            T.copy(A[0:8], B[0, 0:8])
            with T.sblock("inner"):
                T.copy(A[0:8], B[1, 0:8])
                T.annotate_safe_value({A: 13})
            T.copy(A[0:8], B[2, 0:8])
            T.annotate_safe_value({A: 9})
        with T.Kernel(1):
            T.copy(A[0:8], B[3, 0:8])

    _execute(kernel, [9, 13, 9, 0])


def test_native_unmarked_annotations_keep_sequence_semantics():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((2, 8), "int32")):
        with T.Kernel(1):
            T.copy(A[0:8], B[0, 0:8])
            _emit_annotation({"safe_value_map": {A: 9}})
            T.copy(A[0:8], B[1, 0:8])

    _execute(kernel, [0, 9])


def _annotation(values):
    schema, payload = encode_annotations(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def _scope(key, body):
    schema, payload = encode_annotations({"node": None, "key": key, "value": {}})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _function(body):
    return tirx.Function([], body, ir.PrimType("void")).with_attr("target", tvm_tirx.target.Target("llvm"))


def test_collector_merges_maps_without_hoisting_or_crossing_source_owners():
    first = _annotation({"tl.source_block_annotation": True, "mapping": {"x": {"a": 1}}})
    second = _annotation({"tl.source_block_annotation": True, "mapping": {"x": {"b": 2}}})
    inner = _annotation({"tl.source_block_annotation": True, "mapping": {"x": {"forbidden": 3}}})
    i = ir.Var("i", "int32")
    helper = tirx.Function([], _scope("tl.sblock", inner), ir.PrimType("void"))
    deferred = ir.LambdaExpr([], lambda: helper)
    body = ir.SeqStmt(
        [
            first,
            ir.For(i, 0, 1, 1, ir.ForKind.DEFAULT, second),
            _scope("tl.sblock", inner),
            _scope("tl.kernel", inner),
            ir.Bind(ir.Var("helper", helper.ty), helper),
            ir.Bind(ir.Var("deferred", deferred.ty), deferred),
            _annotation({"unmarked": 4}),
        ]
    )
    before = ir.save_json(body)
    result = tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")(body)
    assert set(result) == {"mapping"}
    assert dict(result["mapping"]["x"]) == {"a": 1, "b": 2}
    assert ir.save_json(body) == before
    assert body.seq[0].same_as(first)
    assert body.seq[1].body.seq[0].same_as(second)


@pytest.mark.parametrize("marker", [False, 1, "true"])
def test_invalid_source_origin_marker_is_rejected(marker):
    body = ir.SeqStmt([_annotation({"tl.source_block_annotation": marker})])
    with pytest.raises(ValueError, match="source annotation marker"):
        tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")(body)
    function = _function(_scope("tl.kernel", body))
    with pytest.raises(ValueError, match="source annotation marker"):
        tilelang.transform.LowerSourceAnnotations()(ir.IRModule({"main": function}))


def test_finalizer_consumes_only_the_internal_origin_marker():
    body = _scope("tl.kernel", _annotation({"tl.source_block_annotation": True}))
    function = _function(body)
    finalized = tilelang.transform.LowerSourceAnnotations()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(finalized)
    assert len(finalized.body.seq) == 1
    assert finalized.body.seq[0].op.name == "tl.lexical_scope"
    assert not finalized.body.seq[0].body.seq
    # Ordinary user hints have no consumer; a reserved internal contract must
    # still fail after the source-origin marker itself has been consumed.
    unknown = _scope("tl.kernel", _annotation({"tl.source_block_annotation": True, "tl.unconsumed_source_block_contract": 1}))
    pending = tilelang.transform.LowerSourceAnnotations()(ir.IRModule({"main": _function(unknown)}))
    with pytest.raises(ValueError, match="Unconsumed TileLang call"):
        tilelang.transform.VerifyLoweredAnnotations()(pending)
