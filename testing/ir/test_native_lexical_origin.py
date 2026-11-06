"""Explicit allocation intent survives native scope lowering and persistence."""

import pytest
from tvm_tirx import ir, tirx

from tilelang import transform
from test_native_reducer_contracts import allocate, annotation, attribute, function, integer, objects, tensor


def scopes(value):
    return [node for node in objects(value, ir.RegionStmt) if node.op.name == "tl.lexical_scope"]


@pytest.mark.parametrize("carrier", ["attribute", "block_annotation", "statement_annotation"])
def test_explicit_allocation_origin_survives_each_carrier(carrier):
    scratch = tensor("scratch", [4])
    body = [allocate(scratch), ir.TensorStore(scratch, [integer(0)], ir.const(7, "float32"))]
    if carrier == "attribute":
        body = [attribute(None, "lexical_alloc_scope", 1, body)]
    elif carrier == "block_annotation":
        body = [attribute(None, "tl.sblock", {"name": "scratch", "annotations": {"lexical_alloc_scope": 1}}, body)]
    else:
        body = [annotation({"lexical_alloc_scope": 1}), *body]
    original = function([], body)
    lowered = transform.LowerSourceAnnotations()(ir.IRModule({"main": original}))["main"]
    assert tirx.analysis.check_well_formed(lowered)
    marked = [node for node in scopes(lowered) if node.attrs.get("explicit_alloc_scope", False)]
    assert len(marked) == 1
    assert any(bind.var.same_as(scratch) for bind in objects(marked[0].body, ir.Bind))
    restored = ir.load_json(ir.save_json(lowered))
    assert len([node for node in scopes(restored) if node.attrs.get("explicit_alloc_scope", False)]) == 1
    ir.assert_structural_equal(lowered, restored)
    repeated = transform.LowerSourceAnnotations()(ir.IRModule({"main": lowered}))["main"]
    ir.assert_structural_equal(lowered, repeated)


def test_unmarked_block_keeps_definition_scope_without_inventing_explicit_origin():
    scratch = tensor("scratch", [4])
    body = [attribute(None, "tl.sblock", {"name": "ordinary"}, [allocate(scratch)])]
    original = function([], body)
    lowered = transform.LowerSourceAnnotations()(ir.IRModule({"main": original}))["main"]
    (scope,) = scopes(lowered)
    assert "explicit_alloc_scope" not in scope.attrs
    assert tirx.analysis.check_well_formed(lowered)
    invalid = lowered.with_body([scope, ir.Evaluate(scratch)])
    assert not tirx.analysis.check_well_formed(invalid, assert_mode=False)


@pytest.mark.parametrize("flag", [False, True])
def test_flag_does_not_change_definition_lifetime(flag):
    scratch = tensor("scratch", [1])
    scope = ir.RegionStmt("tl.lexical_scope", [], [], {"explicit_alloc_scope": flag}, [allocate(scratch)])
    valid = function([], [scope])
    assert tirx.analysis.check_well_formed(valid)
    assert not tirx.analysis.check_well_formed(valid.with_body([scope, ir.Evaluate(scratch)]), assert_mode=False)


@pytest.mark.parametrize("value", [0, 1, "yes", [], None])
def test_explicit_origin_rejects_non_boolean(value):
    with pytest.raises(TypeError, match="explicit_alloc_scope must be a boolean"):
        ir.RegionStmt("tl.lexical_scope", [], [], {"explicit_alloc_scope": value}, [])


def test_lexical_scope_rejects_unknown_metadata():
    with pytest.raises(ValueError, match="Unknown tl.lexical_scope attribute"):
        ir.RegionStmt("tl.lexical_scope", [], [], {"unknown": True}, [])


def test_other_builtin_regions_still_reject_attributes():
    with pytest.raises(ValueError, match="expects no results or attributes"):
        ir.RegionStmt("tirx.parallel_launch", [], [], {"explicit_alloc_scope": True}, [])
