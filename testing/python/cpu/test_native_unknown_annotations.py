"""Generic user hints preserve body execution and native definition scope."""

import tilelang.language as T
import tilelang
import pytest
import torch
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.language._native import encode_annotations


def make_kernel(style, hint_key):
    @T.prim_func
    def kernel(A: T.Tensor((1,), "int32"), B: T.Tensor((3,), "int32")):
        with T.Kernel(1):
            B[0] = 0
            if style == "plain":
                B[0] = B[0] + 1
                value = A[0] + 7
                B[1] = value
            elif style == "attr":
                with T.attr(0, hint_key, 1):
                    B[0] = B[0] + 1
                    value = A[0] + 7
                    B[1] = value
            elif style == "sblock_before":
                with T.sblock("hint"):
                    T.sblock_attr({hint_key: 1})
                    B[0] = B[0] + 1
                    value = A[0] + 7
                    B[1] = value
            elif style == "sblock_after":
                with T.sblock("hint"):
                    B[0] = B[0] + 1
                    value = A[0] + 7
                    B[1] = value
                    T.sblock_attr({hint_key: 1})
            B[2] = B[0] + B[1]

    return kernel


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("style", ["attr", "sblock_before", "sblock_after"])
@pytest.mark.parametrize("hint_key", ["user_hint_152", "pragma_unknown_152", "code_block_unknown_152"])
def test_unknown_hints_execute_legal_body_once(target, style, hint_key):
    function = make_kernel(style, hint_key)
    assert tirx.analysis.check_well_formed(function)
    executable = tilelang.compile(function, target=target, execution_backend="cython" if target == "c" else "tvm_ffi")
    for value in (-4, 0, 9):
        a = torch.tensor([value], dtype=torch.int32)
        b = torch.full((3,), -123, dtype=torch.int32)
        executable(a, b)
        assert b.tolist() == [1, value + 7, value + 8]


def _attribute(key, value, body, node=None):
    schema, payload = encode_annotations({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _annotation(values):
    schema, payload = encode_annotations(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def _function(body, parameters=()):
    return tirx.Function(list(parameters), body, ir.PrimType("void")).with_attr("target", tvm_tirx.target.Target("llvm"))


def _finalize(function):
    assert tirx.analysis.check_well_formed(function)
    return tilelang.transform.FinalizeAnnotations("llvm")(ir.IRModule({"main": function}))["main"]


@pytest.mark.parametrize("key", ["user_hint_152", "pragma_unknown_152", "code_block_unknown_152"])
def test_unknown_attribute_preserves_adjacent_definition_scopes(key):
    left, right = ir.Var("local_value", "int32"), ir.Var("local_value", "int32")
    body = [
        _attribute(key, 1, [ir.Bind(left, ir.const(7)), ir.Evaluate(left + 1)]),
        _attribute(key, 1, [ir.Bind(right, ir.const(9)), ir.Evaluate(right + 2)]),
    ]
    result = _finalize(_function(body))
    assert tirx.analysis.check_well_formed(result)
    assert tirx.analysis.check_well_formed(tirx.renew_def(result))
    assert len(result.body.seq) == 2
    for scope, local in zip(result.body.seq, (left, right)):
        assert scope.op.name == "tl.lexical_scope"
        assert not scope.args and not scope.body_params and not scope.result_vars
        assert scope.body.seq[0].var.same_as(local)
    assert not result.body.seq[0].body.seq[0].var.same_as(result.body.seq[1].body.seq[0].var)


@pytest.mark.parametrize("carrier", ["region", "annotation"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("barrier_init", {}),
        ("l2_hit_ratio_map", {}),
        ("metal_context", ""),
        ("safe_value_map", {}),
        ("layout_map", {}),
        ("kWarpSpecializationScope", 0),
        ("kCustomWarpSpecialization", 1),
        ("cluster_dims", [2, 1, 1]),
        ("use_2cta", True),
        ("tl.unconsumed_contract", 1),
        ("tirx.unconsumed_contract", 1),
    ],
)
def test_known_and_internal_contracts_still_require_their_owning_pass(carrier, key, value):
    statement = _attribute(key, value, []) if carrier == "region" else _annotation({key: value})
    with pytest.raises(ValueError, match="Unconsumed|must be consumed|cannot be discarded"):
        _finalize(_function([statement]))


@pytest.mark.parametrize(
    "key,value",
    [
        ("pragma_import_c", "// test source"),
        ("pragma_unroll_factor", 4),
        ("code_block_source", "// test source"),
        ("code_block_signature", []),
        ("code_block_entry_name", "test_entry"),
        ("volatile_scope", 1),
        ("fragment_shape", "16,16,16"),
        ("fragment_layout", "row_major"),
        ("threadblock_swizzle_pattern", ["rasterization2DRow", 2]),
        ("tl.min_blocks_per_sm", 2),
    ],
)
def test_real_codegen_keys_are_preserved_by_finalizer(key, value):
    # These are transport checks; individual consumers retain their own value
    # validation and target constraints. Do not declare a backend compile here.
    original = _attribute(key, value, [])
    result = _finalize(_function([original]))
    region = result.body.seq[0]
    assert region.op.name == "tl.attr"
    assert tvm_ffi.structural_equal(region.args, original.args)
    assert tvm_ffi.structural_equal(region.attrs, original.attrs)


def test_known_import_c_keeps_body_and_local_bind_execution():
    @T.prim_func
    def kernel(A: T.Tensor((1,), "int32"), B: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            with T.attr(0, "pragma_import_c", "// user prelude for legal scope test"):
                value = A[0] + 7
                B[0] = value
            B[1] = B[0] + 1

    assert tirx.analysis.check_well_formed(kernel)
    executable = tilelang.compile(kernel, target="c", execution_backend="cython")
    assert "// user prelude for legal scope test" in executable.get_kernel_source()
    a, b = torch.tensor([9], dtype=torch.int32), torch.full((2,), -123, dtype=torch.int32)
    executable(a, b)
    assert b.tolist() == [16, 17]
