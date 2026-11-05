"""The production WMMA metadata pass retains the old whole-function contract.

The eight ordering cases follow s_tir/transform/tensorcore_infer_fragment.cc
at TVM 4211874e9e4b8770fd20ad0d98ab80afd1df5028.  They test metadata consistency,
not initialization dominance or GPU instruction execution.
"""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx import op

import tilelang
import tilelang.language as T


CASES = [
    "before",
    "after",
    "store_only_d",
    "missing_c",
    "fill_a_then_column_load",
    "column_load_a_then_fill",
    "mismatch_shape",
    "bmma_before",
]
ERRORS = {
    "missing_c": "must be filled",
    "fill_a_then_column_load": "Inconsistent WMMA fragment layout",
    "mismatch_shape": "Inconsistent WMMA fragment shape",
}


def _make_function(case, *, same_names=False, alias=False):
    names = ["fragment"] * 4 if same_names else ["a", "b", "c", "d"]
    specs = [("float16", "wmma.matrix_a"), ("float16", "wmma.matrix_b"), ("float32", "wmma.accumulator"), ("float32", "wmma.accumulator")]
    fragments = [ir.Var(name, tirx.TensorType([256], dtype, scope=scope)) for name, (dtype, scope) in zip(names, specs)]
    a, b, c, d = fragments
    source = ir.Var("source", ir.PtrType("float16"))
    destination = ir.Var("destination", ir.PtrType("float32"))
    allocations = [ir.Bind(var, ir.Call("tirx.alloc_tensor", [], ty_args=[var.ty])) for var in fragments]
    a_load = a
    if alias:
        a_load = ir.Var("a_view", a.ty)
        allocations.append(ir.Bind(a_load, ir.Call("tirx.decl_tensor", [op.tensor_data_ptr(a)], ty_args=[a.ty])))

    def fill(var):
        return ir.Evaluate(T.tvm_fill_fragment(var, 16, 16, 16, 0, 0))

    def load(var, layout="row_major", m=16):
        return ir.Evaluate(T.tvm_load_matrix_sync(var, m, 16, 16, 0, source, 16, layout))

    def store(var):
        return ir.Evaluate(T.tvm_store_matrix_sync(var, 16, 16, 16, 0, destination, 16, "row_major"))

    mma_call = T.tvm_bmma_sync if case == "bmma_before" else T.tvm_mma_sync
    mma = ir.Evaluate(mma_call(d, 0, a, 0, b, 0, c, 0))
    initializers = [load(a_load), load(b), fill(c), fill(d)]
    if case == "after":
        statements = [mma, *initializers]
    elif case == "store_only_d":
        statements = [*initializers[:3], mma, store(d)]
    elif case == "missing_c":
        statements = [initializers[0], initializers[1], initializers[3], mma]
    elif case == "fill_a_then_column_load":
        statements = [fill(a), load(a_load, "column_major"), *initializers[1:], mma]
    elif case == "column_load_a_then_fill":
        statements = [load(a_load, "column_major"), fill(a), *initializers[1:], mma]
    elif case == "mismatch_shape":
        statements = [load(a_load), load(b, m=8), *initializers[2:], mma]
    else:
        statements = [*initializers, mma]
    return tirx.Function([source, destination], allocations + statements, ret_type=ir.PrimType("void"))


def _infer(function):
    factory = tvm_ffi.get_global_func("tl.transform.InferWmmaFragmentMetadata")
    return factory()(ir.IRModule({"main": function}))["main"]


def _assert_metadata(function, case):
    for index, bind in enumerate(list(function.body.seq)[:4]):
        assert isinstance(bind, ir.Bind) and bind.value.op.name == "tirx.alloc_tensor"
        assert str(bind.value.attrs["tl.wmma_shape"]) == "16, 16, 16"
        if index == 3 and case == "store_only_d":
            assert "tl.wmma_layout" not in bind.value.attrs
        else:
            expected = "column_major" if index == 0 and case == "column_load_a_then_fill" else "row_major"
            assert str(bind.value.attrs["tl.wmma_layout"]) == expected


@pytest.mark.parametrize("case", CASES)
def test_production_wmma_metadata_order(case, tmp_path):
    before = _make_function(case)
    assert tirx.analysis.check_well_formed(before)
    before_json = ir.save_json(before)
    (tmp_path / "before.json").write_text(before_json)
    if case in ERRORS:
        with pytest.raises(tilelang.errors.InternalError, match=ERRORS[case]):
            _infer(before)
        assert ir.save_json(before) == before_json
        return
    after = _infer(before)
    assert tirx.analysis.check_well_formed(after)
    _assert_metadata(after, case)
    assert ir.save_json(before) == before_json, "Copy-on-write must preserve the shared input"
    (tmp_path / "after.json").write_text(ir.save_json(after))
    again = _infer(after)
    ir.assert_structural_equal(after, again)
    restored = tirx.renew_def(ir.load_json(ir.save_json(after)))
    assert tirx.analysis.check_well_formed(restored)
    ir.assert_structural_equal(after, restored)
    _assert_metadata(restored, case)
    assert not restored.body.seq[0].var.same_as(after.body.seq[0].var)
    ir.assert_structural_equal(restored, _infer(restored))


@pytest.mark.parametrize("case", ["before", "missing_c"])
def test_same_name_fragments_keep_distinct_identity(case):
    before = _make_function(case, same_names=True)
    assert tirx.analysis.check_well_formed(before)
    assert all(not before.body.seq[0].var.same_as(before.body.seq[i].var) for i in (1, 2, 3))
    if case in ERRORS:
        with pytest.raises(tilelang.errors.InternalError, match=ERRORS[case]):
            _infer(before)
    else:
        _assert_metadata(_infer(before), case)


def test_alias_metadata_uses_the_physical_storage_root():
    before = _make_function("before", alias=True)
    assert tirx.analysis.check_well_formed(before)
    after = _infer(before)
    _assert_metadata(after, "before")
    # The alias declaration has no second allocation or independent metadata.
    alias = after.body.seq[4]
    assert alias.value.op.name == "tirx.decl_tensor"
    assert alias.value.attrs is None
    restored = tirx.renew_def(ir.load_json(ir.save_json(after)))
    ir.assert_structural_equal(restored, _infer(restored))


def test_no_wmma_operations_preserve_the_function():
    function = tirx.Function([], [ir.Evaluate(ir.const(0, "int32"))], ir.PrimType("void"))
    result = _infer(function)
    ir.assert_structural_equal(function, result)
