"""Private native functions retain visibility across both frontend phases."""

import importlib

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T


def test_unadorned_prim_func_remains_public():
    @T.prim_func
    def helper(a: T.Buffer((1,), "float32")):
        a[0] = a[0]

    assert helper.attrs["global_symbol"] == "helper"
    assert tirx.analysis.check_well_formed(helper)


@pytest.mark.parametrize("namespace", ["tilelang.language", "tilelang.cuda.language", "tilelang.ascend.language"])
@pytest.mark.parametrize("private", [False, True])
def test_prim_func_private_controls_the_native_export_symbol(namespace, private):
    language = importlib.import_module(namespace)

    @language.prim_func(private=private)
    def helper(a: language.Buffer((1,), "float32")):
        a[0] = a[0]

    assert isinstance(helper, tirx.Function)
    assert ("global_symbol" not in helper.attrs) is private
    if not private:
        assert helper.attrs["global_symbol"] == "helper"
    assert tirx.analysis.check_well_formed(helper)
    module = ir.IRModule.from_expr(helper)
    assert module["main" if private else "helper"].same_as(helper)


def test_same_name_private_helpers_have_independent_module_entries():
    def make_helper(amount):
        @T.prim_func(private=True)
        def helper(a: T.Buffer((1,), "float32")):
            a[0] = a[0] + amount

        return helper

    first, second = make_helper(1.0), make_helper(2.0)
    module = ir.IRModule({"first": first, "second": second})
    assert "global_symbol" not in module["first"].attrs
    assert "global_symbol" not in module["second"].attrs
    assert not tvm_ffi.structural_equal(module["first"], module["second"])
    assert tirx.analysis.check_well_formed(module)


@pytest.mark.parametrize("private", [False, True])
def test_eager_template_phase1_and_phase2_preserve_private(private):
    @T.prim_func(eager_jit=True, private=private)
    def helper(a: T.Buffer((8,), "float32")):
        for i in T.serial(8):
            a[i] = a[i] + 1.0

    helper.set_mode("eager")
    value = np.zeros(8, dtype=np.float32)
    phase2 = helper.get_tir(value)
    assert len(helper.p1_cache) == 1
    (template,) = helper.p1_cache.values()
    assert ("global_symbol" not in template.prim_func.attrs) is private
    assert ("global_symbol" not in phase2.attrs) is private
    assert tirx.analysis.check_well_formed(template.prim_func)
    assert tirx.analysis.check_well_formed(phase2)
    again = helper.get_tir(value)
    ir.assert_structural_equal(phase2, again)


@pytest.mark.parametrize("private", [False, True])
def test_lazy_template_private_applies_to_the_returned_function(private):
    @T.prim_func(eager_jit=True, private=private)
    def factory():
        @T.prim_func
        def helper(a: T.Buffer((1,), "float32")):
            a[0] = a[0] + 1.0

        return helper

    factory.set_mode("lazy")
    function = factory.get_tir()
    assert ("global_symbol" not in function.attrs) is private
    assert tirx.analysis.check_well_formed(function)
    assert factory.get_tir().same_as(function)


def test_private_function_rejects_an_explicit_export_attribute():
    with pytest.raises(Exception, match="A private function may not have"):

        @T.prim_func(private=True)
        def helper(a: T.Buffer((1,), "float32")):
            T.func_attr({"global_symbol": "explicit_export"})
            a[0] = a[0]
