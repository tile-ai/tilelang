"""Capability gates must not silently turn runtime tests into compile-only tests."""

import sys

import pytest
from tvm_tirx import runtime

import tilelang.testing as testing
from tilelang.testing.features import Feature, _test_targets


def skipped(marks):
    return any(mark.name == "skipif" and mark.args[0] for mark in marks)


def test_native_testing_has_no_old_compiler_import():
    assert "tvm" not in sys.modules
    assert testing.requires_cuda is Feature._all_features["cuda"]
    assert testing.requires_ascend is Feature._all_features["ascend"]


def test_compile_only_does_not_consult_runtime_or_device():
    def no_runtime_query():
        raise AssertionError("compile-only must not probe execution availability")

    feature = Feature("llvm", compile_time_check=lambda: True, run_time_check=no_runtime_query)
    previous = testing.requires_llvm
    try:
        assert not skipped(feature.marks("compile-only"))
        assert not skipped(feature.marks("optional"))
    finally:
        Feature._all_features["llvm"] = previous


def test_runtime_and_compile_support_are_both_required():
    feature = Feature("llvm", compile_time_check=lambda: True, run_time_check=lambda: False)
    try:
        assert skipped(feature.marks())
        assert not skipped(feature.marks("compile-only"))
        feature.compile_time_check = lambda: "SDK is unavailable"
        marks = feature.marks("compile-only")
        assert skipped(marks)
        assert any(mark.kwargs.get("reason") == "SDK is unavailable" for mark in marks)
    finally:
        Feature._all_features["llvm"] = testing.requires_llvm


def test_target_selection_keeps_json_kind_and_compile_only_filter(monkeypatch):
    monkeypatch.setenv("TVM_TEST_TARGETS", ' {"kind": "llvm", "mcpu": "generic"}; llvm; llvm ')
    assert _test_targets() == [{"kind": "llvm", "mcpu": "generic"}, "llvm"]
    assert (not skipped(testing.requires_llvm.marks("compile-only"))) == bool(runtime.enabled("llvm"))
    assert skipped(testing.requires_cuda.marks("compile-only"))
    assert not skipped(testing.requires_cuda.marks("optional"))


def test_parent_checks_and_markers_are_retained():
    feature = Feature("cuda", parent_features="gpu", compile_time_check=lambda: True)
    parent = Feature._all_features["gpu"]
    check = parent.run_time_check
    parent.run_time_check = lambda: False
    try:
        assert skipped(feature.marks())
        assert not skipped(feature.marks("compile-only"))
        assert {mark.name for mark in feature.marks("optional")} == {"gpu", "cuda"}
    finally:
        parent.run_time_check = check
        Feature._all_features["cuda"] = testing.requires_cuda


@pytest.mark.parametrize("level", ["runtime-only", "", None])
def test_unknown_support_level_is_an_error(level):
    with pytest.raises(ValueError, match="Unknown feature support type"):
        testing.requires_cuda(support_required=level)


def test_requires_package_checks_real_importability():
    @testing.requires_package("json")
    def installed():
        pass

    @testing.requires_package("tilelang_missing_test_dependency_43299")
    def unavailable():
        pass

    assert not skipped(installed.pytestmark)
    assert skipped(unavailable.pytestmark)
