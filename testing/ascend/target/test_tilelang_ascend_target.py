import pytest

from tilelang import tvm
from tilelang.backend import get_backend
from tilelang.backend.target import determine_target
from tilelang.ascend.target import (
    normalize_ascend_target,
    target_is_ascend,
)
from tvm.target import Target


def test_native_ascend_target_kind():
    assert "ascend" in Target.list_kinds()

    target = Target("ascend")
    assert target.kind.name == "ascend"
    assert list(target.keys) == ["ascend"]
    assert target.get_target_device_type() == 12
    assert target_is_ascend(target)


def test_ascend_normalizer_only_handles_shorthand():
    target = normalize_ascend_target("ascend")
    assert target is not None
    assert target.kind.name == "ascend"

    assert normalize_ascend_target(Target("ascend")) is None
    assert normalize_ascend_target({"kind": "ascend"}) is None


def test_ascend_target_options():
    target = Target({"kind": "ascend", "arch": "dav-3510"})
    assert target.kind.name == "ascend"
    assert str(target.attrs["arch"]) == "dav-3510"

    target = determine_target(
        {"kind": "ascend", "mcpu": "dav-3510"},
        return_object=True,
    )
    assert target.kind.name == "ascend"
    assert str(target.attrs["mcpu"]) == "dav-3510"


def test_target_kind_is_authoritative_for_ascend():
    with pytest.raises(AssertionError, match="Target c -keys=ascend is not supported"):
        determine_target("c -keys=ascend", return_object=True)

    c_target = Target({"kind": "c", "keys": ["ascend"]})
    assert not target_is_ascend(c_target)
    assert get_backend("cpu").get_pipeline(c_target).name == "c"
    assert get_backend("cpu").get_device_codegen(c_target).name == "c"


def test_ascend_backend_registries_use_native_kind():
    target = Target("ascend")
    assert get_backend("ascend").get_device_codegen(target).name == "ascend"
    assert set(get_backend("ascend").allowed_execution_backends(target)) == {"cython", "tvm_ffi"}
    assert get_backend("ascend").resolve_execution_backend("auto", target).name == "tvm_ffi"
    assert tvm.ffi.get_global_func("target.build.ascend", allow_missing=True) is not None


def test_native_ascend_kind_drives_backend_identity():
    target = determine_target({"kind": "ascend", "keys": ["custom"]}, return_object=True)
    assert list(target.keys) == ["custom"]
    assert target_is_ascend(target)
    assert get_backend("ascend").get_pipeline(target).name == "ascend"
    assert get_backend("ascend").get_device_codegen(target).name == "ascend"
    assert get_backend("ascend").resolve_execution_backend("auto", target).name == "tvm_ffi"
