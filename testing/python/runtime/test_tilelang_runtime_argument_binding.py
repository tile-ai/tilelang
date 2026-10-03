import pytest
import tilelang  # noqa: F401
import tvm_ffi


def _callee():
    tvm_ffi.register_global_func(
        "testing.tilelang.bound_function.sum",
        f=lambda first, second, third: first + second + third,
        override=True,
    )
    return tvm_ffi.get_global_func("testing.tilelang.bound_function.sum")


def test_native_binding_builds_a_complete_stable_argument_frame():
    callee = _callee()
    bind = tvm_ffi.get_global_func("tilelang.runtime.bind_packed_function")

    bound = bind(callee, 3, [0, 2], [10, 30], [1])

    assert bound(2) == 42
    assert bound(3) == 43


def test_native_binding_rejects_an_incomplete_or_overlapping_abi():
    callee = _callee()
    bind = tvm_ffi.get_global_func("tilelang.runtime.bind_packed_function")

    with pytest.raises(Exception, match="Check failed"):
        bind(callee, 3, [0], [10], [1])
    with pytest.raises(Exception, match="Check failed"):
        bind(callee, 3, [0], [10], [0, 1, 2])
