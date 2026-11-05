"""Runtime registrations share the provider FFI and preserve error details."""

import pytest
import tvm_ffi


@pytest.mark.parametrize(
    "name,args,message",
    [
        ("dtype_mismatch", ("kernel", "A", 0, 32, 1, 2, 32, 1), "dtype expected float32, but got int32"),
        ("ndim_mismatch", ("kernel", "A", 2, 1), "ndim expected 2, but got 1"),
        ("byte_offset_mismatch", ("kernel", "A", 0, 4), "byte_offset expected 0, but got 4"),
        ("device_type_mismatch", ("kernel", "A", 1, 2), "device_type expected cpu, but got cuda"),
        ("null_ptr", ("kernel", "A", "data"), "data expected non-NULL, but got NULL"),
        ("expect_eq", ("kernel", "A", "shape", 5, 6), "shape expected 5, but got 6"),
        ("constraint_violation", ("kernel", "A", "shape", "divisible by two"), "constraint not satisfied: divisible by two"),
    ],
)
def test_native_runtime_errors_keep_fields(name, args, message):
    with pytest.raises(RuntimeError, match=message):
        tvm_ffi.get_global_func("__tvm_error_" + name)(*args)


def test_cuda_helper_registry_is_complete_when_backend_is_built():
    if tvm_ffi.get_global_func("ffi.Module.create.cuda", allow_missing=True) is None:
        pytest.skip("CUDA provider runtime is not built")
    for name in (
        "__tvm_tensormap_create_tiled",
        "__tvm_tensormap_create_im2col",
        "__tvm_cuda_stream_set_access_policy_window",
        "__tvm_cuda_stream_reset_access_policy_window",
    ):
        assert tvm_ffi.get_global_func(name, allow_missing=True) is not None, name
