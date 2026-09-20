from tilelang.ascend.kernel_cache import AscendCythonKernelCache, AscendTVMFFIKernelCache
from tilelang.cache import _dispatch_map, _resolve_cache_dispatch


def test_ascend_tvm_ffi_cache_uses_target_specific_source_suffixes():
    cache, context, _ = _resolve_cache_dispatch("ascend", None, "tvm_ffi", False)

    assert type(cache) is AscendTVMFFIKernelCache
    assert cache is not _dispatch_map["tvm_ffi"]
    assert context.module.name == "ascend"
    assert "ascend" in context.target.keys
    assert context.execution_backend.name == "tvm_ffi"
    assert cache.device_kernel_path == "device_kernel.asc"
    assert cache.host_kernel_path == "host_kernel.c"


def test_ascend_cython_cache_uses_bisheng_source_suffixes():
    cache, context, _ = _resolve_cache_dispatch("ascend", None, "cython", False)

    assert type(cache) is AscendCythonKernelCache
    assert cache is not _dispatch_map["cython"]
    assert context.module.name == "ascend"
    assert "ascend" in context.target.keys
    assert context.execution_backend.name == "cython"
    assert cache.device_kernel_path == "device_kernel.asc"
    assert cache.host_kernel_path == "host_kernel.asc"


def test_pto_cache_layout_is_unchanged():
    cache, context, _ = _resolve_cache_dispatch("pto", None, "cython", False)

    assert cache is _dispatch_map["cython"]
    assert context.module.name == "pto"
    assert "pto" in context.target.keys
    assert context.execution_backend.name == "cython"
    assert cache.device_kernel_path == "device_kernel.cu"
    assert cache.host_kernel_path == "host_kernel.cu"


def test_cuda_cache_source_suffixes_are_unchanged():
    cache, context, _ = _resolve_cache_dispatch({"kind": "cuda", "arch": "sm_90"}, None, "tvm_ffi", False)

    assert cache is _dispatch_map["tvm_ffi"]
    assert context.module.name == "cuda"
    assert context.execution_backend.name == "tvm_ffi"
    assert cache.device_kernel_path == "device_kernel.cu"
    assert cache.host_kernel_path == "host_kernel.cu"
