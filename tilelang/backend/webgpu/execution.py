from __future__ import annotations

from tilelang.backend.common.execution import make_cython_execution_spec, make_tvm_ffi_execution_spec


WEBGPU_DEFAULT_EXECUTION_BACKEND = "cython"

WEBGPU_EXECUTION_SPECS = (
    make_cython_execution_spec(
        c_source_wrapper_factory=None,
        library_compile_spec=None,
    ),
    make_tvm_ffi_execution_spec(),
)
WEBGPU_EXECUTION_BACKENDS = tuple(spec.name for spec in WEBGPU_EXECUTION_SPECS)
