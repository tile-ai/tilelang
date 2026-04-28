from __future__ import annotations

from dataclasses import replace

from tilelang.backend.base import ExecutionBackendSpec
from tilelang.backend.common.execution import make_tvm_ffi_execution_spec


METAL_DEFAULT_EXECUTION_BACKEND = "tvm_ffi"


def _metal_source_wrapper(**kwargs):
    from tilelang.jit.adapter.wrapper import TLMetalSourceWrapper

    return TLMetalSourceWrapper(**kwargs)


def _create_torch_metal_adapter(**kwargs):
    from tilelang.jit.adapter.torch import MetalKernelAdapter

    artifact = kwargs.pop("artifact")
    return MetalKernelAdapter(
        params=kwargs["params"],
        result_idx=kwargs["result_idx"],
        func_or_mod=kwargs["func_or_mod"],
        device_mod=artifact.device_mod,
        kernel_global_source=artifact.kernel_source,
        verbose=kwargs["verbose"],
    )


def _create_torch_cache():
    from tilelang.jit.adapter.torch.kernel_cache import TorchKernelCache

    return TorchKernelCache()


METAL_TVM_FFI_SPEC = make_tvm_ffi_execution_spec()
METAL_TVM_FFI_SPEC = replace(METAL_TVM_FFI_SPEC, c_source_wrapper_factory=_metal_source_wrapper)

METAL_EXECUTION_SPECS = (
    METAL_TVM_FFI_SPEC,
    ExecutionBackendSpec(
        name="torch",
        adapter_factory=_create_torch_metal_adapter,
        database_adapter_factory=None,
        cache_factory=_create_torch_cache,
        kernel_source_from_adapter=False,
        host_source_from_adapter=False,
    ),
)
METAL_EXECUTION_BACKENDS = tuple(spec.name for spec in METAL_EXECUTION_SPECS)
