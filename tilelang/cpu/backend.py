"""CPU backend manifest."""

import warnings

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import HostCodegenHook, STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline


class CPUBackendModule(BackendModule):
    def resolve_execution_backend(self, requested: str | None, target: Target) -> ExecutionBackendSpec:
        if target.kind.name == "c" and requested is not None and str(requested).lower() == "cython":
            warnings.warn(
                "The CPU Cython execution backend is deprecated; using tvm_ffi instead. Use execution_backend='tvm_ffi' or 'auto'.",
                FutureWarning,
                stacklevel=2,
            )
            requested = "tvm_ffi"
        return super().resolve_execution_backend(requested, target)


BACKEND = register_backend(
    CPUBackendModule(
        name="cpu",
        target_kinds=("c", "llvm"),
        pipelines={
            "c": PassPipeline("c", pipeline.CPUPassPipelineBody),
            "llvm": PassPipeline("llvm", pipeline.CPUPassPipelineBody),
        },
        device_codegens={
            "c": DeviceCodegen("c", build_without_compile=codegen.build_c),
            "llvm": DeviceCodegen(
                "llvm",
                build=codegen.build_llvm,
                build_without_compile=codegen.build_llvm,
            ),
        },
        host_codegens=STANDARD_HOST_CODEGENS,
        host_codegen_hooks={"c": (HostCodegenHook("cpu_types", codegen.include_cpu_types_in_host),)},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
