from __future__ import annotations

from tvm.target import Target

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, register_backend
from tilelang.backend.codegen import build_device_with_global_func
from tilelang.cpu.pipeline import CPUPassPipelineBody


def cpu_device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    if target.kind.name == "c":
        return build_device_with_global_func(device_mod, target, "target.build.tilelang_c")
    if target.kind.name == "llvm":
        return build_device_with_global_func(device_mod, target, "target.build.llvm")
    raise ValueError(f"Target {target.kind.name} is not supported by the CPU backend")


cpu_backend = Backend(
    name="cpu",
    target_kinds=("c", "llvm"),
    import_path="tilelang.cpu",
    pipeline=CPUPassPipelineBody,
    device_codegen_without_compile=cpu_device_codegen_without_compile,
    execution_backends={
        "cython": ExecutionBackendSpec(name="cython"),
        "tvm_ffi": ExecutionBackendSpec(
            name="tvm_ffi",
            enable_host_codegen=True,
            enable_device_compile=False,
        ),
    },
    default_execution_backend="cython",
    cmake_name="CPU",
)

register_backend(cpu_backend, override=True)
