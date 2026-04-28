from __future__ import annotations

from typing import Any

from tvm.target import Target

from tilelang.backend.base import LibraryCompileSpec
from tilelang.backend.common.execution import make_cython_execution_spec, make_tvm_ffi_execution_spec


HIP_DEFAULT_EXECUTION_BACKEND = "tvm_ffi"


def _hip_source_wrapper(**kwargs):
    from tilelang.jit.adapter.wrapper import TLHIPSourceWrapper

    return TLHIPSourceWrapper(**kwargs)


def _hip_library_command(target: Target, source_path: str, library_path: str, pass_configs: dict[str, Any]) -> list[str]:
    from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
    from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    rocm_path = find_rocm_path()
    arch = get_rocm_arch(rocm_path)
    return [
        "hipcc",
        "-std=c++17",
        "-fPIC",
        f"--offload-arch={arch}",
        "--shared",
        source_path,
        "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
        "-I" + TILELANG_TEMPLATE_PATH,
        "-o",
        library_path,
    ]


HIP_LIBRARY_COMPILE_SPEC = LibraryCompileSpec(
    source_suffix=".cpp",
    library_suffix=".so",
    command_factory=_hip_library_command,
)

HIP_EXECUTION_SPECS = (
    make_tvm_ffi_execution_spec(),
    make_cython_execution_spec(
        c_source_wrapper_factory=_hip_source_wrapper,
        library_compile_spec=HIP_LIBRARY_COMPILE_SPEC,
    ),
)
HIP_EXECUTION_BACKENDS = tuple(spec.name for spec in HIP_EXECUTION_SPECS)
