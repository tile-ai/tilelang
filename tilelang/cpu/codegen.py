from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.backend.device_codegen import global_func_device_codegen

build_c = global_func_device_codegen("target.build.tilelang_c")
build_llvm = global_func_device_codegen("target.build.llvm")


def include_cpu_types_in_host(mod: IRModule, target_host: Target, target: Target) -> IRModule:
    """Give CPU C host declarations the same half/vector types as device code."""
    if target_host.kind.name != "c":
        return mod
    result = mod.clone()
    for global_var, func in mod.functions.items():
        body = tirx.AttrStmt(0, "pragma_import_c", "#include <tl_templates/cpp/common.h>\n", func.body)
        result.update_func(global_var, func.with_body(body))
    return result
