from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen

# Plain Ascend and PTO share the "ascend" target kind, so they are declared as
# two BackendModules (see backend.py) whose mutually exclusive `supports_target`
# predicates select one; DeviceCodegen itself no longer carries a predicate.
PTO_CODEGEN = DeviceCodegen(
    "pto",
    build=global_func_device_codegen("target.build.tilelang_pto"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_pto_without_compile"),
)

ASCEND_CODEGEN = DeviceCodegen(
    "ascend",
    build=global_func_device_codegen("target.build.tilelang_ascend"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_ascend_without_compile"),
)
