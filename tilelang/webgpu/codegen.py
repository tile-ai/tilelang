from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


register_device_codegen(
    "webgpu",
    DeviceCodegen(
        "webgpu",
        build_without_compile=global_func_device_codegen("target.build.webgpu"),
    ),
    override=True,
)
