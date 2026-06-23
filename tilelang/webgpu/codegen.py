from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


_build_webgpu = global_func_device_codegen("target.build.webgpu")


register_device_codegen(
    "webgpu",
    DeviceCodegen(
        "webgpu",
        build=_build_webgpu,
        build_without_compile=_build_webgpu,
    ),
    override=True,
)
