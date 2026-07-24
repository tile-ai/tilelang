from __future__ import annotations

from tilelang.backend.module import BackendModule


webgpu_backend_module = BackendModule("webgpu")
webgpu_backend_module.register_execution_backend("tilelang.backend.common")
webgpu_backend_module.register_device_codegen()
