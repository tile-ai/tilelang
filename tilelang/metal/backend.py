from __future__ import annotations

from tilelang.backend.module import BackendModule


metal_backend_module = BackendModule("metal")
metal_backend_module.register_execution_backend()
metal_backend_module.register_device_codegen()
metal_backend_module.register_host_hooks_module()
