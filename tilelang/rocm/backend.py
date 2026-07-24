from __future__ import annotations

from tilelang.backend.module import BackendModule


rocm_backend_module = BackendModule("rocm", "hip")
rocm_backend_module.register_execution_backend()
rocm_backend_module.register_device_codegen()
