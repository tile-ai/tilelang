from __future__ import annotations

from tilelang.backend.module import BackendModule


cuda_backend_module = BackendModule("cuda")
cuda_backend_module.register_execution_backend()
cuda_backend_module.register_device_codegen()
