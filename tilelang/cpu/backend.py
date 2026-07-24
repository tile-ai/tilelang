from __future__ import annotations

from tilelang.backend.module import BackendModule


cpu_backend_module = BackendModule("cpu", ("c", "llvm"))
cpu_backend_module.register_execution_backend()
cpu_backend_module.register_device_codegen()
cpu_backend_module.register_host_module()
