from __future__ import annotations

from tilelang.backend.module import BackendModule


def test_backend_module_registers_lazy_paths():
    target_kind = "unit-backend-module"
    package = "tilelang.unit_backend_module"

    backend_module = (
        BackendModule("unit", target_kind, package=package)
        .register_execution_backend()
        .register_device_codegen()
        .register_host_module()
        .register_host_hooks_module()
    )

    assert backend_module.name == "unit"
    assert backend_module.target_kinds == (target_kind,)
    assert backend_module.execution_backend_module == f"{package}.execution_backend"
    assert backend_module.device_codegen_module == f"{package}.codegen"
    assert backend_module.host_codegen_module == f"{package}.codegen"
    assert backend_module.host_codegen_hooks_module == f"{package}.codegen"
