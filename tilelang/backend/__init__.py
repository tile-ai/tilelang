"""Backend interfaces, loaded when the owning compiler component is selected."""

from importlib import import_module

_EXPORTS = {
    "PassPipeline": "pass_pipeline",
    "DeviceCodegen": "device_codegen",
    "HostCodegen": "host_codegen",
    "HostCodegenHook": "host_codegen",
    "ExecutionBackendSpec": "execution_backend",
    "BackendContext": "module",
    "BackendModule": "module",
    "create_backend_context": "module",
    "get_backend": "module",
    "list_backends": "module",
    "register_backend": "module",
    "auto_detect_target": "target",
    "list_target_detectors": "target",
    "register_target_detector": "target",
    "register_target_execution_normalizer": "target",
    "register_target_normalizer": "target",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
