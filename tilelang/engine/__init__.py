"""Compilation entry points and native kernel parameter metadata."""

from importlib import import_module

# Bind these names after loading the submodule. Otherwise a direct import of
# engine.lower (for example by JIT) installs the module in this package's
# namespace and bypasses __getattr__ when users later request the function.
from .lower import is_device_call as is_device_call, lower as lower

_EXPORTS = {
    "KernelParam": ("param", "KernelParam"),
    "register_cuda_postproc": ("callback", "register_cuda_postproc"),
    "register_hip_postproc": ("callback", "register_hip_postproc"),
    "register_c_postproc": ("callback", "register_c_postproc"),
}
__all__ = ["lower", "is_device_call", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
