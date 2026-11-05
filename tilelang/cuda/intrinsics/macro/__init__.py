"""Public tensor-core generators with architecture-local imports."""

from importlib import import_module


_EXPORTS = {
    "TensorCoreIntrinEmitter": "mma_macro_generator",
    "TensorCoreIntrinEmitterWithLadderTransform": "mma_macro_generator",
    "TensorCoreIntrinEmitterSM120": "mma_sm120_macro_generator",
    "SparseTensorCoreIntrinEmitter": "mma_sp_macro_generator",
}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
