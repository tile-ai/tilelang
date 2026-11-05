"""CUDA intrinsic generators, loaded when their architecture is selected."""

from importlib import import_module


_EXPORTS = {
    "mma_store_index_map": ("layout.utils", "mma_store_index_map"),
    "get_ldmatrix_offset": ("layout.utils", "get_ldmatrix_offset"),
    "get_mma_micro_size": ("layout.utils", "get_mma_micro_size"),
    "TensorCoreIntrinEmitter": ("macro.mma_macro_generator", "TensorCoreIntrinEmitter"),
    "TensorCoreIntrinEmitterWithLadderTransform": ("macro.mma_macro_generator", "TensorCoreIntrinEmitterWithLadderTransform"),
    "TensorCoreIntrinEmitterSM120": ("macro.mma_sm120_macro_generator", "TensorCoreIntrinEmitterSM120"),
    "get_swizzle_layout": ("layout.mma_layout", "get_swizzle_layout"),
    "make_mma_swizzle_layout": ("layout.mma_layout", "make_mma_swizzle_layout"),
}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
