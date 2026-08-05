"""TileLang Relax utilities."""

from . import _ffi_api


def LowerPrimitiveFunctionsToTIR():
    """Lower primitive Relax functions into TIR PrimFuncs."""
    return _ffi_api.LowerPrimitiveFunctionsToTIR()
