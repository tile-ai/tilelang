"""TileLang Relax utilities."""

from . import _ffi_api


def FuseTIR():
    """Fuse TIR blocks into a single block."""
    return _ffi_api.FuseTIR()
