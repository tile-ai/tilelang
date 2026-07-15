"""Exception types raised by the TileIR backend."""

from __future__ import annotations


class TileIRLoweringError(RuntimeError):
    """Base class for TileLang TIR to CUDA Tile IR lowering failures."""


class TileIRLoweringNotImplementedError(TileIRLoweringError, NotImplementedError):
    """Raised until a structured lowering implementation owns a TIR node."""


class _UnsupportedTileIRNode(TileIRLoweringNotImplementedError):
    """Internal signal for frontend AST nodes that are not lowered yet."""


class _UnboundScopeVariable(_UnsupportedTileIRNode):
    """Internal signal for a TIR ``Var`` whose binding is not available yet.

    ``_lower_let`` catches this signal to defer pure ``tirx.Bind`` expressions
    until their referenced loop variables enter scope.
    """


class TileIRAssemblyError(TileIRLoweringError):
    """Raised when CUDA Tile IR verification, translation, or assembly fails."""
