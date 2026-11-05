"""TileLang assignment semantics are local to the F1 builder.

The provider parser registry is left unchanged. Native scalar declarations,
mutable cells, chained writes and augmented assignments are handled here.
"""

from tilelang.language.eager.builder import Builder

__all__ = ["Builder"]
