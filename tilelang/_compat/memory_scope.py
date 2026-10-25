"""TileLang-owned memory spaces, consumed before native storage lowering."""

from tvm_tirx.runtime import MemoryScope


def fragment_scope() -> MemoryScope:
    """Get the fragment scope without conflating it with physical local memory."""
    try:
        return MemoryScope("local.fragment")
    except ValueError:
        return MemoryScope.register("local.fragment")


def reducer_scope() -> MemoryScope:
    """Get virtual reducer storage, materialized before native allocation lowering."""
    try:
        return MemoryScope("local.reducer")
    except ValueError:
        return MemoryScope.register("local.reducer")
