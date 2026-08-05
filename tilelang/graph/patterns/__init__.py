"""Built-in pattern rewriters for common LLM kernels.

Each submodule exposes its pattern(s) as module-level constants of type
:class:`tilelang.graph.pattern_rewrite._RegisteredPattern`, built via
:func:`tilelang.graph.pattern_rewrite.make_pattern`.  ``DEFAULT_PATTERNS``
is the ordered list used by the default pipeline; the order is also the
matching priority.

Callers compose their own list explicitly and pass it to
:class:`PatternRewritePass`::

    from tilelang.graph.patterns import DEFAULT_PATTERNS
    PatternRewritePass(DEFAULT_PATTERNS)

To use a subset or add your own pattern, build a list::

    from tilelang.graph.patterns import DEFAULT_PATTERNS
    from tilelang.graph.pattern_rewrite import make_pattern
    custom = make_pattern(...)
    PatternRewritePass(DEFAULT_PATTERNS + [custom])
"""

DEFAULT_PATTERNS = [
    # residual_rmsnorm.PATTERN,
    # rmsnorm.PATTERN,
    # fused_rope.PATTERN,
    # reshape_transpose.RESHAPE_PERMUTE,
    # reshape_transpose.PERMUTE_RESHAPE,
]

__all__ = ["DEFAULT_PATTERNS"]
