"""Custom exceptions for TileLang JIT compilation."""


class JITNoBuilderError(Exception):
    """
    Exception raised when JIT-related operations require a Builder but none exists.

    In eager mode, TileLang constructs AST directly without an explicit prim_func,
    so there must be a Builder available. This error is raised when eager-only
    features like T.const() or T.Kernel() are called outside of a JIT/prim_func context.
    """

    pass


class EagerJITBuildError(Exception):
    """
    Exception raised for errors when building TileLang eager JIT kernels.

    This error indicates that something went wrong during the eager-style
    kernel construction process.
    """

    pass
