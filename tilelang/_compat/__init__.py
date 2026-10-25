"""TileLang adapters for the standalone TIRx compiler.

Adapters live in TileLang's namespace. Importing this package never installs a
global ``tvm`` alias or modifies TIRx's registered Python classes.
"""

from . import stmt_functor
from .functor import PyStmtExprMutator, PyStmtExprVisitor

__all__ = ["PyStmtExprMutator", "PyStmtExprVisitor", "stmt_functor"]
