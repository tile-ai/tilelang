from typing import (
    Any,
    TypeVar,
)
from typing_extensions import ParamSpec


# --- Mocking dependencies for the example to run ---
# In your actual code, these would be your real types.
class Program:
    """Placeholder for the type returned by the original decorated function."""

    def __init__(self, data: str):
        self.data = data

    def __repr__(self):
        return f"Program('{self.data}')"


class Kernel:
    """Placeholder for the type of the compiled kernel."""

    def __init__(self, source: str, out_idx: Any):
        self.source_code = source
        self.out_idx = out_idx

    def get_kernel_source(self) -> str:
        return self.source_code

    def __repr__(self):
        return f"Kernel('{self.source_code[:20]}...')"


# --- End Mocking ---

# P (Parameters) captures the argument types of the decorated function.
_P = ParamSpec("_P")
# R_prog (Return type of Program) captures the return type of the original decorated function.
# We assume the original function returns something compatible with 'Program'.
_RProg = TypeVar("_RProg", bound=Program)

__all__ = ["Program", "Kernel", "_P", "_RProg"]
