# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs. 
It includes functionality to JIT-compile TileLang programs into a runnable 
kernel adapter using TVM.
"""

from typing import (
    Any,
    List,
    Union,
    Callable,
    Tuple,
    TypeVar,
    overload,
    Literal,
    Dict,  # For type hinting dicts
    Optional,
)
from typing_extensions import ParamSpec
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.jit.kernel import JITKernel
from tilelang.cache import cached
from os import path, makedirs
from logging import getLogger
import functools

logger = getLogger(__name__)


def compile(
    func: PrimFunc = None,
    out_idx: Union[List[int], int, None] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    verbose: bool = False,
    pass_configs: Optional[Dict[str, Any]] = None,
) -> JITKernel:
    """
    Compile the given TileLang PrimFunc with TVM and build a JITKernel.
    Parameters
    ----------
    func : tvm.tir.PrimFunc, optional
        The TileLang TIR function to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["dlpack", "ctypes"], optional
        Execution backend to use for kernel execution (default: "dlpack").
    target : Union[str, Target], optional
        Compilation target, either as a string or a TVM Target object (default: "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output (default: False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Available options:
            "tir.disable_vectorize": bool, default: False
            "tl.disable_tma_lower": bool, default: False
            "tl.disable_warp_specialized": bool, default: False
            "tl.config_index_bitwidth": int, default: None
            "tl.disable_dynamic_tail_split": bool, default: False
            "tl.dynamic_vectorize_size_bits": int, default: 128
            "tl.disable_safe_memory_legalize": bool, default: False
    """
    return cached(
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
    )


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


class jit:
    # Overload __init__ to help type checkers understand the effect of return_program
    # The '-> None' is for __init__ itself. The crucial part is Literal for return_program.
    @overload
    def __init__(self,
                 out_idx: Any = None,
                 target: Union[str, Target] = "auto",
                 target_host: Union[str, Target] = None,
                 execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 debug_root_path: Optional[str] = None,
                 *,
                 return_program: Literal[True]) -> None:
        ...

    @overload
    def __init__(self,
                 out_idx: Any = None,
                 target: Union[str, Target] = "auto",
                 target_host: Union[str, Target] = None,
                 execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 debug_root_path: Optional[str] = None,
                 *,
                 return_program: Literal[False] = False) -> None:
        ...

    # Actual implementation of __init__
    def __init__(self,
                 out_idx: Any = None,
                 target: Union[str, Target] = "auto",
                 target_host: Union[str, Target] = None,
                 execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 debug_root_path: Optional[str] = None,
                 *,
                 return_program: bool = False):
        """
        Initializes the JIT compiler decorator.

        Parameters
        ----------
        out_idx : Any, optional
            Index(es) of the output tensors to return from the compiled kernel
            (default: None, meaning all outputs are returned or determined by the kernel itself).
        target : Union[str, Target], optional
            Compilation target for TVM. Can be a string (e.g., "cuda", "llvm")
            or a TVM Target object. If "auto", the target is determined automatically
            (default: "auto").
        target_host : Union[str, Target], optional
            Target host for cross-compilation, similar to `target` (default: None).
        execution_backend : Literal["dlpack", "ctypes", "cython"], optional
            The backend used for kernel execution and argument passing.
            "dlpack" is generally preferred for zero-copy tensor passing with compatible frameworks.
            "ctypes" uses standard C types. "cython" uses Cython for potentially faster execution.
            (default: "cython").
        verbose : bool, optional
            If True, enables verbose logging during compilation (default: False).
        pass_configs : Optional[Dict[str, Any]], optional
            A dictionary of configurations for TVM's pass context. These can fine-tune
            the compilation process. Examples include "tir.disable_vectorize"
            (default: None).
        debug_root_path : Optional[str], optional
            If provided, the compiled kernel's source code will be saved to a file
            in this directory. This is useful for debugging the generated code.
            If None, no debug information is saved (default: None).
            If a relative path is given, it's made absolute relative to the project root
            or current working directory.
        return_program : bool, optional
            If True, the decorated function will return a tuple containing the
            original program's result and the compiled kernel. If False, only the
            compiled kernel is returned (default: False).
        """
        if debug_root_path is None:
            # This logic was previously under 'if debug and debug_root_path is None:'
            # Now, if debug_root_path is explicitly None, we don't try to set a default path.
            # If a user wants debugging, they must provide a path.
            pass
        elif not path.isabs(debug_root_path):  # If a relative path is given, make it absolute
            try:
                # This assumes the file is part of a typical project structure
                base_path = path.dirname(path.dirname(path.dirname(__file__)))
                debug_root_path = path.join(base_path, debug_root_path)
            except NameError:  # __file__ is not defined (e.g., in a REPL or notebook)
                # Fallback to making it absolute based on current working directory if __file__ fails
                debug_root_path = path.abspath(debug_root_path)

        self.out_idx = out_idx
        self.execution_backend = execution_backend
        self.target = target
        self.target_host = target_host
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.debug_root_path: Optional[str] = debug_root_path
        self.return_program: bool = return_program

        # Type hint the caches
        self._program_cache: Dict[tuple, _RProg] = {}
        self._kernel_cache: Dict[tuple, Kernel] = {}

    # Overload __call__ based on the value of self.return_program
    # This tells the type checker what the *wrapper* function will return.
    # The wrapper will take the same parameters P as the original function.

    # Case 1: return_program is True
    @overload
    def __call__(self, func: Callable[_P, _RProg]) -> Callable[_P, Tuple[_RProg, Kernel]]:
        # This signature is chosen by the type checker if self.return_program is True
        # (inferred from the __init__ call).
        ...

    # Case 2: return_program is False (or not specified, defaulting to False)
    @overload
    def __call__(self, func: Callable[_P, _RProg]) -> Callable[_P, Kernel]:
        # This signature is chosen if self.return_program is False.
        ...

    # Actual implementation of __call__
    def __call__(
        self, func: Union[Callable[_P, _RProg], PrimFunc]
    ) -> Callable[_P, Any]:  # Any for implementation flexibility

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:  # Use _P.args and _P.kwargs
            # Create a hashable key. args is already a tuple.
            # For kwargs, convert to a sorted tuple of items to ensure consistent ordering.
            key_args_tuple = args
            key_kwargs_tuple = tuple(sorted(kwargs.items()))
            key = (key_args_tuple, key_kwargs_tuple)

            # Check if both program and kernel are cached.
            # If program is not cached, we'll recompute both.
            # (The original check 'key not in self._program_cache or key not in self._kernel_cache'
            # implies that if either is missing, both are recomputed and stored.
            # A simpler 'key not in self._program_cache' would often suffice if they are always
            # added together.)
            if key not in self._program_cache:  # Assuming if program isn't there, kernel isn't either or needs refresh
                if isinstance(func, PrimFunc):
                    program_result = func
                elif isinstance(func, Callable):
                    program_result = func(*args, **kwargs)
                else:
                    raise ValueError(f"Invalid function type: {type(func)}")

                kernel_result = compile(
                    program_result,
                    out_idx=self.out_idx,
                    execution_backend=self.execution_backend,
                    target=self.target,
                    target_host=self.target_host,
                    verbose=self.verbose,
                    pass_configs=self.pass_configs,
                )

                if self.debug_root_path:  # Check if a path is provided
                    func_name = func.__name__
                    kernel_file = f'tilelang_jit_kernel_{func_name}.c'
                    # Ensure the debug directory exists
                    makedirs(self.debug_root_path, exist_ok=True)
                    with open(path.join(self.debug_root_path, kernel_file), 'w') as f:
                        print(kernel_result.get_kernel_source(), file=f)

                self._program_cache[key] = program_result
                self._kernel_cache[key] = kernel_result

            # Retrieve from cache (even if just populated)
            cached_program = self._program_cache[key]
            cached_kernel = self._kernel_cache[key]

            if self.return_program:
                return cached_program, cached_kernel
            else:
                return cached_kernel

        return wrapper
