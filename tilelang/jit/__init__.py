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
    overload,
    Literal,
    Dict,  # For type hinting dicts
    Optional,
)
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.jit.kernel import JITKernel
from tilelang.cache import cached
from os import path, makedirs
from logging import getLogger
import functools
from tilelang.jit.param import Kernel, _P, _RProg

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


class _JitImplementation:

    out_idx: Any
    target: Union[str, Target]
    target_host: Union[str, Target]
    execution_backend: Literal["dlpack", "ctypes", "cython"]
    verbose: bool
    pass_configs: Optional[Dict[str, Any]]
    debug_root_path: Optional[str]

    def __init__(self,
                 out_idx: Any = None,
                 target: Union[str, Target] = "auto",
                 target_host: Union[str, Target] = None,
                 execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 debug_root_path: Optional[str] = None):
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
        """
        self.out_idx = out_idx
        self.execution_backend = execution_backend
        self.target = target
        self.target_host = target_host
        self.verbose = verbose
        self.pass_configs = pass_configs

        # Corrected debug_root_path handling
        self.debug_root_path = debug_root_path
        if self.debug_root_path is not None and not path.isabs(self.debug_root_path):
            try:
                base_path = path.dirname(path.dirname(path.dirname(__file__)))
                self.debug_root_path = path.join(base_path, self.debug_root_path)
            except NameError:
                self.debug_root_path = path.abspath(self.debug_root_path)

        self._kernel_cache: Dict[tuple, Kernel] = {}

    # This tells the type checker what the *wrapper* function will return.
    # this is for linting, please do not remove it.
    @overload
    def __call__(self, func: Callable[_P, _RProg]) -> Callable[_P, Tuple[_RProg, Kernel]]:
        ...

    @overload
    def __call__(self, func: Callable[_P, _RProg]) -> Callable[_P, Kernel]:
        ...

    # Actual implementation of __call__
    def __call__(
        self,
        func: Callable[_P, _RProg]  # func is Union[Callable[_P, _RProg], PrimFunc] in original
    ) -> Callable[_P, Any]:

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
            # Separate out the tuning parameters from the user's kwargs
            tune_params = kwargs.pop('__tune_params', {})

            key_args_tuple = args
            key_kwargs_tuple = tuple(sorted(kwargs.items()))
            key = (key_args_tuple, key_kwargs_tuple)

            if key not in self._kernel_cache:
                # Ensure 'func' (the original user function) is used correctly
                program_result_source = func
                if isinstance(program_result_source, PrimFunc):
                    program_result = program_result_source
                elif callable(program_result_source):
                    program_result = program_result_source(*args, **kwargs, **tune_params)
                else:
                    raise ValueError(f"Invalid function type: {type(program_result_source)}")

                kernel_result = compile(
                    program_result,
                    out_idx=self.out_idx,
                    execution_backend=self.execution_backend,
                    target=self.target,
                    target_host=self.target_host,
                    verbose=self.verbose,
                    pass_configs=self.pass_configs,
                )

                if self.debug_root_path:
                    func_name = getattr(func, '__name__', 'jit_kernel')  # Use func for name
                    kernel_file = f'tilelang_jit_kernel_{func_name}.c'
                    program_file = f'tilelang_jit_program_{func_name}.py'
                    makedirs(self.debug_root_path, exist_ok=True)
                    with open(path.join(self.debug_root_path, kernel_file), 'w') as f:
                        print(kernel_result.get_kernel_source(), file=f)
                    with open(path.join(self.debug_root_path, program_file), 'w') as f:
                        print(program_result.script(), file=f)

                self._kernel_cache[key] = kernel_result

            return self._kernel_cache[key]

        return wrapper


def jit(  # This is the new public interface
        func: Union[Callable[_P, _RProg], PrimFunc, None] = None,
        *,  # Indicates subsequent arguments are keyword-only
        out_idx: Any = None,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        verbose: bool = False,
        pass_configs: Optional[Dict[str, Any]] = None,
        debug_root_path: Optional[str] = None):
    """
    Just-In-Time (JIT) compiler decorator for TileLang functions.

    This decorator can be used without arguments (e.g., `@tilelang.jit`):
       Applies JIT compilation with default settings.

    Parameters
    ----------
    func_or_out_idx : Any, optional
        If using `@tilelang.jit(...)` to configure, this is the `out_idx` parameter.
        If using `@tilelang.jit` directly on a function, this argument is implicitly
        the function to be decorated (and `out_idx` will be `None`).
    target : Union[str, Target], optional
        Compilation target for TVM (e.g., "cuda", "llvm"). Defaults to "auto".
    target_host : Union[str, Target], optional
        Target host for cross-compilation. Defaults to None.
    execution_backend : Literal["dlpack", "ctypes", "cython"], optional
        Backend for kernel execution and argument passing. Defaults to "cython".
    verbose : bool, optional
        Enables verbose logging during compilation. Defaults to False.
    pass_configs : Optional[Dict[str, Any]], optional
        Configurations for TVM's pass context. Defaults to None.
    debug_root_path : Optional[str], optional
        Directory to save compiled kernel source for debugging. Defaults to None.

    Returns
    -------
    Callable
        Either a JIT-compiled wrapper around the input function, or a configured decorator
        instance that can then be applied to a function.
    """
    if callable(func):
        # Case 1: Used as @jit (func_or_out_idx is the function, others are defaults)
        # Create a default _JitImplementation instance and apply it to the function.
        default_decorator = _JitImplementation(
            out_idx=out_idx,  # Explicitly None for the default case
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            pass_configs=pass_configs,
            debug_root_path=debug_root_path)
        return default_decorator(func)
    elif isinstance(func, PrimFunc):
        raise ValueError("Use tilelang.jit to decorate prim_func is not supported yet.")
    else:
        # Case 2: Used as @jit(...) to configure, or func_or_out_idx is meant as out_idx.
        # Create a _JitImplementation instance with the provided/defaulted arguments.
        # This instance is a decorator that will be applied to the function later.
        configured_decorator = _JitImplementation(
            out_idx=out_idx,  # Pass along; could be an actual out_idx or None
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            pass_configs=pass_configs,
            debug_root_path=debug_root_path)
        return configured_decorator
