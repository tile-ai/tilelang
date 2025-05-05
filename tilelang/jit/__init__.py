# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs. 
It includes functionality to JIT-compile TileLang programs into a runnable 
kernel adapter using TVM.
"""

from typing import Callable, List, Literal, Union, Any, Optional, Dict

from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.utils.target import determine_target, AVALIABLE_TARGETS
from tilelang.cache import cached
from logging import getLogger

logger = getLogger(__name__)


def jit(
    func: Callable = None,
    *,  # Enforce keyword-only arguments from here on
    out_idx: Union[List[int], int] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    verbose: bool = False,
    **pass_config_kwargs: Optional[Dict[str, Any]],
) -> BaseKernelAdapter:
    """
    A decorator (or decorator factory) that JIT-compiles a given TileLang PrimFunc 
    into a runnable kernel adapter using TVM. If called with arguments, it returns 
    a decorator that can be applied to a function. If called without arguments, 
    it directly compiles the given function.

    Parameters
    ----------
    func : Callable, optional
        The TileLang PrimFunc to JIT-compile. If None, this function returns a 
        decorator that expects a TileLang PrimFunc.
    out_idx : Union[List[int], int], optional
        The index (or list of indices) of the function outputs. This can be used
        to specify which outputs from the compiled function will be returned.
    execution_backend : Literal["dlpack", "ctypes"], optional
        The wrapper type to use for the kernel adapter. Currently, only "dlpack"
        and "ctypes" are supported.
    target : Union[str, Target], optional
        The compilation target for TVM. If set to "auto", an appropriate target
        will be inferred automatically. Otherwise, must be one of the supported
        strings in AVALIABLE_TARGETS or a TVM Target instance.

    Returns
    -------
    BaseKernelAdapter
        An adapter object that encapsulates the compiled function and can be
        used to execute it.

    Raises
    ------
    AssertionError
        If the provided target is an invalid string not present in AVALIABLE_TARGETS.
    """

    # If the target is specified as a string, ensure it is valid and convert to a TVM Target.
    if isinstance(target, str):
        assert target in AVALIABLE_TARGETS, f"Invalid target: {target}"
        target = determine_target(target)

    target = Target(target)

    assert execution_backend in ["dlpack", "ctypes", "cython"], "Invalid execution backend."

    def _compile_and_create_adapter(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        """
        Compile the given TileLang PrimFunc with TVM and build a kernel adapter.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.

        Returns
        -------
        BaseKernelAdapter
            The compiled and ready-to-run kernel adapter.
        """
        if verbose:
            logger.info(f"Compiling TileLang function:\n{tilelang_func}")
        return compile(
            tilelang_func,
            target=target,
            verbose=verbose,
            execution_backend=execution_backend,
            out_idx=out_idx,
            **pass_config_kwargs,
        ).adapter

    # If `func` was given, compile it immediately and return the adapter.
    if func is not None:
        return _compile_and_create_adapter(func)

    # Otherwise, return a decorator that expects a function to compile.
    def real_decorator(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        return _compile_and_create_adapter(tilelang_func)

    return real_decorator


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