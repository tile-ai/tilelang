# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs. 
It includes functionality to JIT-compile TileLang programs into a runnable 
kernel adapter using TVM.
"""

from typing import Callable, List, Literal, Union

import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.utils.adapter import TorchDLPackKernelAdapter, BaseKernelAdapter
from tilelang.utils.target import determine_target, AVALIABLE_TARGETS


def jit(
    func: Callable = None,
    *,  # Enforce keyword-only arguments from here on
    out_idx: Union[List[int], int] = None,
    wrapper: Literal["auto", "dl_pack", "torch_cpp"] = "auto",
    target: Union[str, Target] = "auto",
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
    wrapper : Literal["auto", "dl_pack", "torch_cpp"], optional
        The wrapper type to use for the kernel adapter. Currently only "auto" 
        and "dl_pack" are effectively supported in this code, but "torch_cpp"
        can be integrated as well. Defaults to "auto".
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
        # Merge static shared memory where possible during lowering.
        with tvm.transform.PassContext(opt_level=3):
            rt_mod, params = tilelang.lower(tilelang_func, target=target)

        # Create an adapter using TorchDLPack for bridging.
        adapter = TorchDLPackKernelAdapter(rt_mod, params=params, result_idx=out_idx)
        return adapter

    # If `func` was given, compile it immediately and return the adapter.
    if func is not None:
        return _compile_and_create_adapter(func)

    # Otherwise, return a decorator that expects a function to compile.
    def real_decorator(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        return _compile_and_create_adapter(tilelang_func)

    return real_decorator
