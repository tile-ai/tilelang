"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs.
It includes functionality to JIT-compile TileLang programs into a runnable
kernel adapter using TVM.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    overload,
    Literal,
)
from collections.abc import Iterable
# Python 3.9 compatibility for ParamSpec
try:
    from typing import ParamSpec
except ImportError:  # Python < 3.10
    from typing_extensions import ParamSpec
from tilelang import tvm as tvm
from tilelang.language.v2 import PrimFunc
from tilelang.jit.adapter.utils import is_metal_target
from tvm.target import Target

from tilelang.jit.kernel import JITKernel
from tilelang.utils.target import determine_target
from tilelang.cache import cached
from os import path, makedirs
from logging import getLogger
from tilelang.jit.param import Kernel
import concurrent.futures

from tqdm.auto import tqdm

logger = getLogger(__name__)

_P = ParamSpec('_P')
_KP = ParamSpec('_KP')
_T = TypeVar('_T')


def compile(
    func: PrimFunc[_KP, _T] = None,
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel[_KP, _T]:
    """
    Compile the given TileLang PrimFunc with TVM and build a JITKernel.
    Parameters
    ----------
    func : tvm.tir.PrimFunc, optional
        The TileLang TIR function to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["dlpack", "ctypes", "cython", "nvrtc"], optional
        Execution backend to use for kernel execution (default: "cython").
    target : Union[str, Target], optional
        Compilation target, either as a string or a TVM Target object (default: "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output (default: False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.transform.PassConfigKey` for supported options.
    """
    assert isinstance(func, PrimFunc), f"target function must be a PrimFunc but got {type(func)}"
    if isinstance(compile_flags, str):
        compile_flags = [compile_flags]

    # This path is not a performance critical path, so we can afford to convert the target.
    target = Target(determine_target(target))

    if is_metal_target(target):
        assert execution_backend == 'torch', 'Currently metal target only support `tl.jit(execution_backend="torch")`'

    return cached(
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def par_compile(funcs: Iterable[PrimFunc[_KP, _T]],
                out_idx: list[int] | int | None = None,
                execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
                target: str | Target = "auto",
                target_host: str | Target | None = None,
                verbose: bool = False,
                pass_configs: dict[str, Any] | None = None,
                compile_flags: list[str] | str | None = None,
                num_workers: int = None,
                ignore_error: bool = False) -> list[JITKernel[_KP, _T]]:
    """
    Parallel compile multiple TileLang PrimFunc with TVM and build JITKernels.
    Parameters
    ----------
    funcs : Iterable[tvm.tir.PrimFunc]
        The TileLang TIR functions to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["dlpack", "ctypes", "cython", "nvrtc"], optional
        Execution backend to use for kernel execution (default: "cython").
    target : Union[str, Target], optional
        Compilation target, either as a string or a TVM Target object (default: "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output (default: False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.transform.PassConfigKey` for supported options.
    """
    with concurrent.futures.ThreadPoolExecutor(num_workers, 'tl-par-comp') as executor:
        futures = []
        future_map = {}
        for i, func in enumerate(funcs):
            future = executor.submit(
                compile,
                func=func,
                out_idx=out_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
            future_map[future] = i
            futures.append(future)
        results = [... for _ in futures]
        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Parallel Compiling",
        ):
            idx = future_map[future]
            if ignore_error:
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Error compiling function at index {idx}: {e}")
                    results[idx] = None
            else:
                results[idx] = future.result()
        return results
    return results


@dataclass
class JITImpl(Generic[_P, _KP, _T]):
    func: Callable[_P, _T] | PrimFunc[_KP, _T]
    out_idx: list[int] | int | None
    execution_backend: Literal["dlpack", "ctypes", "cython"]
    target: str | Target
    target_host: str | Target
    verbose: bool
    pass_configs: dict[str, Any] | None
    debug_root_path: str | None
    compile_flags: list[str] | str | None
    func_source: str
    signature: inspect.Signature

    def __post_init__(self):
        if self.debug_root_path is not None and not path.isabs(self.debug_root_path):
            try:
                base_path = path.dirname(path.dirname(path.dirname(__file__)))
                self.debug_root_path = path.join(base_path, self.debug_root_path)
            except NameError:
                self.debug_root_path = path.abspath(self.debug_root_path)
        self._kernel_cache: dict[tuple, Kernel] = {}

    def get_tir(self, *args: _P.args, **kwargs: _P.kwargs) -> PrimFunc[_KP, _T]:
        program_result_source = self.func
        if isinstance(program_result_source, PrimFunc):
            program_result = program_result_source
        elif callable(program_result_source):
            program_result = program_result_source(*args, **kwargs)
        else:
            raise ValueError(f"Invalid function type: {type(program_result_source)}")
        return program_result

    def par_compile(self,
                    configs: Iterable[dict[str, Any] | tuple[str, Any]],
                    num_workers: int = None,
                    ignore_error: bool = False) -> list[JITKernel[_KP, _T]]:
        configs = list(configs)
        funcs = []
        for cfg in tqdm(configs, desc='Elaborating'):
            if isinstance(cfg, tuple):
                funcs.append(self.get_tir(*cfg))
            elif isinstance(cfg, dict):
                funcs.append(self.get_tir(**cfg))
            else:
                raise ValueError(f"Invalid config type: {type(cfg)}, expected tuple or dict.")
        return par_compile(
            funcs,
            out_idx=self.out_idx,
            execution_backend=self.execution_backend,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
            num_workers=num_workers,
            ignore_error=ignore_error)

    def compile(self, *args: _P.args, **kwargs: _P.kwargs) -> JITKernel[_KP, _T]:
        func = self.get_tir(*args, **kwargs)
        kernel_result = compile(
            func,
            out_idx=self.out_idx,
            execution_backend=self.execution_backend,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )

        if self.debug_root_path:
            if isinstance(self.func, PrimFunc):
                func_name = self.func.attrs['global_symbol']
            else:
                func_name = getattr(self.func, '__name__', 'jit_kernel')
            kernel_file = f'tilelang_jit_kernel_{func_name}.c'
            program_file = f'tilelang_jit_program_{func_name}.py'
            makedirs(self.debug_root_path, exist_ok=True)
            with open(path.join(self.debug_root_path, kernel_file), 'w') as f:
                print(kernel_result.get_kernel_source(), file=f)
            with open(path.join(self.debug_root_path, program_file), 'w') as f:
                print(func.script(), file=f)

        return kernel_result

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> JITKernel[_KP, _T]:
        # Separate out the tuning parameters from the user's kwargs
        tune_params = kwargs.pop('__tune_params', {})
        # Whether to return the compile arguments (out_idx, target, target_host, etc.) for autotuner cache
        return_compile_arguments = kwargs.pop('__return_compile_arguments', False)
        if return_compile_arguments:
            compile_args = {
                'out_idx': self.out_idx,
                'execution_backend': self.execution_backend,
                'target': self.target,
                'target_host': self.target_host,
                'verbose': self.verbose,
                'pass_configs': self.pass_configs,
                'compile_flags': self.compile_flags,
            }
            return compile_args

        key_args_tuple = args
        key_kwargs_tuple = tuple(sorted(kwargs.items()))
        tuned_key_kwargs_tuple = tuple(sorted(tune_params.items()))
        key = (key_args_tuple, key_kwargs_tuple, tuned_key_kwargs_tuple)

        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.compile(*args, **kwargs, **tune_params)

        return self._kernel_cache[key]


@overload
def jit(func: Callable[_P, PrimFunc[_KP, _T]]) -> JITImpl[_P, _KP, _T]:
    ...


@overload
def jit(
    *,  # Indicates subsequent arguments are keyword-only
    out_idx: Any = None,
    target: str | Target = "auto",
    target_host: str | Target = None,
    execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
    verbose: bool = False,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None
) -> Callable[[Callable[_P, PrimFunc[_KP, _T]]], JITImpl[_P, _KP, _T]]:
    ...


def jit(  # This is the new public interface
        func: Callable[_P, _T] | PrimFunc | None = None,
        *,  # Indicates subsequent arguments are keyword-only
        out_idx: Any = None,
        target: str | Target = "auto",
        target_host: str | Target = None,
        execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        debug_root_path: str | None = None,
        compile_flags: list[str] | str | None = None):
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
    execution_backend : Literal["dlpack", "ctypes", "cython", "nvrtc"], optional
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
    if isinstance(compile_flags, str):
        compile_flags = [compile_flags]

    def decorator(func: Callable[_P, _T]) -> JITImpl[_P, _T]:
        if isinstance(func, PrimFunc):
            orig_func = func.orig_func
        else:
            orig_func = func
        return JITImpl(
            func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
            debug_root_path=debug_root_path,
            compile_flags=compile_flags,
            func_source=inspect.getsource(orig_func),
            signature=inspect.signature(orig_func),
        )

    if func is not None:
        return decorator(func)
    else:
        return decorator
