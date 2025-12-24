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
from tilelang.language.v2 import PrimFunc, PrimFuncCreater, prim_func
from tilelang.language.v2.annot import Annot
from tvm.target import Target

from tilelang.jit.kernel import JITKernel
from tilelang.cache import cached
from os import path, makedirs
from logging import getLogger
from tilelang.jit.param import Kernel
import concurrent.futures

from tqdm.auto import tqdm

logger = getLogger(__name__)

_P = ParamSpec("_P")
_KP = ParamSpec("_KP")
_T = TypeVar("_T")
_Ret = TypeVar("_Ret")


def compile(
    func: PrimFunc[_KP, _T] = None,
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
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
    execution_backend : Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"], optional
        Execution backend to use for kernel execution. If None, reads from
        TILELANG_EXECUTION_BACKEND environment variable (defaults to "auto").
    target : Union[str, Target], optional
        Compilation target, either as a string or a TVM Target object. If None, reads from
        TILELANG_TARGET environment variable (defaults to "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output. If None, reads from
        TILELANG_VERBOSE environment variable (defaults to False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.transform.PassConfigKey` for supported options.

    Environment Variables
    ---------------------
    TILELANG_TARGET : str
        Default compilation target (e.g., "cuda", "llvm"). Defaults to "auto".
    TILELANG_EXECUTION_BACKEND : str
        Default execution backend. Defaults to "auto".
    TILELANG_VERBOSE : str
        Set to "1", "true", "yes", or "on" to enable verbose compilation by default.
    """

    assert isinstance(func, PrimFunc), f"target function must be a PrimFunc but got {type(func)}"

    if hasattr(func, "out_idx_override"):
        if func.out_idx_override is not None and out_idx is not None:
            raise ValueError("Out index conflict: out_idx is specified and prim_func have returned `T.empty` tensors")
        out_idx = func.out_idx_override or out_idx

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


def par_compile(
    funcs: Iterable[PrimFunc[_KP, _T]],
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    num_workers: int | None = None,
    ignore_error: bool = False,
) -> list[JITKernel[_KP, _T]]:
    """
    Parallel compile multiple TileLang PrimFunc with TVM and build JITKernels.

    Parameters
    ----------
    funcs : Iterable[tvm.tir.PrimFunc]
        The TileLang TIR functions to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"], optional
        Execution backend to use for kernel execution. If None, reads from
        TILELANG_EXECUTION_BACKEND environment variable (defaults to "auto").
    target : Union[str, Target], optional
        Compilation target, either as a string or a TVM Target object. If None, reads from
        TILELANG_TARGET environment variable (defaults to "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output. If None, reads from
        TILELANG_VERBOSE environment variable (defaults to False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.transform.PassConfigKey` for supported options.

    Environment Variables
    ---------------------
    TILELANG_TARGET : str
        Default compilation target (e.g., "cuda", "llvm"). Defaults to "auto".
    TILELANG_EXECUTION_BACKEND : str
        Default execution backend. Defaults to "auto".
    TILELANG_VERBOSE : str
        Set to "1", "true", "yes", or "on" to enable verbose compilation by default.
    """

    with concurrent.futures.ThreadPoolExecutor(num_workers, "tl-par-comp") as executor:
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
class JITImpl(Generic[_P, _KP, _T, _Ret]):
    """
    Detailed Just-In-Time wrapper for TileLang programs.

    This dataclass encapsulates the configuration and runtime helpers used by the
    top-level `jit` and `jit2` decorators. It represents a configured JIT
    "factory" that can (a) elaborate TileLang/PrimFunc creators into concrete
    TIR (PrimFunc), (b) compile those TIR functions into runnable kernels via
    the TVM bridge, (c) cache compiled kernels keyed by call-site arguments
    (and optional tuning parameters), and (d) provide parallel compilation
    helpers for batch autotuning workflows.

    Attributes
    ----------
    out_idx : list[int] | int | None
        Which output tensor(s) of the compiled kernel should be returned to the
        caller. Accepts a single index, a list of indices, or None to return all.
    execution_backend : Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"]
        Backend used for exchanging arguments and executing the generated kernel.
    target : str | tvm.target.Target
        TVM compilation target (e.g. "cuda", "llvm", or "auto").
    target_host : str | tvm.target.Target | None
        Host target used for cross-compilation, or None to infer/default.
    verbose : bool
        Enable verbose messages during compilation/build.
    pass_configs : dict[str, Any] | None
        Extra TVM pass configuration options forwarded to the compiler's
        PassContext.
    debug_root_path : str | None
        If provided, compiled kernel source and the elaborated Python program
        are written to this directory to ease debugging and inspection.
    compile_flags : list[str] | str | None
        Additional flags passed to the compiler. A single string will be converted
        to a single-element list.
    func_source : str
        Original Python source string from which the PrimFunc or creator was
        derived. Used for diagnostics and debug dumps.
    signature : inspect.Signature
        Function signature of the original Python function (useful for tooling).
    v2 : bool
        Indicates whether the object wraps a "v2" PrimFunc creator (True) or a
        plain callable / PrimFunc (False). v2-mode enables argument conversion
        hooks and a distinct cache keying strategy.
    func : Callable | PrimFunc | PrimFuncCreater
        The underlying object: either a user function that returns a PrimFunc
        (creator), a PrimFuncCreater, or an already-constructed PrimFunc.
        For presentation/readability the function is stored last in the dataclass.

    Behavioral summary
    ------------------
    - get_tir(*args, **kwargs)
        Converts provided call-site arguments into a concrete PrimFunc. If the
        wrapped object is a PrimFuncCreater or a user callable, it is invoked
        with the given arguments. If the wrapped object is already a PrimFunc,
        it is returned as-is.

    - compile(...)
        A convenience wrapper that elaborates and immediately compiles a single
        PrimFunc into a JITKernel using the module-level `compile` function.
        When `debug_root_path` is set, the compiled C kernel and the source
        Python program are saved for inspection.

    - par_compile(configs, ...)
        Accepts an iterable of configs (either dicts mapping keyword args or
        tuples mapping to positional args). Each config is elaborated to a
        PrimFunc and the resulting set is compiled in parallel via the
        module-level `par_compile` helper. Returns a list of JITKernel objects
        in the same order as the provided configs.
    """

    out_idx: list[int] | int | None
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None
    target: str | Target | None
    target_host: str | Target | None
    verbose: bool | None
    pass_configs: dict[str, Any] | None
    debug_root_path: str | None
    compile_flags: list[str] | str | None
    func_source: str
    signature: inspect.Signature
    lazy_jit: bool
    # place func at the last element for better __repr__
    func: Callable[_P, _T] | PrimFunc[_KP, _T]

    @property
    def annot(self) -> dict[str, Annot]:
        assert self.lazy_jit, "annot is only support in @tilelang.jit2"
        return self.func.func_annot.annots

    def __post_init__(self):
        if self.debug_root_path is not None and not path.isabs(self.debug_root_path):
            try:
                base_path = path.dirname(path.dirname(path.dirname(__file__)))
                self.debug_root_path = path.join(base_path, self.debug_root_path)
            except NameError:
                self.debug_root_path = path.abspath(self.debug_root_path)
        self._kernel_cache: dict[tuple, Kernel] = {}
        self._tuner_cache: dict[tuple, Kernel] = {}

    def get_tir(self, *args: _P.args, **kwargs: _P.kwargs) -> PrimFunc[_KP, _T]:
        """
        Retrieve a TIR (Tensor Intermediate Representation) PrimFunc from the stored callable or object.
        """
        if isinstance(self.func, PrimFuncCreater):
            tir = self.func(*args, **kwargs)
        elif isinstance(self.func, PrimFunc):
            tir = self.func
        elif callable(self.func):
            tir = self.func(*args, **kwargs)
        else:
            raise ValueError(f"Invalid function type: {type(self.func)}")
        assert isinstance(tir, PrimFunc), f"target function must be a PrimFunc but got {type(tir)}"
        return tir

    def par_compile(
        self, configs: Iterable[dict[str, Any] | tuple[str, Any]], num_workers: int = None, ignore_error: bool = False
    ) -> list[JITKernel[_KP, _T]]:
        """
        Parallel compile multiple TileLang PrimFunc with TVM and build JITKernels.
        Parameters
        ----------
        configs : Iterable[Union[dict[str, Any], tuple[Any, ...]]]
            The configurations to elaborate and compile. Each config can be either
            a dictionary mapping keyword arguments to values, or a tuple of positional
            arguments.
        num_workers : int, optional
            Number of parallel workers to use for compilation. Defaults to None,
            which lets the system decide.
        ignore_error : bool, optional
            If True, compilation errors for individual configs will be logged
            as warnings and the corresponding result will be None. If False,
            any compilation error will raise an exception. Defaults to False.
        Returns
        -------
        List[JITKernel]
            A list of compiled JITKernel objects corresponding to the provided configs.
        """
        configs = list(configs)
        funcs = []
        for cfg in tqdm(configs, desc="Elaborating"):
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
            ignore_error=ignore_error,
        )

    def compile(self, *args: _P.args, **kwargs: _P.kwargs) -> _Ret:
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
                func_name = self.func.attrs["global_symbol"]
            else:
                func_name = getattr(self.func, "__name__", "jit_kernel")
            kernel_file = f"tilelang_jit_kernel_{func_name}.c"
            program_file = f"tilelang_jit_program_{func_name}.py"
            makedirs(self.debug_root_path, exist_ok=True)
            with open(path.join(self.debug_root_path, kernel_file), "w") as f:
                print(kernel_result.get_kernel_source(), file=f)
            with open(path.join(self.debug_root_path, program_file), "w") as f:
                print(func.script(), file=f)

        return kernel_result

    def parse_cache_key(self, *args: _P.args, **kwargs: _P.kwargs):
        if isinstance(self.func, PrimFuncCreater):
            tune_params = kwargs.pop("__tune_params", {})
            return self.func.func_annot.parse_key(*args, **kwargs, **tune_params)
        else:
            tune_params = kwargs.pop("__tune_params", {})
            key_args_tuple = args
            key_kwargs_tuple = tuple(sorted(kwargs.items()))
            tuned_key_kwargs_tuple = tuple(sorted(tune_params.items()))
            key = (key_args_tuple, key_kwargs_tuple, tuned_key_kwargs_tuple)
            return key

    def convert_kernel_args(self, *args: _P.args, **kwargs: _P.kwargs):
        if isinstance(self.func, PrimFuncCreater):
            tune_params = kwargs.pop("__tune_params", {})
            return self.func.func_annot.convert_to_kernel_args(*args, **kwargs, **tune_params)
        else:
            raise NotImplementedError("convert_arg_to_kernel_args is only implemented for PrimFuncCreater.")

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _Ret:
        # Separate out the tuning parameters from the user's kwargs
        # Whether to return the compile arguments (out_idx, target, target_host, etc.) for autotuner cache
        return_compile_arguments = kwargs.pop("__return_compile_arguments", False)
        if return_compile_arguments:
            logger.warning("`__return_compile_arguments` is deprecated and will be removed in future versions.")
            compile_args = {
                "out_idx": self.out_idx,
                "execution_backend": self.execution_backend,
                "target": self.target,
                "target_host": self.target_host,
                "verbose": self.verbose,
                "pass_configs": self.pass_configs,
                "compile_flags": self.compile_flags,
            }
            return compile_args

        key = self.parse_cache_key(*args, **kwargs)

        tune_params = kwargs.pop("__tune_params", {})

        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.compile(*args, **kwargs, **tune_params)
            self._kernel_cache[key] = kernel

        if self.lazy_jit:
            args = self.func.func_annot.convert_to_kernel_args(*args, **kwargs, **tune_params)
            return kernel(*args)
        else:
            return kernel


ExecutionBackend = Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"]


@overload
def jit(func: Callable[_P, PrimFunc[_KP, _T]]) -> JITImpl[_P, _KP, _T, JITKernel[_KP, _T]]: ...


@overload
def jit(
    *,  # Indicates subsequent arguments are keyword-only
    out_idx: Any = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable[[Callable[_P, PrimFunc[_KP, _T]]], JITImpl[_P, _KP, _T, JITKernel[_KP, _T]]]: ...


def jit(  # This is the new public interface
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    out_idx: Any = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
):
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
        Compilation target for TVM (e.g., "cuda", "llvm"). If None, reads from
        TILELANG_TARGET environment variable (defaults to "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation. Defaults to None.
    execution_backend : Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"], optional
        Backend for kernel execution and argument passing. If None, reads from
        TILELANG_EXECUTION_BACKEND environment variable (defaults to "auto").
    verbose : bool, optional
        Enables verbose logging during compilation. If None, reads from
        TILELANG_VERBOSE environment variable (defaults to False).
    pass_configs : Optional[Dict[str, Any]], optional
        Configurations for TVM's pass context. Defaults to None.
    debug_root_path : Optional[str], optional
        Directory to save compiled kernel source for debugging. Defaults to None.

    Environment Variables
    ---------------------
    TILELANG_TARGET : str
        Default compilation target (e.g., "cuda", "llvm"). Defaults to "auto".
    TILELANG_EXECUTION_BACKEND : str
        Default execution backend. Defaults to "auto".
    TILELANG_VERBOSE : str
        Set to "1", "true", "yes", or "on" to enable verbose compilation by default.

    Returns
    -------
    Callable
        Either a JIT-compiled wrapper around the input function, or a configured decorator
        instance that can then be applied to a function.
    """

    def decorator(func: Callable[_P, _T]) -> JITImpl[_P, _T]:
        if isinstance(func, (PrimFunc, PrimFuncCreater)):
            orig_func = func.orig_func
        else:
            orig_func = func
        return JITImpl(
            func=func,
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
            lazy_jit=False,
        )

    if func is not None:
        return decorator(func)
    else:
        return decorator


@overload
def lazy_jit(func: Callable[_KP, _T]) -> JITImpl[_KP, _KP, _T, _T]: ...


@overload
def lazy_jit(
    *,
    out_idx: Any = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable[[Callable[_KP, _T]], JITImpl[_KP, _KP, _T, _T]]: ...


def lazy_jit(
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
):
    """
    Lazy JIT compiler decorator - returns the kernel object on first call, then executes it.

    Supports environment variable defaults for target, execution_backend, and verbose.
    See `jit` documentation for parameter details and environment variables.
    """

    compile_args = dict(
        out_idx=None,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        debug_root_path=debug_root_path,
        compile_flags=compile_flags,
    )

    def decorator(func: Callable[_P, _T]):
        pf: PrimFunc[_P, _T] | PrimFuncCreater[_P, _T] = prim_func(func, generator=True)
        # if isinstance(pf, PrimFunc):
        #     compile_args.pop('debug_root_path', None)
        #     return compile(pf, **compile_args)
        # else:
        return JITImpl(
            func=pf, **compile_args, func_source=inspect.getsource(pf.orig_func), signature=inspect.signature(pf.orig_func), lazy_jit=True
        )

    return decorator(func) if func is not None else decorator
