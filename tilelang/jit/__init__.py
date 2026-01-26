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
from tilelang.language.eager import PrimFunc, prim_func, JITFunc
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
    Just-In-Time compilation wrapper for TileLang programs.

    This class provides a unified interface for compiling and executing TileLang
    kernels. It supports two execution modes that are automatically inferred:

    Execution Modes
    ---------------
    - **lazy**: The decorated function returns a PrimFunc explicitly. Calling the
      JIT wrapper returns a compiled kernel object, which can be invoked separately.
      This mode is useful when you want to inspect or reuse the kernel object.

      Example (lazy mode)::

          @tilelang.jit(out_idx=[-1])
          def matmul(M, N, K, block_M, block_N, block_K):
              @T.prim_func
              def kernel(A: T.Tensor((M, K), dtype), ...):
                  ...
              return kernel  # explicitly return PrimFunc

          kernel = matmul(1024, 1024, 1024, 128, 128, 32)  # returns kernel
          result = kernel(a, b)  # execute separately

    - **eager**: The decorated function uses the DSL builder pattern with tensor
      type annotations. Calling the JIT wrapper compiles and immediately executes
      the kernel, returning the result directly.

      Example (eager mode)::

          @tilelang.jit
          def gemm(A, B, C, block_M: int = 64):
              M, N, K = T.const("M N K")
              A: T.Tensor[[M, K], dtype]  # tensor shape via annotation
              B: T.Tensor[[K, N], dtype]
              C: T.Tensor[[M, N], dtype]
              with T.Kernel(...):
                  ...

          gemm(A, B, C)  # compiles and executes immediately

    The mode is automatically inferred based on whether the function returns a
    PrimFunc (lazy) or uses the builder pattern (eager).

    Attributes
    ----------
    out_idx : list[int] | int | None
        Index(es) of output tensor(s) to return (lazy mode only).
    execution_backend : str | None
        Backend for kernel execution ("auto", "dlpack", "tvm_ffi", etc.).
    target : str | Target | None
        TVM compilation target (e.g., "cuda", "llvm", "auto").
    target_host : str | Target | None
        Host target for cross-compilation.
    verbose : bool | None
        Enable verbose compilation output.
    pass_configs : dict[str, Any] | None
        TVM pass configuration options.
    debug_root_path : str | None
        Directory to save compiled kernel source for debugging.
    compile_flags : list[str] | str | None
        Additional compiler flags.
    func_source : str
        Original Python source code of the decorated function.
    signature : inspect.Signature
        Function signature of the original function.
    mode : Literal["auto", "lazy", "eager"]
        Execution mode. "auto" infers from function behavior.
    func : JITFunc
        The wrapped function object.
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
    mode: Literal["auto", "lazy", "eager"]
    # place func at the last element for better __repr__
    func: JITFunc[_KP, _T]

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
        self.initialize_jit_mode(*args, **kwargs)
        if isinstance(self.func, PrimFunc):
            tir = self.func
        elif callable(self.func):
            tir = self.func(*args, **kwargs)
        else:
            raise ValueError(f"Invalid function type: {type(self.func)}")
        assert isinstance(tir, PrimFunc), f"target function must be a PrimFunc but got {type(tir)}"
        return tir

    def _infer_jit_mode(self, *args: _P.args, **kwargs: _P.kwargs) -> Literal["lazy", "eager"]:
        """
        Infer the JIT execution mode based on function behavior.

        Returns "lazy" if the function explicitly returns a PrimFunc,
        or "eager" if it uses the DSL builder pattern.
        """
        if self.mode in ("lazy", "eager"):
            return self.mode
        # auto: infer by checking if function returns PrimFunc directly
        if not isinstance(self.func, JITFunc):
            return "lazy"
        is_lazy_style = self.func._is_lazy_style(*args, **kwargs)
        return "lazy" if is_lazy_style else "eager"

    def initialize_jit_mode(self, *args: _P.args, **kwargs: _P.kwargs) -> Literal["lazy", "eager"]:
        if self.mode == "auto":
            self.mode = self._infer_jit_mode(*args, **kwargs)
        self.func.set_mode(self.mode)
        if self.mode == "eager" and self.out_idx is not None:
            raise ValueError("out_idx is only supported in lazy mode. In eager mode, use T.empty() to declare output tensors instead.")
        return self.mode

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
        prim_func = self.get_tir(*args, **kwargs)
        kernel_result = compile(
            prim_func,
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
                print(prim_func.script(), file=f)

        return kernel_result

    def parse_cache_key(self, *args: _P.args, **kwargs: _P.kwargs):
        tune_params = kwargs.pop("__tune_params", {})
        key_args_tuple = args
        key_kwargs_tuple = tuple(sorted(kwargs.items()))
        tuned_key_kwargs_tuple = tuple(sorted(tune_params.items()))
        key = (key_args_tuple, key_kwargs_tuple, tuned_key_kwargs_tuple)
        return key

    def get_kernel_source(self, *args: _P.args, **kwargs: _P.kwargs) -> str:
        kernel = self.compile(*args, **kwargs)
        return kernel.get_kernel_source()

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

        kwargs.update(kwargs.pop("__tune_params", {}))

        # infer mode early, before parse_args needs it
        if self.mode == "auto":
            self.mode = self._infer_jit_mode(*args, **kwargs)
            self.func.set_mode(self.mode)

        key, kernel_args = self.func.parse_args(*args, **kwargs)
        kernel = self._kernel_cache.get(key, None)
        if kernel is None:
            kernel = self.compile(*args, **kwargs)
            self._kernel_cache[key] = kernel

        # eager mode: execute kernel immediately and return result
        # lazy mode: return kernel object for manual invocation
        if self.mode == "eager":
            return kernel(*kernel_args.values())
        else:
            return kernel


ExecutionBackend = Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"]


@overload
def jit(func: Callable[_KP, _T]) -> JITImpl[_KP, _KP, _T, _T]: ...


@overload
def jit(
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


def jit(
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    out_idx: list[int] | int | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable[[Callable[_P, _T]], JITImpl[_KP, _KP, _T, _T]]:
    """
    JIT compiler decorator for TileLang functions.

    Supports two execution modes (automatically inferred):
    - **lazy**: Function returns PrimFunc explicitly. Returns compiled kernel object.
    - **eager**: Function uses DSL builder pattern. Executes kernel immediately.

    Parameters
    ----------
    out_idx : list[int] | int | None
        Output tensor index(es). Only supported in lazy mode.
    target : str | Target | None
        TVM compilation target (e.g., "cuda", "llvm", "auto").
    target_host : str | Target | None
        Host target for cross-compilation.
    execution_backend : ExecutionBackend | None
        Backend for kernel execution.
    verbose : bool | None
        Enable verbose compilation output.
    pass_configs : dict[str, Any] | None
        TVM pass configuration options.
    debug_root_path : str | None
        Directory to save compiled kernel source for debugging.
    compile_flags : list[str] | str | None
        Additional compiler flags.
    """

    compile_args = dict(
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        debug_root_path=debug_root_path,
        compile_flags=compile_flags,
    )

    def decorator(func: Callable[_P, _T]):
        mode = "auto"
        pf: JITFunc[_P, _T] = prim_func(func, eager_jit=True)
        func_source = inspect.getsource(pf.orig_func)
        signature = inspect.signature(pf.orig_func)

        return JITImpl(
            func=pf,
            **compile_args,
            func_source=func_source,
            signature=signature,
            mode=mode,
        )

    return decorator(func) if func is not None else decorator
