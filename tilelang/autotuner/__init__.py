# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The auto-tune module for tilelang programs.

This module provides functionality for auto-tuning tilelang programs, including JIT compilation
and performance optimization through configuration search.
"""

import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target
import inspect
from functools import partial
from typing import (Callable, List, Literal, Any, Optional, Union, Dict, overload, Tuple)
from tqdm import tqdm
import logging
import functools
import concurrent.futures
import torch
import os
import sys
import signal
import json
import hashlib
import threading
from pathlib import Path

from tilelang.env import TILELANG_CACHE_DIR, is_cache_enabled
from tilelang.autotuner.param import CompileArgs, ProfileArgs, AutotuneResult
from tilelang.jit.param import _P, _RProg
from tilelang.version import __version__


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result


# Configure logging for the autotuner module
# TODO: Consider creating a common logger in utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Lazy handler initialization flag
_logger_handlers_initialized = False


def _init_logger_handlers():
    global _logger_handlers_initialized
    if _logger_handlers_initialized:
        return
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    file_handler = logging.FileHandler('autotuner.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    _logger_handlers_initialized = True


def get_available_cpu_count() -> int:
    """Gets the number of CPU cores available to the current process.
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()

    return cpu_count


class AutoTuner:
    """Auto-tuner for tilelang programs.

    This class handles the auto-tuning process by testing different configurations
    and finding the optimal parameters for program execution.

    Args:
        fn: The function to be auto-tuned.
        configs: List of configurations to try during auto-tuning.
    """
    compile_args = CompileArgs()
    profile_args = ProfileArgs()

    _lock = threading.Lock()  # For thread safety
    _memory_cache = {}  # In-memory cache dictionary
    cache_dir: Path = Path(TILELANG_CACHE_DIR)

    def __init__(self, fn: Callable, configs):
        self.fn = fn
        self.configs = configs
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None
        self.jit_compile = None

    @classmethod
    def from_kernel(cls, kernel: Callable, configs):
        """Create an AutoTuner instance from a kernel function.

        Args:
            kernel: The kernel function to auto-tune.
            configs: List of configurations to try.

        Returns:
            AutoTuner: A new AutoTuner instance.
        """
        return cls(kernel, configs)

    def set_compile_args(self,
                         out_idx: Union[List[int], int, None] = None,
                         target: Literal['auto', 'cuda', 'hip'] = 'auto',
                         execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                         target_host: Union[str, Target] = None,
                         verbose: bool = False,
                         pass_configs: Optional[Dict[str, Any]] = None):
        """Set compilation arguments for the auto-tuner.

        Args:
            out_idx: List of output tensor indices.
            target: Target platform.
            execution_backend: Execution backend to use for kernel execution.
            target_host: Target host for cross-compilation.
            verbose: Whether to enable verbose output.
            pass_configs: Additional keyword arguments to pass to the Compiler PassContext.

        Returns:
            AutoTuner: Self for method chaining.
        """
        self.compile_args = CompileArgs(
            out_idx=out_idx,
            target=target,
            execution_backend=execution_backend,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs)

        return self

    def set_profile_args(self,
                         warmup: int = 25,
                         rep: int = 100,
                         timeout: int = 30,
                         supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
                         ref_prog: Callable = None,
                         supply_prog: Callable = None,
                         rtol: float = 1e-2,
                         atol: float = 1e-2,
                         max_mismatched_ratio: float = 0.01,
                         skip_check: bool = False,
                         manual_check_prog: Callable = None,
                         cache_input_tensors: bool = True):
        """Set profiling arguments for the auto-tuner.

        Args:
            supply_type: Type of tensor supply mechanism. Ignored if `supply_prog` is provided.
            ref_prog: Reference program for validation.
            supply_prog: Supply program for input tensors.
            rtol: Relative tolerance for validation.
            atol: Absolute tolerance for validation.
            max_mismatched_ratio: Maximum allowed mismatch ratio.
            skip_check: Whether to skip validation.
            manual_check_prog: Manual check program for validation.
            cache_input_tensors: Whether to cache input tensors.
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutoTuner: Self for method chaining.
        """
        self.profile_args = ProfileArgs(
            supply_type=supply_type,
            ref_prog=ref_prog,
            supply_prog=supply_prog,
            rtol=rtol,
            atol=atol,
            max_mismatched_ratio=max_mismatched_ratio,
            skip_check=skip_check,
            manual_check_prog=manual_check_prog,
            cache_input_tensors=cache_input_tensors,
            warmup=warmup,
            rep=rep,
            timeout=timeout)

        # If a custom `supply_prog` is provided, the profiler's `supply_type` setting
        # becomes ineffective. The custom supply program will be used instead.
        if supply_prog is not None and supply_type != tilelang.TensorSupplyType.Auto:
            logger.warning("Ignoring `supply_type` passed to `set_profile_args` because "
                           "`supply_prog` is not None.")

        return self

    def generate_cache_key(self) -> Optional[AutotuneResult]:
        """Generate a cache key for the auto-tuning process.
        """
        func_source = inspect.getsource(self.fn)
        key_data = {
            "version": __version__,
            "func_source": func_source,
            "configs": self.configs,
            "compile_args": hash(self.compile_args),
            "profile_args": hash(self.profile_args),
        }
        # Sort keys to ensure consistency
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _save_result_to_disk(self, key, result: AutotuneResult):
        result.save_to_disk(self.cache_dir / key)

    def _load_result_from_disk(self, key) -> AutotuneResult:
        result = AutotuneResult.load_from_disk(self.cache_dir / key, self.compile_args)
        return result

    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30):
        """Run the auto-tuning process.

        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()

        key = self.generate_cache_key()
        with self._lock:
            if is_cache_enabled():
                # First check in-memory cache
                if key in self._memory_cache:
                    self.logger.warning("Found kernel in memory cache. For better performance," \
                                        " consider using `@tilelang.autotune` instead of direct AutoTuner.from_kernel.")
                    return self._memory_cache[key]

                # Then check disk cache
                result = self._load_result_from_disk(key)
                if result is not None:
                    # Populate memory cache with disk result
                    self._memory_cache[key] = result
                    return result

        sig = inspect.signature(self.fn)
        parameters = sig.parameters
        best_latency: float = 1e8
        best_config: Optional[Dict[str, Any]] = None
        best_kernel: Optional[tilelang.JITKernel] = None

        def _compile(**config_arg) -> tilelang.JITKernel:
            compile_args = self.compile_args
            return compile_args.compile_program(self.fn(**config_arg))

        if self.jit_compile is None:
            self.jit_compile = _compile

        def target_fn(jit_kernel: tilelang.JITKernel):
            # Unpack the context
            profile_args = self.profile_args
            supply_type = profile_args.supply_type
            skip_check = profile_args.skip_check
            manual_check_prog = profile_args.manual_check_prog
            cache_input_tensors = profile_args.cache_input_tensors
            ref_prog = profile_args.ref_prog
            supply_prog = profile_args.supply_prog
            rtol = profile_args.rtol
            atol = profile_args.atol
            max_mismatched_ratio = profile_args.max_mismatched_ratio

            profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)

            # Factory functions for generating input tensors.
            # This encapsulates the logic of using either a custom supply program (`supply_prog`)
            # or the default profiler input generation (`profiler._get_inputs`).
            def get_input_tensors_supply(with_output: bool):

                def func():
                    if supply_prog is not None:
                        return supply_prog(profiler._get_params(with_output=with_output))
                    else:
                        return profiler._get_inputs(with_output=with_output)

                return func

            jit_input_tensors_supply = get_input_tensors_supply(with_output=False)
            ref_input_tensors_supply = get_input_tensors_supply(with_output=False)

            if cache_input_tensors:
                if supply_prog is not None:
                    logger.warning(
                        "Incompatible input tensor properties detected between cached tensors and "
                        "tensors regenerated for the current configuration trial. "
                        "This can happen if different tuning configurations require different input shapes/dtypes "
                        "and input tensor caching is enabled.\n"
                        "To ensure fresh, compatible inputs are generated for every trial "
                        "you can disable caching by setting:\n"
                        "  `cache_input_tensors=False`\n"
                        "within your `.set_compile_args(...)` call.\n")
                self.jit_input_tensors = jit_input_tensors_supply(
                ) if self.jit_input_tensors is None else self.jit_input_tensors
            else:
                self.jit_input_tensors = jit_input_tensors_supply()

            if (not skip_check) and (ref_prog is not None):
                if manual_check_prog is not None:
                    profiler.manual_assert_close(
                        ref_prog,
                        input_tensors=self.jit_input_tensors,
                        manual_check_prog=manual_check_prog)
                else:
                    profiler.assert_allclose(
                        ref_prog,
                        input_tensors=self.jit_input_tensors,
                        rtol=rtol,
                        atol=atol,
                        max_mismatched_ratio=max_mismatched_ratio)
            latency = profiler.do_bench(
                warmup=warmup, rep=rep, input_tensors=self.jit_input_tensors)
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = ref_input_tensors_supply()
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog, n_warmup=warmup, n_repeat=rep, input_tensors=self.ref_input_tensors)

            return latency, self.ref_latency_cache

        config_args = []
        for config in self.configs:
            new_kwargs = {}
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            config_args.append(new_kwargs)

        num_workers = max(1, int(get_available_cpu_count() * 0.9))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = []
        future_to_index = {}

        def device_wrapper(func, device, **config_arg):
            torch.cuda.set_device(device)
            return func(**config_arg)

        for i, config_arg in enumerate(config_args):
            future = pool.submit(
                functools.partial(device_wrapper, self.jit_compile, torch.cuda.current_device()),
                **config_arg,
            )
            futures.append(future)
            future_to_index[future] = i

        results_with_configs = []
        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Compiling configurations"):
            idx = future_to_index[future]
            config = config_args[idx]
            try:
                result = future.result()
                results_with_configs.append((result, config))
            except Exception as e:
                logger.debug(
                    f"Compilation failed for config {config} at index {idx} with error: {e}")
                continue

        ref_latency = None
        progress_bar = tqdm(range(len(results_with_configs)), desc="Bench configurations")
        for i in progress_bar:
            jit_kernel, config = results_with_configs[i]
            try:
                # Cannot ThreadPoolExecutor to enforce timeout on target_fn execution
                # Because tma init may behave strangely with one thread
                # latency, ref_latency = target_fn(jit_kernel)
                latency, ref_latency = run_with_timeout(target_fn, timeout, jit_kernel)
            except TimeoutException:
                logger.info(
                    f"A timeout occurred while testing config {config}, checkout autotuner.log for more details"
                )
                continue
            except Exception as e:
                logger.info(
                    f"An error occurred while testing config {config}, checkout autotuner.log for more details"
                )
                logger.debug(f"Error: {e}")
                continue

            logging.debug(f"Config {config} latency: {latency} at index {i}")

            if latency < best_latency:
                best_latency = latency
                best_config = config
                best_kernel = jit_kernel

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")

        pool.shutdown()

        if best_kernel is None:
            error_msg = ("Auto-tuning failed: No configuration successfully "
                         "compiled and passed benchmarking/validation.")
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        best_kernel: tilelang.JITKernel = best_kernel.update_tuner_result(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
        )

        autotuner_result = AutotuneResult(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
            libcode=best_kernel.get_kernel_source(),
            func=best_kernel.prim_func,
            kernel=best_kernel)

        if self.compile_args.execution_backend == "dlpack":
            logger.warning("DLPack backend does not support cache saving to disk.")
        else:
            with self._lock:
                if is_cache_enabled():
                    self._save_result_to_disk(key, autotuner_result)

        self._memory_cache[key] = autotuner_result

        return autotuner_result

    def __call__(self) -> Any:
        """Make the AutoTuner callable, running the auto-tuning process.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        return self.run()


class _AutoTunerImplementation:
    # Overload __init__ to help type checkers understand the effect of return_program
    # The '-> None' is for __init__ itself. The crucial part is Literal for return_program.

    warmup: int = 25
    rep: int = 100
    timeout: int = 100
    configs: Any = None

    def __init__(self, configs: Any, warmup: int = 25, rep: int = 100, timeout: int = 100) -> None:
        """Initialize the AutoTunerImplementation.

        Args:
            configs: Configuration space to explore during auto-tuning.
            warmup: Number of warmup iterations before timing.
            rep: Number of repetitions for timing measurements.
            timeout: Maximum time (in seconds) allowed for each configuration.
        """
        self.configs = configs
        self.warmup = warmup
        self.rep = rep
        self.timeout = timeout

        self._tuner_cache: Dict[tuple, tilelang.JITKernel] = {}

    # This tells the type checker what the *wrapper* function will return.
    # this is for linting, please do not remove it.
    @overload
    def __call__(self, fn: Callable[_P, _RProg]) -> Callable[_P, Tuple[_RProg, AutotuneResult]]:
        ...

    @overload
    def __call__(self, fn: Callable[_P, _RProg]) -> Callable[_P, AutotuneResult]:
        ...

    # Actual implementation of __call__
    def __call__(self, fn: Callable[_P, _RProg]) -> Callable[_P, Any]:
        warmup = self.warmup
        rep = self.rep
        timeout = self.timeout
        configs = self.configs

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            key_args_tuple = args
            key_kwargs_tuple = tuple(sorted(kwargs.items()))
            key = (key_args_tuple, key_kwargs_tuple)

            if key not in self._tuner_cache:

                def jit_compile(**config_arg):
                    return fn(*args, **kwargs, __tune_params=config_arg)

                autotuner = AutoTuner(fn, configs=configs)
                autotuner.jit_compile = jit_compile
                autotuner.run = partial(autotuner.run, warmup, rep, timeout)

                artifact = autotuner.run()
                self._tuner_cache[key] = artifact.kernel

            return self._tuner_cache[key]

        return wrapper


def autotune(  # This is the new public interface
        func: Union[Callable[_P, _RProg], PrimFunc, None] = None,
        *,  # Indicates subsequent arguments are keyword-only
        configs: Any,
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 100):
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
        # Case 1: Used as @autotune (func_or_out_idx is the function, others are defaults)
        # This is a placeholder for a real auto tuner implementation
        raise ValueError(
            "Use tilelang.autotune to decorate func without arguments is not supported yet.")
    elif isinstance(func, PrimFunc):
        raise ValueError("Use tilelang.jit to decorate prim_func is not supported yet.")
    else:
        # Case 2: Used as @autotune(...) to configure, or func_or_out_idx is meant as out_idx.
        # Create a _AutoTunerImplementation instance with the provided/defaulted arguments.
        # This instance is a decorator that will be applied to the function later.
        configured_decorator = _AutoTunerImplementation(
            configs=configs, warmup=warmup, rep=rep, timeout=timeout)
        return configured_decorator
