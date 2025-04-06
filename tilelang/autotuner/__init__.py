# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The auto-tune module for tilelang programs.

This module provides functionality for auto-tuning tilelang programs, including JIT compilation
and performance optimization through configuration search.
"""

import tilelang
from tilelang import tvm as tvm
import inspect
from functools import wraps, partial
from typing import Callable, List, Literal, Any, Optional
from tqdm import tqdm
import logging
from dataclasses import dataclass
import concurrent.futures
import torch
import os
import sys

# Configure logging for the autotuner module
# TODO: Consider creating a common logger in utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')

file_handler = logging.FileHandler('autotuner.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


@dataclass(frozen=True)
class JITContext:
    """Context object for Just-In-Time compilation settings.

    Attributes:
        out_idx: List of output tensor indices.
        ref_prog: Reference program for correctness validation.
        supply_prog: Supply program for input tensors.
        rtol: Relative tolerance for output validation.
        atol: Absolute tolerance for output validation.
        max_mismatched_ratio: Maximum allowed ratio of mismatched elements.
        skip_check: Whether to skip validation checks.
        cache_input_tensors: Whether to cache input tensors for each compilation.
        kernel: JITKernel instance for performance measurement.
        supply_type: Type of tensor supply mechanism.
        target: Target platform ('cuda' or 'hip').
    """
    out_idx: List[int]
    ref_prog: Callable
    supply_prog: Callable
    rtol: float
    atol: float
    max_mismatched_ratio: float
    skip_check: bool
    cache_input_tensors: bool
    kernel: tilelang.JITKernel
    supply_type: tilelang.TensorSupplyType
    target: Literal['cuda', 'hip']


@dataclass(frozen=True)
class AutotuneResult:
    """Results from auto-tuning process.

    Attributes:
        latency: Best achieved execution latency.
        config: Configuration that produced the best result.
        ref_latency: Reference implementation latency.
        libcode: Generated library code.
        func: Optimized function.
        kernel: Compiled kernel function.
    """
    latency: float
    config: dict
    ref_latency: float
    libcode: str
    func: Callable
    kernel: Callable


class AutoTuner:
    """Auto-tuner for tilelang programs.

    This class handles the auto-tuning process by testing different configurations
    and finding the optimal parameters for program execution.

    Args:
        fn: The function to be auto-tuned.
        configs: List of configurations to try during auto-tuning.
    """

    def __init__(self, fn: Callable, configs):
        self.fn = fn
        self.configs = configs
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None

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
                         out_idx: List[int],
                         supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
                         ref_prog: Callable = None,
                         supply_prog: Callable = None,
                         rtol: float = 1e-2,
                         atol: float = 1e-2,
                         max_mismatched_ratio: float = 0.01,
                         skip_check: bool = False,
                         cache_input_tensors: bool = True,
                         target: Literal['auto', 'cuda', 'hip'] = 'auto'):
        """Set compilation arguments for the auto-tuner.

        Args:
            out_idx: List of output tensor indices.
            supply_type: Type of tensor supply mechanism. Ignored if `supply_prog` is provided.
            ref_prog: Reference program for validation.
            supply_prog: Supply program for input tensors.
            rtol: Relative tolerance for validation.
            atol: Absolute tolerance for validation.
            max_mismatched_ratio: Maximum allowed mismatch ratio.
            skip_check: Whether to skip validation.
            cache_input_tensors: Whether to cache input tensors.
            target: Target platform.

        Returns:
            AutoTuner: Self for method chaining.
        """

        # If a custom `supply_prog`` is provided, the profiler's `supply_type` setting
        # becomes ineffective. The custom supply program will be used instead.
        if ref_prog is not None and supply_type != tilelang.TensorSupplyType.Auto:
            logger.warning("Ignoring `supply_type` passed to `set_compile_args` because "
                           "`ref_prog` is not None.")

        def _compile(*config_arg):
            kernel = tilelang.compile(self.fn(*config_arg), out_idx=out_idx, target=target)
            jit_context = JITContext(
                out_idx=out_idx,
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                skip_check=skip_check,
                cache_input_tensors=cache_input_tensors,
                kernel=kernel,
                supply_type=supply_type,
                target=target)
            return jit_context

        self.jit_compile = _compile
        return self

    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 100):
        """Run the auto-tuning process.

        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        sig = inspect.signature(self.fn)
        keys = list(sig.parameters.keys())
        bound_args = sig.bind()
        bound_args.apply_defaults()
        best_latency = 1e8
        best_config = None
        best_jit_context = None

        def target_fn(jit_context: JITContext):
            # Unpack the context
            kernel = jit_context.kernel
            supply_type = jit_context.supply_type
            skip_check = jit_context.skip_check
            cache_input_tensors = jit_context.cache_input_tensors
            ref_prog = jit_context.ref_prog
            supply_prog = jit_context.supply_prog
            rtol = jit_context.rtol
            atol = jit_context.atol
            max_mismatched_ratio = jit_context.max_mismatched_ratio

            profiler = kernel.get_profiler(tensor_supply_type=supply_type)

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

            jit_input_tensors_supply = get_input_tensors_supply(with_output=(profiler == "tvm"))
            ref_input_tensors_supply = get_input_tensors_supply(with_output=False)

            if cache_input_tensors:
                jit_input_tensors = jit_input_tensors_supply()
                if self.jit_input_tensors is not None:
                    if not check_tensor_list_compatibility(self.jit_input_tensors,
                                                           jit_input_tensors):
                        logger.warning(
                            "Incompatible input tensor properties detected between cached tensors and "
                            "tensors regenerated for the current configuration trial. "
                            "This can happen if different tuning configurations require different input shapes/dtypes "
                            "and input tensor caching is enabled.\n"
                            "To ensure fresh, compatible inputs are generated for every trial "
                            "you can disable caching by setting:\n"
                            "  `cache_input_tensors=False`\n"
                            "within your `.set_compile_args(...)` call.\n")
                    self.jit_input_tensors = jit_input_tensors
                self.jit_input_tensors = jit_input_tensors
            else:
                self.jit_input_tensors = jit_input_tensors_supply()

            if (not skip_check) and (ref_prog is not None):
                profiler.assert_allclose(
                    ref_prog,
                    input_tensors=self.jit_input_tensors,
                    rtol=rtol,
                    atol=atol,
                    max_mismatched_ratio=max_mismatched_ratio)

            latency = profiler.do_bench(
                profiler.func, n_warmup=warmup, n_repeat=rep, input_tensors=self.jit_input_tensors)
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = ref_input_tensors_supply()
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog, n_warmup=warmup, n_repeat=rep, input_tensors=self.ref_input_tensors)

            return latency, self.ref_latency_cache

        config_args = []
        for config in self.configs:
            new_args = []
            for name, value in bound_args.arguments.items():
                if name not in keys:
                    new_args.append(value)
                else:
                    if name not in config:
                        raise ValueError(f"Configuration {config} does not contain key {name}")
                    new_args.append(config[name])
            new_args = tuple(new_args)
            config_args.append(new_args)

        num_workers = max(1, int(os.cpu_count() * 0.9))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = []
        future_to_index = {}
        for i, config_arg in enumerate(config_args):
            future = pool.submit(
                self.jit_compile,
                *config_arg,
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
            jit_context, config = results_with_configs[i]
            try:
                # Cannot ThreadPoolExecutor to enforce timeout on target_fn execution
                # Because tma init may behave strangely with one thread
                latency, ref_latency = target_fn(jit_context)
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
                best_jit_context = jit_context

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")

        pool.shutdown()

        if best_jit_context is None:
            error_msg = ("Auto-tuning failed: No configuration successfully "
                         "compiled and passed benchmarking/validation.")
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return AutotuneResult(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
            libcode=best_jit_context.kernel.get_kernel_source(),
            func=self.fn(*best_config),
            kernel=best_jit_context.kernel)

    def __call__(self) -> Any:
        """Make the AutoTuner callable, running the auto-tuning process.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        return self.run()


def autotune(configs: Any, warmup: int = 25, rep: int = 100, timeout: int = 100) -> AutotuneResult:
    """Decorator for auto-tuning tilelang programs.

    Args:
        configs: Configuration space to explore during auto-tuning.
        warmup: Number of warmup iterations before timing.
        rep: Number of repetitions for timing measurements.
        timeout: Maximum time (in seconds) allowed for each configuration.

    Returns:
        Callable: Decorated function that performs auto-tuning.
    """

    def decorator(fn: Callable) -> AutoTuner:
        autotuner = AutoTuner(fn, configs=configs)
        autotuner.jit_compile = fn
        autotuner.run = partial(autotuner.run, warmup, rep, timeout)
        return autotuner

    return decorator


def jit(out_idx: Optional[List[int]] = None,
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
        ref_prog: Callable = None,
        supply_prog: Callable = None,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        max_mismatched_ratio: float = 0.01,
        skip_check: bool = False,
        cache_input_tensors: bool = True,
        target: Literal['auto', 'cuda', 'hip'] = 'auto') -> Callable:
    """Just-In-Time compilation decorator for tilelang programs.

    Args:
        out_idx: List of output tensor indices.
        supply_type: Type of tensor supply mechanism. Ignored if `supply_prog` is provided.
        ref_prog: Reference program for correctness validation.
        supply_prog: Supply program for input tensors.
        rtol: Relative tolerance for output validation.
        atol: Absolute tolerance for output validation.
        max_mismatched_ratio: Maximum allowed ratio of mismatched elements.
        skip_check: Whether to skip validation checks.
        cache_input_tensors: Whether to cache input tensors for each compilation.
        target: Target platform ('auto', 'cuda', or 'hip').

    Returns:
        Callable: Decorated function that performs JIT compilation.
    """

    # If a custom `supply_prog`` is provided, the profiler's `supply_type` setting
    # becomes ineffective. The custom supply program will be used instead.
    if supply_prog is not None and supply_type != tilelang.TensorSupplyType.Auto:
        logger.warning("Ignoring `supply_type` passed to `autotune.jit` because "
                       "`supply_prog` is not None.")

    def wrapper(fn: Callable):

        @wraps(fn)
        def decorator(*args, **kwargs) -> float:

            kernel = tilelang.compile(fn(*args, **kwargs), out_idx=out_idx, target=target)

            return JITContext(
                out_idx=out_idx,
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                skip_check=skip_check,
                cache_input_tensors=cache_input_tensors,
                kernel=kernel,
                supply_type=supply_type,
                target=target)

        return decorator

    return wrapper


def check_tensor_list_compatibility(
    list1: List[torch.Tensor],
    list2: List[torch.Tensor],
) -> bool:
    """Checks if two lists of tensors are compatible.
    
    Compatibility checks performed include:
    1. Lists have the same length.
    2. Corresponding tensors have the same shape.

    Args:
        list1: First list of tensors.
        list2: Second list of tensors.
    """
    if len(list1) != len(list2):
        return False

    return all(tensor1.shape == tensor2.shape for tensor1, tensor2 in zip(list1, list2))
