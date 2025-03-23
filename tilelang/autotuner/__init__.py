# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The auto-tune module for tilelang programs."""

import tilelang
from tilelang import tvm as tvm
import inspect
from functools import wraps
from typing import Any, Callable, List, Literal
from tqdm import tqdm
import logging
from dataclasses import dataclass
import concurrent.futures
import os
from functools import partial

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='autotuner.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s')


@dataclass(frozen=True)
class JITContext:
    out_idx: List[int]
    supply_type: tilelang.TensorSupplyType
    ref_prog: Callable
    rtol: float
    atol: float
    max_mismatched_ratio: float
    skip_check: bool
    profiler: tilelang.Profiler
    target: Literal['cuda', 'hip']


class Autotuner:

    def __init__(
        self,
        fn: Callable,
        configs: Any,
        keys: List[str],
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 30,
    ):
        self.fn = fn
        self.configs = configs
        self.keys = keys
        self.warmup = warmup
        self.rep = rep
        self.timeout = timeout

        # Precompute cached variables
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None

    def jit_compile(self, config_arg) -> JITContext:
        jit_context = self.fn(*config_arg)
        return jit_context

    def run(self, *args: Any, **kwds: Any) -> Any:
        sig = inspect.signature(self.fn)
        bound_args = sig.bind(*args, **kwds)
        bound_args.apply_defaults()

        best_latency = 1e8
        best_config = None

        def target_fn(jit_context):
            # Unpack the context
            profiler = jit_context.profiler
            skip_check = jit_context.skip_check
            ref_prog = jit_context.ref_prog
            rtol = jit_context.rtol
            atol = jit_context.atol
            max_mismatched_ratio = jit_context.max_mismatched_ratio

            self.jit_input_tensors = profiler._get_inputs(
                with_output=profiler ==
                "tvm") if self.jit_input_tensors is None else self.jit_input_tensors

            if (not skip_check) and (ref_prog is not None):
                profiler.assert_allclose(
                    ref_prog, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio)

            latency = profiler.do_bench(
                profiler.func,
                n_warmup=self.warmup,
                n_repeat=self.rep,
                input_tensors=self.jit_input_tensors)
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = profiler._get_inputs(
                    with_output=False) if self.ref_input_tensors is None else self.ref_input_tensors
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog,
                    n_warmup=self.warmup,
                    n_repeat=self.rep,
                    input_tensors=self.ref_input_tensors)

            return latency, self.ref_latency_cache

        # Parallel compilation
        config_args = []

        for config in self.configs:
            new_args = []
            for name, value in bound_args.arguments.items():
                if name not in self.keys:
                    new_args.append(value)
                else:
                    new_args.append(config[name])
            new_args = tuple(new_args)
            config_args.append(new_args)

        worker = partial(self.jit_compile, **kwds)

        # 90% utilization
        num_workers = max(1, int(os.cpu_count() * 0.9))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

        # Submit all compilation jobs
        futures = []
        future_to_index = {}  # Track which future corresponds to which config
        for i, config_arg in enumerate(config_args):
            future = pool.submit(worker, config_arg)
            futures.append(future)
            future_to_index[future] = i

        # Process results with error handling
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
            except Exception:
                logger.debug(f"Compilation failed for config {config} at index {idx}")
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

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")

        pool.shutdown()
        return best_latency, best_config, ref_latency

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)


def autotune(configs: Any,
             keys: List[str],
             warmup: int = 25,
             rep: int = 100,
             timeout: int = 100) -> Callable:
    """
    Decorator for tilelang program
    """

    def decorator(fn: Callable) -> Autotuner:
        return Autotuner(fn, configs=configs, keys=keys, warmup=warmup, rep=rep, timeout=timeout)

    return decorator


def jit(out_idx: List[int],
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Normal,
        ref_prog: Callable = None,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        max_mismatched_ratio: float = 0.01,
        skip_check: bool = False,
        target: Literal['auto', 'cuda', 'hip'] = 'auto') -> Callable:

    def wrapper(fn: Callable):

        @wraps(fn)
        def decorator(*args, **kwargs) -> float:

            kernel = tilelang.compile(fn(*args, **kwargs), out_idx=out_idx, target=target)

            profiler = kernel.get_profiler()

            return JITContext(
                out_idx=out_idx,
                supply_type=supply_type,
                ref_prog=ref_prog,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                skip_check=skip_check,
                profiler=profiler,
                target=target)

        return decorator

    return wrapper
