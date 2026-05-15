"""The auto-tune module for tilelang programs.

This module provides functionality for auto-tuning tilelang programs, including JIT compilation
and performance optimization through configuration search.
"""

from __future__ import annotations
from dataclasses import dataclass

import tilelang
from tilelang import tvm as tvm
from tilelang import env
from tilelang.jit import JITImpl
from tilelang.jit.kernel import JITKernel
from tvm.tir import PrimFunc, Var
from tvm.target import Target
import inspect
from functools import partial
from typing import Callable, Generic, Literal, Any, TypeVar

# Python 3.9 compatibility for ParamSpec
try:
    from typing import ParamSpec
except ImportError:  # Python < 3.10
    from typing_extensions import ParamSpec
from tqdm.auto import tqdm
import logging
import concurrent.futures
import queue
import torch
import os
import sys
import json
import hashlib
import signal
import threading
import traceback
from pathlib import Path

from tilelang.autotuner.param import CompileArgs, ProfileArgs, AutotuneResult
from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
from tilelang.utils.language import get_prim_func_name
from tilelang.autotuner.capture import get_autotune_inputs
from tilelang.utils.target import determine_target
from tilelang import __version__


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
    raise TimeoutException("Operation timed out")


def _run_with_sigalrm(func, timeout, *args, **kwargs):
    """POSIX path: ``SIGALRM`` interrupts ``func`` synchronously inside the
    same thread. The runaway call actually stops, releasing CPU/GPU resources
    held by e.g. a stuck CUDA benchmark.
    """
    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout))
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def _async_raise_in_thread(tid: int, exc_type: type) -> bool:
    """Asynchronously raise ``exc_type`` in the thread identified by ``tid``.

    Uses CPython's :c:func:`PyThreadState_SetAsyncExc` — the same primitive the
    interpreter itself uses to deliver ``KeyboardInterrupt`` to the main thread,
    and the Python-level analogue of POSIX ``SIGALRM``. This is the strongest
    cross-thread cancellation available without breaking process invariants:

    * ``TerminateThread`` (WINAPI) is rejected on purpose — Microsoft's own
      documentation labels it dangerous because it leaves CRT locks, DLL
      thread-detach callbacks, and (relevant here) CUDA driver state in an
      inconsistent state, which can corrupt subsequent kernels.
    * ``PyThreadState_SetAsyncExc`` only fires at Python bytecode boundaries,
      so a thread stuck in a C call (e.g. ``cudaStreamSynchronize``) won't
      respond until it returns to Python. POSIX ``SIGALRM`` has the same
      practical limitation.

    Returns ``True`` when the exception was queued, ``False`` otherwise.
    """
    import ctypes

    set_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
    set_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)
    set_exc.restype = ctypes.c_int

    affected = set_exc(ctypes.c_ulong(tid), ctypes.py_object(exc_type))
    if affected == 0:
        return False
    if affected > 1:
        # If more than one thread was affected (should not happen with a
        # single tid), undo the side-effect to avoid leaving threads with a
        # pending async exception.
        set_exc(ctypes.c_ulong(tid), None)
        return False
    return True


def _run_with_thread(func, timeout, *args, **kwargs):
    """Cross-platform fallback when ``SIGALRM`` is unavailable (Windows) or
    when called outside the main thread (``signal.signal`` is restricted to
    the main thread on POSIX).

    Mirrors POSIX ``SIGALRM`` semantics as closely as Python allows by running
    ``func`` **on the current thread** while a daemon watchdog thread sleeps
    until the deadline. On timeout the watchdog uses
    ``PyThreadState_SetAsyncExc`` to inject :class:`TimeoutException` back into
    the caller, which surfaces at the next Python bytecode boundary.

    Crucially, this preserves ``threading.local()``-backed state (e.g.
    ``set_autotune_inputs``'s capture stack) — moving ``func`` onto a worker
    thread would silently zero out that state and break callers that read it.
    """
    target_tid = threading.get_ident()
    cancel = threading.Event()
    fired = threading.Event()

    def _watchdog():
        if cancel.wait(timeout):
            # ``func`` finished before the deadline — nothing to do.
            return
        if _async_raise_in_thread(target_tid, TimeoutException):
            fired.set()

    watchdog = threading.Thread(target=_watchdog, daemon=True, name="tl-tuner-watchdog")
    watchdog.start()
    try:
        return func(*args, **kwargs)
    except TimeoutException:
        # Re-raise with the standard message regardless of whether the async
        # exception we injected won the race against an in-flight Python
        # exception in ``func``.
        raise TimeoutException(f"Operation timed out after {timeout}s") from None
    finally:
        # Stop the watchdog. If we got here without timing out the watchdog is
        # blocked in ``cancel.wait`` and will exit immediately. If the watchdog
        # already fired, ``func`` raised TimeoutException above (caught and
        # re-raised) — but a stray async exception may still be queued, so
        # clear it defensively before the caller continues running Python code.
        cancel.set()
        if fired.is_set():
            import ctypes

            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(target_tid), ctypes.py_object(0))


def run_with_timeout(func, timeout, *args, **kwargs):
    """Run ``func`` under a deadline, raising :class:`TimeoutException` if exceeded.

    Picks the strongest available primitive:

    * **POSIX, main thread**: ``SIGALRM``. The deadline truly interrupts
      ``func`` mid-execution and frees its resources.
    * **Otherwise** (Windows, or POSIX off the main thread): a daemon
      ``ThreadPoolExecutor``. The deadline only unblocks the caller; the
      runaway worker keeps running until it returns on its own.
    """
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        return _run_with_sigalrm(func, timeout, *args, **kwargs)
    return _run_with_thread(func, timeout, *args, **kwargs)


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
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    file_handler = logging.FileHandler("autotuner.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    _logger_handlers_initialized = True


def get_available_cpu_count() -> int:
    """Gets the number of CPU cores available to the current process."""
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()

    return cpu_count or 1


def _normalize_value(value, sort_dict_items: bool = False):
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), value.stride())
    if isinstance(value, Var):
        return str(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(v, sort_dict_items=sort_dict_items) for v in value)
    if isinstance(value, dict):
        items = ((str(k), _normalize_value(v, sort_dict_items=sort_dict_items)) for k, v in value.items())
        if sort_dict_items:
            return tuple(sorted(items))
        return {k: v for k, v in items}
    return value


@dataclass
class _BenchmarkWorkerState:
    jit_input_tensors: Any = None
    ref_input_tensors: Any = None
    ref_latency_cache: float | None = None


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

    _kernel_parameters: tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]] | None = None
    _function_parameters: dict[str, Any] | None = None
    _lock = threading.Lock()  # For thread safety
    _memory_cache = {}  # In-memory cache dictionary

    def __init__(self, fn: Callable, configs):
        self.fn = fn
        self.configs = configs
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None
        self.jit_compile = None
        self.jit_elaborate = None

    @classmethod
    def _get_cache_dir(cls) -> Path:
        from tilelang.cache.kernel_cache import KernelCache

        return Path(KernelCache._get_namespace_root()) / "autotuner"

    @property
    def cache_dir(self) -> Path:
        return self._get_cache_dir()

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

    def set_compile_args(
        self,
        out_idx: list[int] | int | None = None,
        target: Literal["auto", "cuda", "hip", "metal"] | None = None,
        execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch"] | None = None,
        target_host: str | Target | None = None,
        verbose: bool | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        """Set compilation arguments for the auto-tuner.

        Args:
            out_idx: List of output tensor indices.
            target: Target platform. If None, reads from TILELANG_TARGET environment variable (defaults to "auto").
            execution_backend: Execution backend to use for kernel execution. If None, reads from
                TILELANG_EXECUTION_BACKEND environment variable (defaults to "auto").
            target_host: Target host for cross-compilation.
            verbose: Whether to enable verbose output. If None, reads from
                TILELANG_VERBOSE environment variable (defaults to False).
            pass_configs: Additional keyword arguments to pass to the Compiler PassContext.

        Environment Variables:
            TILELANG_TARGET: Default compilation target (e.g., "cuda", "llvm"). Defaults to "auto".
            TILELANG_EXECUTION_BACKEND: Default execution backend. Defaults to "auto".
            TILELANG_VERBOSE: Set to "1", "true", "yes", or "on" to enable verbose compilation by default.

        Returns:
            AutoTuner: Self for method chaining.
        """
        # Apply environment variable defaults if parameters are not explicitly set
        if target is None:
            target = env.get_default_target()
        if execution_backend is None:
            execution_backend = env.get_default_execution_backend()
        if verbose is None:
            verbose = env.get_default_verbose()

        # Normalize target to a concrete TVM Target and resolve execution backend
        t = Target(determine_target(target))
        from tilelang.jit.execution_backend import resolve_execution_backend

        resolved_backend = resolve_execution_backend(execution_backend, t)

        self.compile_args = CompileArgs(
            out_idx=out_idx,
            target=t,
            execution_backend=resolved_backend,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
        )

        return self

    def set_profile_args(
        self,
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
        cache_input_tensors: bool = False,
        backend: Literal["event", "cupti", "cudagraph"] = "event",
    ):
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
            backend: Profiler backend - "event" (CUDA events), "cupti", or "cudagraph".
        Returns:
            AutoTuner: Self for method chaining.
        """
        # If the program is under `with set_autotune_inputs` context,
        # freeze captured tensors now so benchmark worker threads do not
        # lose them via thread-local storage lookups.
        captured_inputs = get_autotune_inputs()
        if captured_inputs is not None:
            if supply_prog is not None:
                logger.warning("`supply_prog` will be ignored as this program is under `with set_autotune_inputs` context.")
            frozen_inputs = list(captured_inputs)
            cached_tensors_by_device = {}

            def supply_prog(device, _frozen_inputs=frozen_inputs, _cached_tensors_by_device=cached_tensors_by_device):
                if not isinstance(device, (int, str, torch.device)):
                    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                if device not in _cached_tensors_by_device:
                    if isinstance(device, torch.device):
                        target_device = device
                    elif isinstance(device, str):
                        target_device = torch.device(device)
                    else:
                        target_device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
                    _cached_tensors_by_device[device] = [
                        tensor.to(device=target_device).clone() if isinstance(tensor, torch.Tensor) else tensor for tensor in _frozen_inputs
                    ]
                return _cached_tensors_by_device[device]

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
            timeout=timeout,
            backend=backend,
        )

        # If a custom `supply_prog` is provided, the profiler's `supply_type` setting
        # becomes ineffective. The custom supply program will be used instead.
        if supply_prog is not None and supply_type != tilelang.TensorSupplyType.Auto:
            logger.warning("Ignoring `supply_type` passed to `set_profile_args` because `supply_prog` is not None.")

        return self

    def set_kernel_parameters(self, k_parameters: tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]], f_parameters: dict[str, Any]):
        # for cache key generation
        self._kernel_parameters = k_parameters
        self._function_parameters = f_parameters

    def generate_cache_key(self, parameters: dict[str, Any], extra_parameters: dict[str, Any]) -> AutotuneResult | None:
        """Generate a cache key for the auto-tuning process."""

        # extract parameters from the function signature
        op_parameters = []
        for _, default_value in parameters.items():
            if default_value.default is not inspect.Parameter.empty:
                op_parameters.append(default_value.default)

        if self._kernel_parameters is not None:
            op_parameters += _normalize_value(self._kernel_parameters)

        func_source = inspect.getsource(self.fn)
        key_data = {
            "version": __version__,
            "op_parameters": tuple(op_parameters),
            "extra_parameters": extra_parameters,
            "func_source": func_source,
            "configs": self.configs,
            "compile_args": hash(self.compile_args),
            "profile_args": hash(self.profile_args),
        }
        # Sort keys to ensure consistency
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _save_result_to_disk(self, key, result: AutotuneResult):
        result.save_to_disk(self.cache_dir / key, self.compile_args.verbose)

    def _load_result_from_disk(self, key) -> AutotuneResult:
        result = AutotuneResult.load_from_disk(self.cache_dir / key, self.compile_args)
        return result

    # Compile-related helpers
    def _default_compile(
        self,
        **config_arg,
    ) -> tilelang.JITKernel:
        compile_args = self.compile_args
        return compile_args.compile_program(self.fn(**config_arg))

    def _default_elaborate(self, **config_arg) -> PrimFunc:
        return self.fn(**config_arg)

    def _ensure_jit_functions(
        self,
    ) -> tuple[Callable[..., tilelang.JITKernel], Callable[..., PrimFunc]]:
        compile_func = self.jit_compile
        elaborate_func = self.jit_elaborate
        if compile_func is None:
            compile_func = self._default_compile
        if elaborate_func is None:
            elaborate_func = self._default_elaborate
        return compile_func, elaborate_func

    def _resolve_grouped_compile_mode(
        self,
        enable_grouped_compile: bool,
        group_compile_size: int,
    ) -> tuple[str, str, bool, str]:
        target_kind = self.compile_args.target.kind.name if isinstance(self.compile_args.target, Target) else str(self.compile_args.target)
        execution_backend = str(self.compile_args.execution_backend)
        grouped_compile_requested = enable_grouped_compile and group_compile_size > 1
        grouped_compile_active = grouped_compile_requested and target_kind == "cuda" and execution_backend == "tvm_ffi"
        grouped_compile_reason = ""
        if grouped_compile_requested and not grouped_compile_active:
            grouped_compile_reason = (
                f"grouped compilation is currently implemented for CUDA+tvm_ffi only; "
                f"fallback to per-config mode (target={target_kind}, execution_backend={execution_backend})"
            )
            logger.warning("%s", grouped_compile_reason)
        return target_kind, execution_backend, grouped_compile_active, grouped_compile_reason

    def _resolve_num_compile_workers(self) -> int:
        available_cpu_count = get_available_cpu_count()
        cpu_utilizations = float(env.TILELANG_AUTO_TUNING_CPU_UTILITIES)
        cpu_counts = int(env.TILELANG_AUTO_TUNING_CPU_COUNTS)
        max_cpu_count = int(env.TILELANG_AUTO_TUNING_MAX_CPU_COUNT)
        if cpu_counts > 0:
            num_workers = min(cpu_counts, available_cpu_count)
            logger.info(f"Auto-tuning with {cpu_counts} CPU counts, {available_cpu_count} CPUs available, {num_workers} CPUs will be used")
        else:
            num_workers = max(1, int(available_cpu_count * cpu_utilizations))
            logger.info(
                f"Auto-tuning with {cpu_utilizations} CPU utilizations, {available_cpu_count} CPUs available, {num_workers} CPUs will be used"
            )

        if max_cpu_count > 0 and num_workers > max_cpu_count:
            logger.warning(
                f"Auto-tuning with {cpu_utilizations} CPU utilizations, {available_cpu_count} CPUs available, {num_workers} CPUs will be used, but the max CPU count is {max_cpu_count}, so we will use {max_cpu_count} CPUs"
            )
            num_workers = max_cpu_count
        return num_workers

    def _prepare_compile_execution(
        self,
        config_args: list[dict[str, Any]],
        grouped_compile_active: bool,
        group_compile_size: int,
        compile_func: Callable[..., tilelang.JITKernel],
        elaborate_func: Callable[..., PrimFunc],
    ) -> tuple[
        concurrent.futures.ThreadPoolExecutor,
        list[concurrent.futures.Future],
        dict[concurrent.futures.Future, list[tuple[int, dict[str, Any]]]],
        str,
    ]:
        num_workers = self._resolve_num_compile_workers()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures: list[concurrent.futures.Future] = []
        future_to_unit: dict[concurrent.futures.Future, list[tuple[int, dict[str, Any]]]] = {}

        def cuda_device_wrapper(func: Callable[..., Any], device: int):
            def inner(**config_arg):
                torch.cuda.set_device(device)
                return func(**config_arg)

            return inner

        def get_compile_func():
            compile_impl = compile_func
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                compile_impl = cuda_device_wrapper(compile_func, device)
            return compile_impl

        def get_elaborate_func():
            elaborate_impl = elaborate_func
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                elaborate_impl = cuda_device_wrapper(elaborate_func, device)
            return elaborate_impl

        def compile_unit(unit_items: list[tuple[int, dict[str, Any]]]):
            if grouped_compile_active:
                return compile_grouped_unit_tvm_ffi(
                    unit_items=unit_items,
                    compile_args=self.compile_args,
                    elaborate_func=get_elaborate_func(),
                )
            compile_impl = get_compile_func()
            unit_results: list[tuple[int, dict[str, Any], tilelang.JITKernel | None, Exception | None]] = []
            for idx, config_arg in unit_items:
                try:
                    jit_kernel = compile_impl(**config_arg)
                    unit_results.append((idx, config_arg, jit_kernel, None))
                except Exception as e:
                    unit_results.append((idx, config_arg, None, e))
            return unit_results

        compile_units: list[list[tuple[int, dict[str, Any]]]] = []
        if grouped_compile_active:
            for start in range(0, len(config_args), group_compile_size):
                end = min(start + group_compile_size, len(config_args))
                compile_units.append([(i, config_args[i]) for i in range(start, end)])
        else:
            for i, config_arg in enumerate(config_args):
                compile_units.append([(i, config_arg)])

        for unit_items in compile_units:
            future = pool.submit(compile_unit, unit_items)
            futures.append(future)
            future_to_unit[future] = unit_items

        compile_desc = "Compiling configurations (grouped)" if grouped_compile_active else "Compiling configurations"
        return pool, futures, future_to_unit, compile_desc

    # Benchmark-related helpers
    def _benchmark_worker_loop(
        self,
        worker_device: int,
        worker_queue: queue.Queue,
        result_queue: queue.Queue,
        start_event: threading.Event,
        target_kind: str,
        benchmark_target: Callable[..., tuple[float, float | None]],
        timeout: int,
        worker_state: _BenchmarkWorkerState,
    ) -> None:
        if target_kind == "cuda":
            try:
                torch.cuda.set_device(worker_device)
            except Exception:
                logger.warning("Failed to bind benchmark worker to cuda:%s", worker_device)
                logger.debug("Error: %s", traceback.format_exc())

        start_event.wait()
        queue_poll_timeout_s = 0.1
        while True:
            try:
                item = worker_queue.get(timeout=queue_poll_timeout_s)
            except queue.Empty:
                continue
            if item is None:
                break
            jit_kernel, config, idx = item
            try:
                if timeout > 0:
                    call_result_queue: queue.Queue = queue.Queue()
                    call_state = _BenchmarkWorkerState(
                        jit_input_tensors=worker_state.jit_input_tensors,
                        ref_input_tensors=worker_state.ref_input_tensors,
                        ref_latency_cache=worker_state.ref_latency_cache,
                    )

                    def _run_benchmark_target(
                        _jit_kernel: tilelang.JITKernel = jit_kernel,
                        _worker_state: _BenchmarkWorkerState = call_state,
                        _call_result_queue: queue.Queue = call_result_queue,
                    ):
                        try:
                            latency, worker_ref_latency = benchmark_target(
                                jit_kernel=_jit_kernel,
                                benchmark_state=_worker_state,
                            )
                            _call_result_queue.put(("ok", latency, worker_ref_latency, ""))
                        except TimeoutException:
                            _call_result_queue.put(("timeout", None, None, ""))
                        except Exception:
                            _call_result_queue.put(("error", None, None, traceback.format_exc()))

                    benchmark_call_thread = threading.Thread(target=_run_benchmark_target, daemon=True)
                    benchmark_call_thread.start()
                    benchmark_call_thread.join(timeout=timeout)
                    if benchmark_call_thread.is_alive():
                        result_queue.put((idx, config, jit_kernel, None, None, "timeout", ""))
                        continue

                    try:
                        status, latency, worker_ref_latency, error_text = call_result_queue.get_nowait()
                    except queue.Empty:
                        result_queue.put(
                            (
                                idx,
                                config,
                                jit_kernel,
                                None,
                                None,
                                "error",
                                "Benchmark call thread exited without returning a result.",
                            )
                        )
                        continue

                    if status == "ok":
                        worker_state.jit_input_tensors = call_state.jit_input_tensors
                        worker_state.ref_input_tensors = call_state.ref_input_tensors
                        worker_state.ref_latency_cache = call_state.ref_latency_cache
                        result_queue.put((idx, config, jit_kernel, latency, worker_ref_latency, None, ""))
                    elif status == "timeout":
                        result_queue.put((idx, config, jit_kernel, None, None, "timeout", ""))
                    else:
                        result_queue.put((idx, config, jit_kernel, None, None, "error", error_text))
                else:
                    latency, worker_ref_latency = benchmark_target(
                        jit_kernel=jit_kernel,
                        benchmark_state=worker_state,
                    )
                    result_queue.put((idx, config, jit_kernel, latency, worker_ref_latency, None, ""))
            except TimeoutException:
                result_queue.put((idx, config, jit_kernel, None, None, "timeout", ""))
            except Exception:
                result_queue.put((idx, config, jit_kernel, None, None, "error", traceback.format_exc()))

    def _benchmark_target(
        self,
        jit_kernel: tilelang.JITKernel,
        warmup: int,
        rep: int,
        benchmark_state: _BenchmarkWorkerState,
    ) -> tuple[float, float | None]:
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
        backend = profile_args.backend

        profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)

        def get_input_tensors_supply(with_output: bool):
            def func():
                if supply_prog is not None:
                    return supply_prog(profiler._get_params(with_output=with_output))
                else:
                    return profiler._get_inputs(with_output=with_output)

            return func

        jit_input_tensors_supply = get_input_tensors_supply(with_output=False)
        ref_input_tensors_supply = get_input_tensors_supply(with_output=False)

        jit_input_tensors_cache = benchmark_state.jit_input_tensors
        ref_input_tensors_cache = benchmark_state.ref_input_tensors
        ref_latency_cache = benchmark_state.ref_latency_cache

        if cache_input_tensors:
            params = profiler._get_params(with_output=False)
            if jit_input_tensors_cache is None:
                jit_input_tensors_cache = jit_input_tensors_supply()
            else:
                assert len(params) == len(jit_input_tensors_cache), "len(params) != len(jit_input_tensors_cache)"
                for p, c in zip(params, jit_input_tensors_cache):
                    if not isinstance(c, torch.Tensor):
                        continue

                    def shape_equal(a, b):
                        return all(
                            a_dim == b_dim or isinstance(a_dim, Var) or isinstance(b_dim, Var) for a_dim, b_dim in zip(a.shape, b.shape)
                        )

                    if p.dtype != c.dtype or not shape_equal(p, c):
                        logger.warning(
                            "\nIncompatible input tensor properties detected between cached tensors and "
                            "tensors regenerated for the current configuration trial. "
                            "This can happen if different tuning configurations require different input shapes/dtypes "
                            "and input tensor caching is enabled.\n"
                            "To ensure fresh, compatible inputs are generated for every trial "
                            "you can disable caching by setting:\n"
                            "  `cache_input_tensors=False`\n"
                            "within your `.set_compile_args(...)` call.\n"
                        )
                        jit_input_tensors_cache = jit_input_tensors_supply()
                        break
        else:
            jit_input_tensors_cache = jit_input_tensors_supply()

        if (not skip_check) and (ref_prog is not None):
            if manual_check_prog is not None:
                profiler.manual_assert_close(ref_prog, input_tensors=jit_input_tensors_cache, manual_check_prog=manual_check_prog)
            else:
                profiler.assert_allclose(
                    ref_prog, input_tensors=jit_input_tensors_cache, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio
                )
        latency = profiler.do_bench(n_warmup=warmup, n_repeat=rep, input_tensors=jit_input_tensors_cache, backend=backend)

        if ref_latency_cache is None and ref_prog is not None:
            ref_input_tensors_cache = ref_input_tensors_supply()
            ref_latency_cache = profiler.do_bench(
                ref_prog,
                n_warmup=warmup,
                n_repeat=rep,
                input_tensors=ref_input_tensors_cache,
                backend=backend,
            )

        benchmark_state.jit_input_tensors = jit_input_tensors_cache
        benchmark_state.ref_input_tensors = ref_input_tensors_cache
        benchmark_state.ref_latency_cache = ref_latency_cache

        return latency, ref_latency_cache

    def _resolve_benchmark_devices(
        self,
        benchmark_multi_gpu: bool,
        benchmark_devices: list[int] | None,
        target_kind: str,
    ) -> tuple[bool, list[int]]:
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
        single_device = [current_device]

        if not benchmark_multi_gpu:
            return False, single_device

        if target_kind != "cuda":
            logger.warning(
                "Multi-GPU benchmark requested but target is '%s'. Falling back to single-device benchmark on cuda:%s.",
                target_kind,
                current_device,
            )
            return False, single_device

        if not torch.cuda.is_available():
            logger.warning("Multi-GPU benchmark requested but CUDA is unavailable. Falling back to single-device benchmark.")
            return False, single_device

        visible_device_count = torch.cuda.device_count()
        if visible_device_count <= 0:
            logger.warning("Multi-GPU benchmark requested but no visible CUDA devices found. Falling back to single-device benchmark.")
            return False, single_device

        requested_devices: list[int] = []
        if benchmark_devices:
            requested_devices = list(dict.fromkeys(int(device) for device in benchmark_devices))
        else:
            raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            parsed_visible_devices = [token.strip() for token in raw_visible_devices.split(",") if token.strip()]
            if parsed_visible_devices:
                requested_devices = list(range(len(parsed_visible_devices)))
            else:
                requested_devices = list(range(visible_device_count)) if visible_device_count > 0 else single_device

        valid_devices: list[int] = []
        invalid_devices: list[int] = []
        for device in requested_devices:
            if 0 <= device < visible_device_count:
                valid_devices.append(device)
            else:
                invalid_devices.append(device)

        valid_devices = list(dict.fromkeys(valid_devices))
        if invalid_devices:
            logger.warning(
                "Ignoring invalid benchmark device ids %s. Visible CUDA device ordinals are [0, %d].",
                invalid_devices,
                max(0, visible_device_count - 1),
            )

        if not valid_devices:
            logger.warning(
                "No valid benchmark devices resolved for multi-GPU benchmark. Falling back to single-device benchmark on cuda:%s.",
                current_device,
            )
            return False, single_device

        return len(valid_devices) > 1, valid_devices

    def run(
        self,
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 180,
        use_pipeline: bool = False,
        enable_grouped_compile: bool = False,
        group_compile_size: int = 2,
        benchmark_devices: list[int] | None = None,
        benchmark_multi_gpu: bool = False,
    ):
        """Run the auto-tuning process.

        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.
            use_pipeline: Whether to pipeline benchmarking with compilation.
            enable_grouped_compile: Whether to enable grouped compilation.
            group_compile_size: Number of configurations in one compile unit.
            benchmark_devices: CUDA device ordinals used for benchmark workers when benchmark_multi_gpu=True.
            benchmark_multi_gpu: Whether to benchmark configurations across multiple CUDA GPUs.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()

        sig = inspect.signature(self.fn)
        parameters = sig.parameters

        # NOTE(chaofan):  We need to extract some parameters from the closure.
        # Consider the case:
        #   def gemm(M, N, K):
        #       def kernel(...)
        # If we only extract source, M/N/K will be symbolic and there will be cache problem.
        extra_parameters: dict[str, Any] = {}
        cells = self.fn.__closure__
        var_names = self.fn.__code__.co_freevars
        if cells is not None:
            assert len(var_names) == len(cells), "Number of free variables does not match"
            for var_name, cell in zip(var_names, cells):
                if var_name in parameters:
                    continue
                # Cell content must be serializable
                assert isinstance(cell.cell_contents, (int, float, str, bool, type(None))), (
                    f"Cell contents {cell.cell_contents} is not serializable: {type(cell.cell_contents)}"
                )
                extra_parameters[var_name] = cell.cell_contents

        if isinstance(self.configs, Callable):
            kernel_args, kernel_kwargs = self._kernel_parameters
            kernel_kwargs = dict(kernel_kwargs)
            self.configs = self.configs(*kernel_args, **kernel_kwargs)

        key = self.generate_cache_key(parameters, extra_parameters)

        with self._lock:
            if env.is_cache_enabled() and not env.is_autotune_cache_disabled():
                # First check in-memory cache
                if key in self._memory_cache:
                    # Include PrimFunc name when hitting autotuner memory cache
                    cached_result = self._memory_cache[key]
                    prim = getattr(cached_result, "func", None)
                    kernel_name = get_prim_func_name(prim, "<unknown>")
                    logger.warning(
                        "Found kernel '%s' in memory cache. For better performance, consider using `@tilelang.autotune` instead of direct AutoTuner.from_kernel.",
                        kernel_name,
                    )
                    return cached_result

                # Then check disk cache
                result = self._load_result_from_disk(key)
                if result is not None:
                    # Populate memory cache with disk result
                    self._memory_cache[key] = result
                    return result

        best_latency: float = 1e8
        best_config: dict[str, Any] | None = None
        best_kernel: tilelang.JITKernel | None = None

        compile_func, elaborate_func = self._ensure_jit_functions()
        self.jit_compile = compile_func
        self.jit_elaborate = elaborate_func

        config_args = []
        for config in self.configs:
            new_kwargs = {}
            keys = config.keys()
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            unused_keys = set(keys) - set(new_kwargs.keys())
            if len(unused_keys) > 0:
                raise ValueError(f"Unused keys in config: {unused_keys}")
            config_args.append(new_kwargs)

        if len(config_args) == 0:
            raise ValueError("No configurations to tune, please check your `@autotune` decorator")

        target_kind, _, grouped_compile_active, _ = self._resolve_grouped_compile_mode(
            enable_grouped_compile=enable_grouped_compile,
            group_compile_size=group_compile_size,
        )

        benchmark_multi_gpu_active, benchmark_device_list = self._resolve_benchmark_devices(
            benchmark_multi_gpu=benchmark_multi_gpu,
            benchmark_devices=benchmark_devices,
            target_kind=target_kind,
        )

        # check if the tunable arguments has been set.
        # get the back config argument
        top_config, *rest = config_args

        if self._kernel_parameters is not None:
            key_args_tuple, key_kwargs_tuple = self._kernel_parameters
            tunable_arguments = [key for key, _ in top_config.items()]

            def check_tunable_argument_value(key, parameters, key_args_tuple) -> bool:
                params_list = list(parameters.keys())
                assert key in params_list, f"Tunable argument {key} not found in function parameters"
                return params_list.index(key) < len(key_args_tuple)

            # Check if all tunable arguments have been tuned by comparing config keys with key_kwargs_tuple
            if any(key in top_config for key, _ in key_kwargs_tuple) or any(
                check_tunable_argument_value(key, self._function_parameters, key_args_tuple) for key in tunable_arguments
            ):
                logger.warning(
                    f"Tunable parameters {tunable_arguments} already provided during auto-tuning. Skipping compilation and using direct JIT"
                )
                # compile the kernel with the provided parameters
                jit_kernel = self.jit_compile()
                autotuner_result = AutotuneResult(libcode=jit_kernel.get_kernel_source(), func=jit_kernel.prim_func, kernel=jit_kernel)
                self._memory_cache[key] = autotuner_result
                return autotuner_result

        # Launch compile tasks
        pool, futures, future_to_unit, compile_desc = self._prepare_compile_execution(
            config_args=config_args,
            grouped_compile_active=grouped_compile_active,
            group_compile_size=group_compile_size,
            compile_func=compile_func,
            elaborate_func=elaborate_func,
        )

        ref_latency = None
        main_thread_benchmark_state = _BenchmarkWorkerState(
            jit_input_tensors=self.jit_input_tensors,
            ref_input_tensors=self.ref_input_tensors,
            ref_latency_cache=self.ref_latency_cache,
        )

        def _record_benchmark_result(latency: float, config: dict[str, Any], jit_kernel: tilelang.JITKernel, idx: int, progress_bar):
            nonlocal best_latency, best_config, best_kernel
            if latency < best_latency:
                best_latency = latency
                best_config = config
                best_kernel = jit_kernel

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {idx}")

        benchmark_worker_devices = benchmark_device_list if benchmark_multi_gpu_active else [benchmark_device_list[0]]
        benchmark_task_queues = [queue.Queue() for _ in benchmark_worker_devices]
        benchmark_result_queue: queue.Queue = queue.Queue()
        benchmark_start_event = threading.Event()
        benchmark_threads: list[threading.Thread] = []
        benchmark_expected_results = 0
        benchmark_processed_results = 0

        if use_pipeline:
            benchmark_start_event.set()

        if timeout > 0:
            logger.warning(
                "Benchmark timeout is enforced in benchmark workers by running each benchmark call "
                "in a daemon sub-thread and waiting up to the configured timeout."
            )
        benchmark_target = partial(
            self._benchmark_target,
            warmup=warmup,
            rep=rep,
        )

        def _enqueue_benchmark_task(jit_kernel: tilelang.JITKernel, config: dict[str, Any], idx: int):
            nonlocal benchmark_expected_results
            queue_idx = idx % len(benchmark_task_queues)
            benchmark_task_queues[queue_idx].put((jit_kernel, config, idx))
            benchmark_expected_results += 1

        def _process_benchmark_result(result_item, progress_bar):
            nonlocal benchmark_processed_results, ref_latency
            idx, config, jit_kernel, latency, worker_ref_latency, status, error_text = result_item
            benchmark_processed_results += 1
            progress_bar.update(1)

            if status == "timeout":
                logger.warning(f"A timeout occurred while testing config {config}, checkout autotuner.log for more details")
                return
            if status == "error":
                logger.warning(f"An error occurred while testing config {config}, checkout autotuner.log for more details")
                if error_text:
                    logger.debug(f"Error: {error_text}")
                return

            if worker_ref_latency is not None:
                ref_latency = worker_ref_latency
            assert latency is not None
            _record_benchmark_result(latency=latency, config=config, jit_kernel=jit_kernel, idx=idx, progress_bar=progress_bar)

        def _drain_benchmark_results(progress_bar, block: bool):
            while benchmark_processed_results < benchmark_expected_results:
                try:
                    if block:
                        result_item = benchmark_result_queue.get(timeout=0.1)
                    else:
                        result_item = benchmark_result_queue.get_nowait()
                except queue.Empty:
                    break
                _process_benchmark_result(result_item, progress_bar)

        # Start benchmark worker threads
        for worker_idx, worker_device in enumerate(benchmark_worker_devices):
            worker_state = _BenchmarkWorkerState() if benchmark_multi_gpu_active else main_thread_benchmark_state
            worker_thread = threading.Thread(
                target=self._benchmark_worker_loop,
                args=(
                    worker_device,
                    benchmark_task_queues[worker_idx],
                    benchmark_result_queue,
                    benchmark_start_event,
                    target_kind,
                    benchmark_target,
                    timeout,
                    worker_state,
                ),
                daemon=True,
            )
            worker_thread.start()
            benchmark_threads.append(worker_thread)

        compile_progress = tqdm(total=len(config_args), desc=compile_desc)
        progress_bar = tqdm(total=len(config_args), desc="Bench configurations")
        pending_futures = set(futures)

        # Main thread loop to process compile results and feed benchmark tasks, end when all compile tasks are done.
        try:
            while pending_futures:
                done, pending_futures = concurrent.futures.wait(
                    pending_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    unit_items = future_to_unit[future]
                    try:
                        unit_results = future.result()
                    except Exception as e:
                        compile_progress.update(len(unit_items))
                        unit_indexes = [idx for idx, _ in unit_items]
                        logger.debug("Compilation unit failed for indexes %s with error: %s", unit_indexes, e)
                        continue

                    compile_progress.update(len(unit_results))
                    for idx, config, jit_kernel, error in unit_results:
                        if error is not None:
                            logger.debug(f"Compilation failed for config {config} at index {idx} with error: {error}")
                            continue
                        assert jit_kernel is not None
                        _enqueue_benchmark_task(jit_kernel=jit_kernel, config=config, idx=idx)

                _drain_benchmark_results(progress_bar=progress_bar, block=False)

            benchmark_start_event.set()
            for worker_queue in benchmark_task_queues:
                worker_queue.put(None)

            while benchmark_processed_results < benchmark_expected_results:
                _drain_benchmark_results(progress_bar=progress_bar, block=True)

            # Avoid misleading unfinished progress bars when compile failures happen.
            progress_bar.total = max(progress_bar.n, benchmark_processed_results)
            progress_bar.refresh()
        finally:
            benchmark_start_event.set()
            for worker_queue in benchmark_task_queues:
                worker_queue.put(None)
            for worker_thread in benchmark_threads:
                worker_thread.join(timeout=1.0)
                if worker_thread.is_alive():
                    logger.warning("Benchmark worker thread did not exit cleanly before shutdown.")
            compile_progress.close()
            progress_bar.close()
            pool.shutdown()

        self.jit_input_tensors = main_thread_benchmark_state.jit_input_tensors
        self.ref_input_tensors = main_thread_benchmark_state.ref_input_tensors
        self.ref_latency_cache = main_thread_benchmark_state.ref_latency_cache

        if best_kernel is None:
            error_msg = "Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation."
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
            kernel=best_kernel,
        )

        if self.compile_args.execution_backend in ("torch"):
            logger.warning("DLPack backend does not support cache saving to disk.")
        else:
            with self._lock:
                if env.is_cache_enabled() and not env.is_autotune_cache_disabled():
                    self._save_result_to_disk(key, autotuner_result)

        self._memory_cache[key] = autotuner_result

        return autotuner_result

    def __call__(self) -> Any:
        """Make the AutoTuner callable, running the auto-tuning process.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        return self.run()


_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass
class AutoTuneImpl(Generic[_P, _T]):
    jit_impl: JITImpl

    warmup: int = 25
    rep: int = 100
    timeout: int = 100
    configs: dict | Callable = None
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
    ref_prog: Callable = None
    supply_prog: Callable = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
    manual_check_prog: Callable = None
    cache_input_tensors: bool = False

    def __post_init__(self):
        self._tuner_cache = {}

    def get_tunner(self):
        autotuner = (
            AutoTuner(self.jit_impl.func, configs=self.configs)
            .set_profile_args(
                supply_type=self.supply_type,
                ref_prog=self.ref_prog,
                supply_prog=self.supply_prog,
                rtol=self.rtol,
                atol=self.atol,
                max_mismatched_ratio=self.max_mismatched_ratio,
                skip_check=self.skip_check,
                manual_check_prog=self.manual_check_prog,
                cache_input_tensors=self.cache_input_tensors,
            )
            .set_compile_args(
                out_idx=self.jit_impl.out_idx,
                execution_backend=self.jit_impl.execution_backend,
                target=self.jit_impl.target,
                target_host=self.jit_impl.target_host,
                verbose=self.jit_impl.verbose,
                pass_configs=self.jit_impl.pass_configs,
            )
        )
        autotuner.run = partial(autotuner.run, self.warmup, self.rep, self.timeout)
        return autotuner

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> JITKernel | _T:
        return_kernel = kwargs.pop("__return_kernel", False)

        mode = self.jit_impl.initialize_jit_mode(*args, **kwargs)

        norm_args = _normalize_value(args, sort_dict_items=True)
        norm_kwargs = _normalize_value(kwargs, sort_dict_items=True)
        key = (norm_args, norm_kwargs)
        if key not in self._tuner_cache:

            def jit_elaborate(**config_arg):
                merged = dict(kwargs)
                merged.update(config_arg)
                return self.jit_impl.get_tir(*args, **merged)

            if mode == "lazy":

                def jit_compile(**config_arg):
                    return self.jit_impl(*args, **kwargs, __tune_params=config_arg)

                autotuner = self.get_tunner()
                autotuner.jit_compile = jit_compile
                autotuner.jit_elaborate = jit_elaborate
                autotuner.set_kernel_parameters(key, self.jit_impl.signature.parameters)
            else:

                def jit_compile(**config_arg):
                    merged = dict(kwargs)
                    merged.update(config_arg)
                    return self.jit_impl.compile(*args, **merged)

                autotuner = self.get_tunner()
                autotuner.jit_compile = jit_compile
                autotuner.jit_elaborate = jit_elaborate
                autotuner.set_kernel_parameters(key, self.jit_impl.signature.parameters)

            artifact = autotuner.run()
            self._tuner_cache[key] = artifact.kernel, artifact.config

        best_kernel, best_config = self._tuner_cache[key]

        if mode == "lazy":
            return best_kernel
        else:
            if return_kernel:
                return best_kernel
            exec_kwargs = dict(kwargs)
            if best_config is not None:
                exec_kwargs.update(best_config)
            _, kernel_args = self.jit_impl.func.parse_args(*args, **exec_kwargs)
            return best_kernel(*kernel_args.values())

    def compile(self, *args: _P.args, **kwargs: _P.kwargs) -> JITKernel:
        return self(*args, **kwargs, __return_kernel=True)


def autotune(  # This is the new public interface
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    configs: dict | Callable,
    # profile arguments
    warmup: int = 25,
    rep: int = 100,
    timeout: int = 100,
    # compile arguments
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
    ref_prog: Callable = None,
    supply_prog: Callable = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    max_mismatched_ratio: float = 0.01,
    skip_check: bool = False,
    manual_check_prog: Callable = None,
    cache_input_tensors: bool = False,
):
    """
    Just-In-Time (JIT) compiler decorator for TileLang functions.

    This decorator can be used without arguments (e.g., `@tilelang.jit`):
       Applies JIT compilation with default settings.

    Tips:
        - If you want to skip the auto-tuning process, you can set override the tunable parameters in the function signature.
            ```python
                if enable_autotune:
                    kernel = flashattn(batch, heads, seq_len, dim, is_causal)
                else:
                    kernel = flashattn(
                        batch, heads, seq_len, dim, is_causal, groups=groups, block_M=128, block_N=128, num_stages=2, threads=256)
            ```

    Parameters
    ----------
    func_or_out_idx : Any, optional
        If using `@tilelang.jit(...)` to configure, this is the `out_idx` parameter.
        If using `@tilelang.jit` directly on a function, this argument is implicitly
        the function to be decorated (and `out_idx` will be `None`).
    configs : Dict or Callable
        Configuration space to explore during auto-tuning.
    warmup : int, optional
        Number of warmup iterations before timing.
    rep : int, optional
        Number of repetitions for timing measurements.
    timeout : int, optional
    target : Union[str, Target], optional
        Compilation target for TVM (e.g., "cuda", "llvm"). Defaults to "auto".
    target_host : Union[str, Target], optional
        Target host for cross-compilation. Defaults to None.
    execution_backend : Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch"], optional
        Backend for kernel execution and argument passing. Use "auto" to pick a sensible
        default per target (cuda->tvm_ffi, metal->torch, others->cython).
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
        raise ValueError("Use tilelang.autotune to decorate func without arguments is not supported yet.")
    elif isinstance(func, PrimFunc):
        raise ValueError("Use tilelang.jit to decorate prim_func is not supported yet.")
    else:

        def decorator(impl):
            assert isinstance(impl, JITImpl), "The @autotune decorator can only be applied to @tilelang.jit decorated instances."
            return AutoTuneImpl(
                jit_impl=impl,
                configs=configs,
                warmup=warmup,
                rep=rep,
                timeout=timeout,
                supply_type=supply_type,
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                skip_check=skip_check,
                manual_check_prog=manual_check_prog,
                cache_input_tensors=cache_input_tensors,
            )

        return decorator
