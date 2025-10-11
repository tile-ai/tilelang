from __future__ import annotations
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from .compile import (
    make_prim_func_generator,
    generate_arg_parser,
    DSLBuilder,
    JITFunc,
    JITPyFunc,
    JITArgParser,
    make_macro_generator,
)
from typing import (Callable, Iterable, Protocol, Union, Tuple, List, Dict, Any, overload,
                    ParamSpec, TypeVar, Generic, TypedDict, Optional, Unpack, Literal)
import cffi
from concurrent.futures import ThreadPoolExecutor, as_completed
from tilelang.utils.target import AVALIABLE_TARGETS, determine_target
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.cache import _kernel_cache_instance as kernel_cache
from tilelang.profiler import do_bench
from dataclasses import dataclass, field
import tilelang
from tilelang import tvm
from tvm.target import Target
import logging
import inspect
import itertools
import torch
import copy
import signal
from .lang import Tune, TuneMany, Place

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)
    return result


class JITLib(Protocol):

    def init(self) -> int:
        ...

    def get_last_error(self) -> bytes:
        ...

    def call(self, *args):
        ...


_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass(slots=True)
class JITKernel(Generic[_P, _T]):
    lib_path: str
    lib: JITLib
    lib_call: Callable
    source: str
    wrapped_source: str
    jitfunc: JITFunc[_P, _T]

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        const_args, dyn_args = self.jitfunc.parse_args(*args, **kws)
        assert const_args == self.jitfunc.const_args, "Const args do not match"
        return self.lib_call(*dyn_args)

    def __repr__(self):
        return (
            "JITKernel(\n"
            f"  lib_path={repr(self.lib_path)},\n"
            f"  lib={repr(self.lib)},\n"
            f"  lib_call={self.lib_call},\n"
            f"  source={repr(self.source)},\n"
            f"  wrapped_source={repr(self.wrapped_source)},\n"
            f"  jitfunc={self.jitfunc.repr_indent(2)},\n"
            ")"
        )

    def get_host_source(self) -> str:
        return self.wrapped_source

    def get_kernel_source(self) -> str:
        return self.source

    def bench(self, *args: _P.args, _config: Optional[BenchConfig] = None, **kws: _P.kwargs) -> float:
        _config = _config or BenchConfig.default()
        timeout = _config.get('timeout', None)
        if 'timeout' in _config:
            del _config['timeout']
        if timeout is not None:
            latency = run_with_timeout(lambda: do_bench(lambda: self(*args, **kws), **_config), timeout)
        else:
            latency = do_bench(lambda: self(*args, **kws), **_config)
        return latency


def compile_compab(func: JITFunc[_P, _T], verbose=False) -> JITKernel[_P, _T]:
    tl_kernel = kernel_cache.cached(
        func.prim_func,
        func.out_idx,
        *func.global_allocs,
        target=func.target,
        target_host=func.target_host,
        execution_backend="cython",
        verbose=verbose,
        pass_configs=func.pass_configs,
        compile_flags=func.compile_flags,
    )
    adaptor: CythonKernelAdapter = tl_kernel.adapter
    lib_path = adaptor.lib_generator.libpath
    source = adaptor.kernel_global_source
    wrapped_source = adaptor.wrapped_source

    ffi = cffi.FFI()
    ffi.cdef(func.get_cffi_sig())
    ffi.cdef("int init();")
    ffi.cdef("const char * get_last_error();")
    lib: JITLib = ffi.dlopen(lib_path)

    result = lib.init()
    if result != 0:
        error_msg = lib.get_last_error().decode("utf-8")
        error_msg += f"\n{wrapped_source}"
        raise RuntimeError(f"Initialization failed: {error_msg}")

    lib_call = func.generate_global_alloc_wrapper(lib.call)

    return JITKernel(
        lib_path=adaptor.lib_generator.libpath,
        lib=lib,
        lib_call=lib_call,
        source=source,
        wrapped_source=wrapped_source,
        jitfunc=func,
    )


# more simpler version of compile, not used due to compability
def compile_ng(func: JITFunc[_P, _T], verbose=False) -> JITKernel[_P, _T]:
    func.generate_global_alloc_wrapper(None)  # test the wrapper generation

    mod = tvm.IRModule({func.prim_func.attrs["global_symbol"]: func.prim_func})

    with tvm.transform.PassContext(opt_level=3, config=func.pass_configs), func.target:
        artifact = tilelang.lower(mod, target=func.target, target_host=func.target_host)

    lib_generator = LibraryGenerator(func.target, verbose)
    lib_generator.assign_pass_configs(func.pass_configs)
    lib_generator.assign_compile_flags(func.compile_flags)

    wrapper = TLWrapper(func.target)
    wrapper.assign_optimized_module(scheduled_ir_module=mod)
    wrapper.assign_pass_configs(func.pass_configs)
    wrapper.assign_host_module(artifact.host_mod)
    wrapper.assign_device_module(artifact.device_mod)

    wrapped_source = wrapper.wrap(artifact.kernel_source)

    lib_generator.update_lib_code(wrapped_source)
    lib_generator.compile_lib()

    lib_path = lib_generator.get_lib_path()

    ffi = cffi.FFI()
    ffi.cdef(func.get_cffi_sig())
    ffi.cdef("int init();")
    ffi.cdef("const char * get_last_error();")
    lib: JITLib = ffi.dlopen(lib_path)

    result = lib.init()
    if result != 0:
        error_msg = lib.get_last_error().decode("utf-8")
        error_msg += f"\n{wrapped_source}"
        raise RuntimeError(f"Initialization failed: {error_msg}")

    lib_call = func.generate_global_alloc_wrapper(lib.call)

    return JITKernel(
        jitfunc=func,
        source=artifact.kernel_source,
        wrapped_source=wrapped_source,
        lib_call=lib_call,
        lib_path=lib_path,
        lib=lib,
    )


def compile(func: JITFunc[_P, _T], verbose=False) -> JITKernel[_P, _T]:
    return compile_compab(func, verbose)


def has_tune(x: Iterable[Any]):
    return any(map(lambda x: isinstance(x, (Tune, TuneMany)), x))


@dataclass(frozen=True, slots=True)
class CallArgs:
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert not has_tune(self.args), "Tune is not allowed in args"
        assert not has_tune(self.kwargs.values()), "Tune is not allowed in kwargs"

    @classmethod
    def from_anycallarg(cls, cfg: AnyCallArgs) -> CallArgs:
        if isinstance(cfg, CallArgs):
            return cfg
        elif isinstance(cfg, dict):
            return cls(kwargs=cfg)
        else:
            return cls(args=cfg)

    @classmethod
    def from_call(cls, *args, **kws) -> CallArgs:
        return cls(args, kws)

    def repr_indent(self, indent: int = 0) -> str:
        return repr_config(self, indent=indent)

    def __repr__(self) -> str:
        return self.repr_indent()


AnyCallArgs = CallArgs | Tuple[Any, ...] | Dict[str, Any]


def repr_config(call_args: CallArgs, fn_name: Optional[str] = "CallArgs", replace_with_place: bool = True, indent: int = 0) -> str:
    args = []
    for arg in call_args.args:
        if replace_with_place and isinstance(arg, (torch.Tensor, Place)):
            shape = ", ".join(map(repr, arg.shape))
            if isinstance(arg, torch.Tensor) and arg.is_contiguous():
                args.append(f"tl.place({shape}, dtype={repr(arg.dtype)})")
            else:
                strides = ", ".join(map(repr, arg.stride()))
                args.append(f"tl.place({shape}, dtype={repr(arg.dtype)}, strides={repr(strides)})")
        else:
            args.append(repr(arg))
    for name, arg in call_args.kwargs.items():
        if replace_with_place and isinstance(arg, (torch.Tensor, Place)):
            shape = ", ".join(map(repr, arg.shape))
            if isinstance(arg, torch.Tensor) and arg.is_contiguous():
                args.append(f"{name}=tl.place({shape}, dtype={repr(arg.dtype)})")
            else:
                strides = ", ".join(map(repr, arg.stride()))
                args.append(f"{name}=tl.place({shape}, dtype={repr(arg.dtype)}, strides={repr(strides)})")
        else:
            args.append(repr(arg))
    if len(fn_name) + sum(map(len, args)) > 60:
        indent = ' ' * indent
        return fn_name + f"(\n{indent}  " + f",\n{indent}  ".join(args) + f"\n{indent})"
    else:
        return fn_name + "(" + ", ".join(args) + ")"


class Record(TypedDict):
    latency: float
    _status: str
    _error: str


@dataclass(frozen=True, slots=True)
class AutoTuneResult:
    name: str
    best_latency: float
    best: Record
    best_args: CallArgs
    records: List[Record]

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def __repr__(self):
        return (
            'AutoTuneResult(\n'
            f'  name={self.name},\n'
            f'  best_latency={self.best_latency},\n'
            f'  best={repr(self.best)},\n'
            f'  best_args={repr_config(self.best_args, fn_name=self.name, indent=2)},\n'
            f'  records=<{len(self.records)} records>,\n'
            ')'
        )

def get_tqdm(*args, **kws):
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except Exception:
        from tqdm import tqdm
    return tqdm(*args, **kws)

class BenchConfig(TypedDict):
    warmup: float# target warmup time in milliseconds
    rep: float # target benchmark time in milliseconds
    fast_flush: bool # Use faster L2 cache flush with int32 vs int8 (default: True)
    backend: Literal["event", "cupti"] # Profiler backend - "event" (CUDA events) or "cupti" (default: "event")
    return_mode: Literal["min", "max", "mean", "median"]
    timeout: Optional[float]

    @classmethod
    def default(cls):
        return cls(
            warmup=25,
            rep=100,
            fast_flush=True,
            backend="cupti",
            return_mode="mean",
            timeout=None
        )


@dataclass(slots=True)
class AutoTuner:
    name: str
    arg_parser: JITArgParser
    configs: List[CallArgs]
    kernels: List[JITKernel | Exception]

    tune_cfg: BenchConfig = field(default_factory=BenchConfig.default)

    def run_one(self, ker: JITKernel, cfg: CallArgs) -> Record:

        def record(latency=float('nan'), **dic: Unpack[Record]):
            const_args, dyn_args = self.arg_parser.parser(*cfg.args, **cfg.kwargs)
            kv = {k: v for k, v in zip(self.arg_parser.const_arg_names, const_args)}
            kv.update(dic)
            kv['latency'] = latency
            return kv

        if isinstance(ker, Exception):
            return record(_status='Compile Error', _error=repr(ker))

        try:
            latency = ker.bench(*cfg.args, _config=self.tune_cfg, **cfg.kwargs)
            return record(_status='Success', latency=latency, _error='')
        except AssertionError as e:
            return record(_status='Output Error', _error=repr(e))
        except RuntimeError as e:
            return record(_status='Runtime Error', _error=repr(e))
        except Exception as e:
            return record(_status='Error', _error=repr(e))

    def run(self):
        records = []
        best_latency = float('-inf')
        best_record = None
        best_args = None
        progress_bar = get_tqdm(
            zip(self.kernels, self.configs),
            total=len(self.kernels),
            desc=f'tune: {self.name}'
        )
        for ker, cfg in progress_bar:
            record = self.run_one(ker, cfg)
            if best_record is None or record['latency'] > best_latency:
                best_latency = record['latency']
                best_record = record
                best_args = cfg
            records.append(record)
        return AutoTuneResult(name=self.name, records=records, best_latency=best_latency, best=best_record, best_args=best_args)


class JITDispatcher(Generic[_P, _T]):

    def __init__(self,
                 func: Callable[_P, _T],
                 target: Union[str, Target] = "auto",
                 target_host: Union[str, Target] = None,
                 verbose: bool = False,
                 pass_configs: Dict[str, Any] = None,
                 compile_flags: List[str] = None,
                 tune_cfg: Optional[Dict[str, Any] | BenchConfig] = None
                 ):
        self.func = func
        self.target_host = target_host
        if isinstance(target, str):
            assert target in AVALIABLE_TARGETS, f"Invalid target: {target}"
            target = determine_target(target)
        self.target = Target(target)
        self.verbose = verbose
        self.pass_configs = pass_configs or {}
        self.compile_flags = compile_flags or []
        self.jit_funcs = {}
        self.kernel_calls = {}
        self.kernels = {}
        self.tune_cache = {}
        self.tune_cfg = tune_cfg or BenchConfig.default()
        self._arg_parser = generate_arg_parser(func.__name__, func)
        self.parse_args = self._arg_parser.parser
        self._jit_func_gen_lazy = None
        self._locked = False

    @property
    def jit_func_gen(self) -> JITPyFunc[_P, None]:
        if self._jit_func_gen_lazy is None:
            self._jit_func_gen_lazy = make_prim_func_generator(self.func)
        return self._jit_func_gen_lazy

    def _partial_notune(self, __const_args__, *args: _P.args, **kws: _P.kwargs):
        const_args = __const_args__
        builder = DSLBuilder(
            target=self.target,
            target_host=self.target_host,
            arg_parser=self.parse_args,
            const_args=const_args,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )
        self.jit_func_gen(builder, *args, **kws)
        jitfunc = builder.get()
        self.jit_funcs[const_args] = jitfunc
        return jitfunc

    def _bench(self, kernel, args, kws):
        const_args = self.parse_args(*args, **kws)
        record = {k: v for k, v in zip(const_args, self._arg_parser.const_arg_names)}
        # 1. check whether compilation error
        if isinstance(kernel, Exception):
            record['_status'] = 'Compilation Error'
            record['_error'] = repr(kernel)
            return record

    def tune_configs(self, configs: Iterable[AnyCallArgs], max_workers: Optional[int]=None, **tune_cfg: Unpack[BenchConfig]) -> AutoTuneResult:
        configs = [CallArgs.from_anycallarg(cfg) for cfg in configs]
        kernels = self.par_compile(configs, max_workers=max_workers)
        tune_cfg = copy.copy(self.tune_cfg)
        tune_cfg.update(tune_cfg)
        tuner = AutoTuner(
            name=self.func.__name__,
            arg_parser=self._arg_parser,
            configs=configs,
            kernels=kernels,
            tune_cfg=tune_cfg
        )
        result = tuner.run()
        return result

    def tune(self, *args: _P.args, **kws: _P.kwargs) -> AutoTuneResult:
        """Tune with the given args, return the tune result

        Args: the same as the decorated tilelang kernel

        Returns: a object represents the tune result
        """
        const_args, _ = self.parse_args(*args, **kws)
        if const_args in self.tune_cache:
            return self.tune_cache[const_args]
        configs = self.get_tune_configs(*args, **kws)
        kernels = self.par_compile(configs)
        tuner = AutoTuner(
            name=self.func.__name__,
            arg_parser=self._arg_parser,
            configs=configs,
            kernels=kernels
        )
        result = tuner.run()
        self.tune_cache[const_args] = result
        return result

    def repr_config(self, config: AnyCallArgs, replace_with_place: bool=True) -> str:
        """Print the config in pretty format"""
        cfg = CallArgs.from_anycallarg(config)
        return repr_config(cfg, fn_name=self.func.__name__, replace_with_place=replace_with_place)

    def partial(self, *args: _P.args, **kws: _P.kwargs) -> JITFunc[_P, _T]:
        """Generate IR from the given args, do not compile it

        Args: the same as the decorated tilelang kernel

        Returns: a jitfunc containing all parameters and tir.prim_func for tilelang compilation
        """
        const_args, _ = self.parse_args(*args, **kws)
        if const_args in self.jit_funcs:
            return self.jit_funcs[const_args]
        if has_tune(const_args):
            result = self.tune(*args, **kws)
            assert not has_tune(result.best_args.args)
            assert not has_tune(result.best_args.kwargs.values())
            return self.partial(*result.best_args.args, **result.best_args.kwargs)
        assert not self._locked, "JITDispatcher is locked, cannot create new JITFunc"
        return self._partial_notune(const_args, *args, **kws)

    def compile(self, *args: _P.args, **kws: _P.kwargs) -> JITKernel[_P, _T]:
        """Compile a kernel with given args

        Args: the same as the decorated tilelang kernel

        Returns: the compiled kernel
        """
        const_args, _ = self.parse_args(*args, **kws)
        kernel = self.kernels.get(const_args, None)
        if kernel is not None:
            return kernel
        assert not self._locked, "JITDispatcher is locked, cannot create new JITKernel"
        func = self.partial(*args, **kws)
        kernel = compile(func)
        self.kernels[const_args] = kernel
        self.kernel_calls[const_args] = kernel.lib_call
        return kernel

    def get_tune_configs(self, *args: _P.args, **kws: _P.kwargs) -> List[CallArgs]:
        """Get producted tune config in list

        Args: the same as the decorated tilelang kernel

        Returns:
            return list of (args, kws) to call kernel(*args, **kws)
        """
        binded_sigs = self._arg_parser.sig.bind(*args, **kws)
        binded_sigs.apply_defaults()
        args = binded_sigs.args
        kws = binded_sigs.kwargs

        def arg_to_tup(arg):
            if isinstance(arg, (Tune, TuneMany)):
                return arg.data
            else:
                return (arg,)

        args_mapped = list(map(arg_to_tup, args))
        kw_keys = kws.keys()
        kw_vals_mapped = list(map(arg_to_tup, kws.values()))

        def count_prod(lis):
            res = 1
            for k in lis:
                res *= len(k)
            return res

        total_elements = count_prod(kw_keys) * count_prod(kw_vals_mapped)
        if total_elements > 10000:
            logger.warning(f"Attempt to generate {total_elements} configs")
        configs = []
        for args_vals in itertools.product(*args_mapped):
            for kw_vals in itertools.product(*kw_vals_mapped):
                configs.append(CallArgs(args_vals, {k: v for k, v in zip(kw_keys, kw_vals)}))
        return configs

    def par_compile(self,
                    configs: Iterable[AnyCallArgs],
                    max_workers: int = None) -> List[JITKernel[_P, _T] | Exception]:
        """Compile multiple config in parallel

        This function compiles the config in parallel, and return the compiled kernels

        Args:
            configs: list of tuple[args, kws] for kernel(*args, **kws)

        Returns:
            A list of all JITKernel
        """
        pool = ThreadPoolExecutor(max_workers, thread_name_prefix='tilelang-par-compile')
        configs = [CallArgs.from_anycallarg(cfg) for cfg in configs]

        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            def compile_with_device(args: CallArgs):
                torch.cuda.set_device(device)
                return self.compile(*args.args, **args.kwargs)
        else:

            def compile_with_device(args: CallArgs):
                return self.compile(*args.args, **args.kwargs)

        futures = []
        future_to_idx = {}
        results = [... for _ in range(len(configs))]

        logger.debug(f"Add {len(configs)} configs into thread pool")

        for i, args in enumerate(configs):
            future = pool.submit(compile_with_device, args)
            futures.append(future)
            future_to_idx[future] = i

        logger.debug(f"Add {len(configs)} configs into thread pool")

        progress_bar = get_tqdm(as_completed(futures), total=len(futures), desc=f'par_compile: {self.func.__name__}')
        for future in progress_bar:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(
                    f"Compiling failed for config {self.repr_config(configs[idx])} with error {repr(e)}", exc_info=True
                )
                results[idx] = e

        pool.shutdown()
        return results

    def bench(self, *args: _P.args, _config: Optional[BenchConfig] = None, **kws: _P.kwargs) -> float:
        kernel = self.compile(*args, **kws)
        return kernel.bench(*args, _config=_config, **kws)

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        const_args, dyn_args = self.parse_args(*args, **kws)
        kernel = self.kernel_calls.get(const_args, None)
        if kernel is not None:
            return kernel(*dyn_args)
        kernel = self.compile(*args, **kws)
        return kernel.lib_call(*dyn_args)

    def __repr__(self):
        return self.jit_func_gen.__tl_code__

    def __str__(self):
        return self.jit_func_gen.__tl_code__

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False


class JITMacro(Generic[_P, _T]):

    def __init__(self, func: Callable[_P, _T]):
        self.func = func

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        return self.func(*args, **kws)

    def __str__(self):
        return self.func.__tl_code__

    def __repr__(self):
        return self.func.__tl_code__


def macro(func: Callable[_P, _T]) -> JITMacro[_P, _T]:
    return JITMacro(make_macro_generator(func))


@overload
def jit(func: Callable[_P, _T]) -> JITDispatcher[_P, _T]:
    ...


@overload
def jit(
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    verbose: bool = False,
    pass_configs: Dict[str, Any] = None,
    compile_flags: List[str] = None,
) -> Callable[[Callable[_P, _T]], JITDispatcher[_P, _T]]:
    ...


def jit(
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    verbose: bool = False,
    pass_configs: Dict[str, Any] = None,
    compile_flags: List[str] = None,
    tune_cfg: Optional[Dict[str, Any] | BenchConfig] = None,
) -> Callable[[Callable[_P, _T]], JITDispatcher[_P, _T]]:
    if inspect.isfunction(target):
        return JITDispatcher(
            target,
            target="auto",
            target_host=None,
            verbose=False,
            pass_configs=None,
            compile_flags=None,
            tune_cfg=tune_cfg
        )

    def wrapper(func):
        return JITDispatcher(
            func,
            target=target,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            tune_cfg=tune_cfg
        )

    return wrapper
