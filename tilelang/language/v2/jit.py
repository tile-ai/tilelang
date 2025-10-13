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
                    ParamSpec, TypeVar, Generic, TypedDict, Optional, Literal)
import cffi
from concurrent.futures import ThreadPoolExecutor
from tilelang.utils.target import AVALIABLE_TARGETS, determine_target
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.cache import _kernel_cache_instance as kernel_cache
from tilelang.transform.pass_config import PassConfigKey
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
from .lang import Tune, TuneMany, Place

logger = logging.getLogger(__name__)


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
        return ("JITKernel(\n"
                f"  lib_path={repr(self.lib_path)},\n"
                f"  lib={repr(self.lib)},\n"
                f"  lib_call={self.lib_call},\n"
                f"  source={repr(self.source)},\n"
                f"  wrapped_source={repr(self.wrapped_source)},\n"
                f"  jitfunc={self.jitfunc.repr_indent(2)},\n"
                ")")

    def get_host_source(self) -> str:
        return self.wrapped_source

    def get_kernel_source(self) -> str:
        return self.source

    def bench(self, *args: _P.args, _config: Optional[Dict[str, Any]] = None, **kws: _P.kwargs):
        _config = _config or {}
        return do_bench(lambda: self(*args, **kws), **_config)


def _compile_compat(func: JITFunc[_P, _T], verbose=False) -> JITKernel[_P, _T]:
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


# more simpler version of compile, not used due to compatibility
def _compile_ng(func: JITFunc[_P, _T], verbose=False) -> JITKernel[_P, _T]:
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
    return _compile_compat(func, verbose)


def _par_compile_in_pool(pool: ThreadPoolExecutor,
                         func: Iterable[JITFunc[_P, _T]],
                         verbose=False,
                         raise_error=True) -> List[JITKernel[_P, _T] | Exception]:

    def do_compile(func, verbose):
        try:
            if isinstance(func, Exception):
                return func
            return _compile_compat(func, verbose)
        except Exception as e:
            logger.error(f'Compilation Error: {repr(e)}', exc_info=True)
            if raise_error:
                raise e
            return e

    futures = [pool.submit(do_compile, func, verbose) for func in func]
    return [future.result() for future in get_tqdm(futures, desc='Compilation')]


def par_compile(func: Iterable[JITFunc[_P, _T]],
                verbose=False,
                max_workers=None,
                raise_error=True,
                pool: ThreadPoolExecutor = None) -> List[JITKernel[_P, _T] | Exception]:
    if pool is None:
        with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix='tilelang-par-compile') as pool:
            return _par_compile_in_pool(pool, func, verbose, raise_error)
    else:
        return _par_compile_in_pool(pool, func, verbose, raise_error)


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

    def _to_record(self):
        res = {}
        for i, arg in enumerate(self.args):
            res[f'arg_{i}'] = arg
        for k, v in self.kwargs.items():
            res[k] = v
        return res


AnyCallArgs = CallArgs | Tuple[Any, ...] | Dict[str, Any]


def repr_config(call_args: CallArgs,
                fn_name: Optional[str] = "CallArgs",
                replace_with_place: bool = True,
                indent: int = 0) -> str:
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
                args.append(
                    f"{name}=tl.place({shape}, dtype={repr(arg.dtype)}, strides={repr(strides)})")
        else:
            args.append(repr(arg))
    if len(fn_name) + sum(map(len, args)) > 60:
        indent = ' ' * indent
        return fn_name + f"(\n{indent}  " + f",\n{indent}  ".join(args) + f"\n{indent})"
    else:
        return fn_name + "(" + ", ".join(args) + ")"


class Record(TypedDict):
    latency: float
    _status: Literal['Success', 'Error']
    _error: str


@dataclass(frozen=True, slots=True)
class AutoTuneResult:
    name: str
    num_errors: int
    best_latency: float
    best: Record
    best_args: CallArgs
    records: List[Record]

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def __repr__(self):
        return ('AutoTuneResult(\n'
                f'  name={self.name},\n'
                f'  num_errors={self.num_errors},\n'
                f'  best_latency={self.best_latency},\n'
                f'  best={repr(self.best)},\n'
                f'  best_args={repr_config(self.best_args, fn_name=self.name, indent=2)},\n'
                f'  records=<{len(self.records)} records>,\n'
                ')')


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
    warmup: float  # target warmup time in milliseconds
    rep: float  # target benchmark time in milliseconds
    fast_flush: bool  # Use faster L2 cache flush with int32 vs int8 (default: True)
    backend: Literal[
        "event", "cupti"]  # Profiler backend - "event" (CUDA events) or "cupti" (default: "event")
    return_mode: Literal["min", "max", "mean", "median"]
    timeout: Optional[float]


def collect_run_record(func: Callable, arg: AnyCallArgs, raise_error: bool = True) -> Record:
    call_args = CallArgs.from_anycallarg(arg)
    record = call_args._to_record()
    try:
        result = func(*call_args.args, **call_args.kwargs)
        record['latency'] = float(result)
        record['_status'] = 'Success'
        record['_result'] = result
    except Exception as e:
        record['_status'] = 'Error'
        record['_error'] = repr(e)
        if raise_error:
            raise e
        else:
            logger.warning(f'Got error when collecting result: {repr(e)}', exc_info=True)
    return record


def run_with_args(func: Callable, args: Iterable[AnyCallArgs], raise_error: bool = True):
    num_errors = 0
    records = []
    for arg in args:
        res = collect_run_record(func, arg, raise_error)
        records.append(res)
        if res['_status'] == 'Error':
            num_errors += 1
    if not raise_error:
        logger.warning(f'run_with_args: {num_errors} errors occurred')
    return records


@dataclass
class AutoTuner:
    name: str
    arg_parser: JITArgParser
    configs: List[CallArgs]
    kernels: List[JITKernel]
    bench_cfg: Dict[str, Any]

    def run(self):
        records = []
        best_latency, best, best_args = None, None, None
        num_errors = 0
        for cfg, ker in zip(self.configs, self.kernels):
            const_args, dyn_args = self.arg_parser(*cfg.args, **cfg.kwargs)  # type: ignore
            record = {k: v for k, v in zip(self.arg_parser.const_arg_names, const_args)}

            def add_record(record, records, status, latency=float('inf'), error=''):
                record.update(
                    {  # ignore: B023
                        'latency': latency,
                        '_status': status,
                        '_error': error,
                    })
                records.append(record)  # ignore: B023

            if isinstance(ker, Exception):
                add_record(record, records, 'Error', error=repr(ker))
                num_errors += 1
                continue
            try:
                latency = ker.bench(*cfg.args, **cfg.kwargs, _config=self.bench_cfg)  # type: ignore
                add_record(record, records, 'Success', latency=float(latency))
                if best_latency is None or latency < best_latency:
                    best_latency = latency
                    best = record
                    best_args = cfg
            except Exception as e:
                add_record(record, records, 'Error', error=repr(e))
                num_errors += 1
                logger.warning(f'Got error when benchmarking: {repr(e)}', exc_info=True)
        return AutoTuneResult(
            name=self.name,
            num_errors=num_errors,
            best_latency=best_latency if best_latency is not None else float('-inf'),
            best_args=best_args if best_args is not None else CallArgs(),
            best=best if best is not None else {},
            records=records,
        )


class JITDispatcher(Generic[_P, _T]):

    def __init__(self,
                 func: Callable[_P, _T],
                 target: Union[str, Target] = "auto",
                 target_host: Union[str, Target] = None,
                 verbose: bool = False,
                 pass_configs: Dict[str, Any] = None,
                 compile_flags: List[str] = None,
                 tune_cfg: Optional[Dict[str, Any] | BenchConfig] = None):
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
        self.tune_cfg = tune_cfg or {}
        self.arg_parser = generate_arg_parser(func.__name__, func)
        self._jit_func_gen_lazy = None
        self._locked = False

    @property
    def jit_func_gen(self) -> JITPyFunc[_P, None]:
        if self._jit_func_gen_lazy is None:
            self._jit_func_gen_lazy = make_prim_func_generator(self.func)
        return self._jit_func_gen_lazy

    def _partial_impl(self, __const_args__, *args: _P.args, **kws: _P.kwargs):
        const_args = __const_args__
        builder = DSLBuilder(
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )
        self.jit_func_gen(builder, *args, **kws)
        jitfunc = JITFunc(
            target=self.target,
            target_host=self.target_host,
            pass_configs=builder.get_pass_configs(),
            compile_flags=builder.get_compile_flags(),
            prim_func=builder.get(),
            global_allocs=builder.get_global_allocs(),
            arg_parser=self.arg_parser,
            const_args=const_args,
            outs=builder.get_outs(),
        )
        self.jit_funcs[const_args] = jitfunc
        return jitfunc

    def tune_configs(self,
                     configs: Iterable[AnyCallArgs],
                     max_workers: Optional[int] = None,
                     _config: Optional[Dict[str, Any]] = None) -> AutoTuneResult:
        configs = [CallArgs.from_anycallarg(cfg) for cfg in configs]
        kernels = self.par_compile(configs, max_workers=max_workers)
        tune_cfg = copy.copy(self.tune_cfg)
        tune_cfg.update(_config or {})
        tuner = AutoTuner(
            name=self.func.__name__,
            arg_parser=self.arg_parser,
            configs=configs,
            kernels=kernels,
            bench_cfg=tune_cfg)
        result = tuner.run()
        return result

    def tune(self,
             *args: _P.args,
             _config: Optional[Dict[str, Any]] = None,
             **kws: _P.kwargs) -> AutoTuneResult:
        """Tune with the given args, return the tune result

        Args: the same as the decorated tilelang kernel

        Returns: a object represents the tune result
        """
        const_args, _ = self.arg_parser(*args, **kws)
        if const_args in self.tune_cache:
            return self.tune_cache[const_args]
        configs = self.get_tune_configs(*args, **kws)
        result = self.tune_configs(configs, _config=_config)
        self.tune_cache[const_args] = result
        return result

    def repr_config(self, config: AnyCallArgs, replace_with_place: bool = True) -> str:
        """Print the config in pretty format"""
        cfg = CallArgs.from_anycallarg(config)
        return repr_config(cfg, fn_name=self.func.__name__, replace_with_place=replace_with_place)

    def partial(self, *args: _P.args, **kws: _P.kwargs) -> JITFunc[_P, _T]:
        """Generate IR from the given args, do not compile it

        Args: the same as the decorated tilelang kernel

        Returns: a jitfunc containing all parameters and tir.prim_func for tilelang compilation
        """
        const_args, _ = self.arg_parser(*args, **kws)
        if const_args in self.jit_funcs:
            return self.jit_funcs[const_args]
        if has_tune(const_args):
            result = self.tune(*args, **kws)
            assert not has_tune(result.best_args.args)
            assert not has_tune(result.best_args.kwargs.values())
            return self.partial(*result.best_args.args, **result.best_args.kwargs)
        assert not self._locked, "JITDispatcher is locked, cannot create new JITFunc"
        return self._partial_impl(const_args, *args, **kws)

    def compile(self, *args: _P.args, **kws: _P.kwargs) -> JITKernel[_P, _T]:
        """Compile a kernel with given args

        Args: the same as the decorated tilelang kernel

        Returns: the compiled kernel
        """
        const_args, _ = self.arg_parser(*args, **kws)
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
        """Get all tune config in list

        Args: the same as the decorated tilelang kernel

        Returns:
            return list of (args, kws) to call kernel(*args, **kws)
        """
        binded_sigs = self.arg_parser.signature.bind(*args, **kws)
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
                    max_workers: int = None,
                    raise_error=True,
                    pool: ThreadPoolExecutor = None) -> List[JITKernel[_P, _T] | Exception]:
        """Compile multiple config in parallel

        This function compiles the config in parallel, and return the compiled kernels

        Args:
            configs: list of tuple[args, kws] for kernel(*args, **kws)

        Returns:
            A list of all JITKernel
        """
        configs = [CallArgs.from_anycallarg(cfg) for cfg in configs]
        jitfuncs = []
        logger.info(f'Elaborate {len(configs)} configs')
        for cfg in get_tqdm(configs, desc='Elaboration'):
            try:
                jitfunc = self.partial(*cfg.args, **cfg.kwargs)
            except Exception as e:
                logger.error(f'Elaboration Error: {repr(e)}', exc_info=True)
                if raise_error:
                    raise e
                jitfunc = e
            jitfuncs.append(jitfunc)
        kernels = par_compile(jitfuncs, max_workers=max_workers, raise_error=raise_error, pool=pool)
        if not raise_error:
            num_error = sum(map(lambda x: isinstance(x, Exception), kernels))
            if num_error > 0:
                logger.warning(f'{num_error} compilation errors occurred')
        return kernels

    def bench(self,
              *args: _P.args,
              _config: Optional[BenchConfig] = None,
              **kws: _P.kwargs) -> float:
        kernel = self.compile(*args, **kws)
        return kernel.bench(*args, _config=_config, **kws)

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        const_args, dyn_args = self.arg_parser(*args, **kws)
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
    pass_configs: Dict[PassConfigKey, Any] = None,
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
            tune_cfg=tune_cfg)

    def wrapper(func):
        return JITDispatcher(
            func,
            target=target,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            tune_cfg=tune_cfg)

    return wrapper
