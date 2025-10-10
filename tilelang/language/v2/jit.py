from .compile import (
    make_prim_func_generator,
    generate_arg_parser,
    DSLBuilder,
    JITFunc,
    JITPyFunc,
    set_current_builder,
)
from typing import (
    Callable,
    Protocol,
    Union,
    List,
    Dict,
    Any,
    overload,
    ParamSpec,
    TypeVar,
    Generic,
)
import cffi
from tilelang.utils.target import AVALIABLE_TARGETS, determine_target
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLWrapper
from dataclasses import dataclass
import tilelang
from tilelang import tvm
from tvm.target import Target
import logging
import ctypes
import inspect

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
    lib: JITLib | ctypes.CDLL
    lib_call: Callable
    source: str
    func: JITFunc[_P, _T]

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        const_args, dyn_args = self.func.parse_args(*args, **kws)
        assert const_args == self.func.const_args, "Const args do not match"
        return self.lib_call(*dyn_args)

    def get_source(self) -> str:
        return self.source


def compile(func: JITFunc[_P, _T], verbose=False) -> JITKernel[_P, _T]:
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
        func=func,
        source=wrapped_source,
        lib_call=lib_call,
        lib_path=lib_path,
        lib=lib,
    )


class JITDispatcher(Generic[_P, _T]):

    def __init__(
        self,
        func: Callable[_P, _T],
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        verbose: bool = False,
        pass_configs: Dict[str, Any] = None,
        compile_flags: List[str] = None,
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
        self.parse_args = generate_arg_parser(func.__name__, func)
        self._jit_func_gen_lazy = None

    @property
    def jit_func_gen(self) -> JITPyFunc[_P, None]:
        if self._jit_func_gen_lazy is None:
            self._jit_func_gen_lazy = make_prim_func_generator(self.func)
        return self._jit_func_gen_lazy

    def partial(self, *args: _P.args, **kws: _P.kwargs) -> JITFunc[_P, _T]:
        const_args, _ = self.parse_args(*args, **kws)
        if const_args in self.jit_funcs:
            return self.jit_funcs[const_args]
        builder = DSLBuilder(
            target=self.target,
            target_host=self.target_host,
            arg_parser=self.parse_args,
            const_args=const_args,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )
        set_current_builder(builder)
        self.jit_func_gen(builder)(*args, **kws)
        set_current_builder()
        self.jit_funcs[const_args] = builder.get()
        return builder.get()

    def compiled(self, *args: _P.args, **kws: _P.kwargs) -> JITKernel[_P, _T]:
        const_args, _ = self.parse_args(*args, **kws)
        kernel = self.kernels.get(const_args, None)
        if kernel is not None:
            return kernel
        func = self.partial(*args, **kws)
        kernel = compile(func)
        self.kernels[const_args] = kernel
        self.kernel_calls[const_args] = kernel.lib_call
        return kernel

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        const_args, dyn_args = self.parse_args(*args, **kws)
        kernel = self.kernel_calls.get(const_args, None)
        if kernel is not None:
            return kernel(*dyn_args)
        kernel = self.compiled(*args, **kws)
        return kernel.lib_call(*dyn_args)

    def __repr__(self):
        return f"JITGen(func={self.func}, target={self.target}, target_host={self.target_host}, verbose={self.verbose}, pass_configs={self.pass_configs}, compile_flags={self.compile_flags})"


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
) -> Callable[[Callable[_P, _T]], JITDispatcher[_P, _T]]:
    if inspect.isfunction(target):
        return JITDispatcher(
            target,
            target="auto",
            target_host=None,
            verbose=False,
            pass_configs=None,
            compile_flags=None,
        )

    def wrapper(func):
        return JITDispatcher(
            func,
            target=target,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    return wrapper
