from __future__ import annotations
import inspect
import torch
from dataclasses import dataclass, field
from tvm import tir
import linecache
import io
from .ast_rewrite import DSLMutator
from .types import (
    DynSchema,
    StridedTensorSchema,
    ConstSchema,
    MakeEmpty,
    Buffer_,
    cvt_dtype,
    cvt_tvm_dtype_to_torch,
    get_ptr_type,
    cvt_tvm_dtype_to_cffi,
    cvt_tvm_dtype_to_ctypes,
)
import threading
import ctypes
from typing import (
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Optional,
    List,
    get_type_hints,
    ContextManager,
    Any,
    Set,
    Generic,
    ParamSpec,
    Protocol,
)
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as tb
from tvm.target import Target
from contextlib import contextmanager
import ast

_P = ParamSpec('_P')
_T = TypeVar("_T")


class JITPyFunc(Protocol[_P, _T]):

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        ...

    __tl_code__: str


def generate_arg_parser(fn_name: str, func: Callable[_P, _T]) -> JITPyFunc[_P, Tuple[Tuple, Tuple]]:
    func_args = []
    code_parse_arg = []
    default_dict = {}
    tup_dyn = []
    tup_const = []
    dyn_dict = {}

    def add_dyn(var_name: str, data: str):
        nonlocal code_parse_arg, tup_dyn, dyn_dict
        if var_name in dyn_dict:
            code_parse_arg.append(
                f"assert {data} == {dyn_dict[var_name]}, 'dyn argument {var_name} mismatch'")
        else:
            dyn_dict[var_name] = data
            tup_dyn.append(data)

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    for param in sig.parameters.values():
        name = param.name
        schema = type_hints.get(name, ConstSchema())
        default = param.default
        if default is not inspect._empty:
            default_name = f"__{name}_default__"
            default_dict[default_name] = default
            func_args.append(f"{name} = {default_name}")
        else:
            func_args.append(f"{name}")
        if isinstance(schema, DynSchema):
            add_dyn(schema.name or name, name)
        elif isinstance(schema, StridedTensorSchema):
            tup_dyn.append(f"{name}.data_ptr()")
            tup_const.append(f"{name}.dtype")
            code_parse_arg.append(
                f'assert {name}.device != __device_cpu__, "Expected a non cpu tensor"')
            if schema.shape is not None:
                code_parse_arg.append(
                    ", ".join([f"{name}__shape_{i}" for i in range(len(schema.stride))]) +
                    f" = {name}.shape")
                # code_parse_arg.append(f'{name}__shape_ = {name}.shape')
                for i, dim in enumerate(schema.shape):
                    if isinstance(dim, DynSchema):
                        var_name = dim.name or f"{name}__shape_{i}"
                        add_dyn(var_name, f"{name}__shape_{i}")
                    else:
                        tup_const.append(f"{name}__shape_{i}")
            else:
                tup_const.append(f"{name}.shape")
            if schema.stride is not None:
                code_parse_arg.append(
                    ", ".join([f"{name}__stride_{i}" for i in range(len(schema.stride))]) +
                    f" = {name}.stride()")
                for i, dim in enumerate(schema.stride):
                    if isinstance(dim, DynSchema):
                        var_name = dim.name or f"{name}__stride_{i}"
                        add_dyn(var_name, f"{name}__stride_{i}")
                    else:
                        tup_const.append(f"{name}__stride_{i}")
            else:
                tup_const.append(f"{name}.stride()")
        else:
            tup_const.append(name)

    closure = {
        "__device_cpu__": torch.device("cpu"),
        **default_dict,
    }

    source = ""
    source += "def parse_args(" + ", ".join(closure.keys()) + "):\n"
    source += f"  def {fn_name}(" + ", ".join(func_args) + ", __stream__=None" + "):\n"
    source += "    " + "\n    ".join(code_parse_arg) + "\n"
    source += "    __const_args__ = (" + ", ".join(tup_const) + ")\n"
    source += "    __dyn_args__ = (" + ", ".join(tup_dyn) + ", __stream__)\n"
    source += "    return __const_args__, __dyn_args__\n"
    source += f"  return {fn_name}\n"

    locs = {}
    line_cache_name = f"{fn_name}_{id(source)}"
    linecache.updatecache(line_cache_name, io.StringIO(source))
    code = compile(source, line_cache_name, "exec")
    exec(code, {}, locs)
    fn = locs["parse_args"](**closure)
    fn.__tl_code__ = source
    return fn


def get_current_stream_functor():
    if torch.cuda.is_available():
        try:
            torch.cuda._lazy_init()
            current_device = torch._C._cuda_getDevice
            get_stream = torch._C._cuda_getCurrentRawStream
            return lambda: get_stream(current_device())
        except ImportError:
            get_stream = torch.cuda.current_stream
            return lambda: get_stream().cuda_stream
    else:
        return lambda: 0


@dataclass(slots=True)
class JITFunc(Generic[_P, _T]):
    target: Target
    target_host: Target
    global_allocs: List[Buffer_]
    outs: List[Buffer_]
    pass_configs: Dict[str, Any]
    compile_flags: List[str]
    prim_func: tir.PrimFunc
    arg_parser: Callable[_P, Tuple[Tuple, Tuple]]
    const_args: Tuple

    @property
    def out_idx(self) -> Tuple[int]:
        return tuple(i.arg_idx for i in self.outs)

    def find_arg(self, var: tir.Var) -> int:
        for i, p in enumerate(self.prim_func.params):
            if p.same_as(var):
                return i
        raise RuntimeError(f"Cannot find argument {var} in function parameters")

    def out_idx_set(self) -> Set[int]:
        out_idx_set = set()
        for out in self.outs:
            if out.arg_idx is not None:
                out_idx_set.add(out.arg_idx)
        return out_idx_set

    def get_ctypes_sig(self) -> Tuple:
        params = []
        for x in self.prim_func.params:
            if isinstance(x, tir.Var):
                params.append(cvt_tvm_dtype_to_ctypes(x.dtype))
            else:
                raise RuntimeError(f"Unsupported argument type: {type(x)}")
        params.append(ctypes.c_void_p)
        return (ctypes.c_int, params)

    def get_cffi_sig(self) -> str:
        params = []
        for x in self.prim_func.params:
            if isinstance(x, tir.Var):
                cffi_type = cvt_tvm_dtype_to_cffi(x.dtype)
                params.append(f"{cffi_type} {x.name}")
            else:
                raise RuntimeError(f"Unsupported argument type: {type(x)}")
        params.append("long __cudastream")
        return "int call(" + ", ".join(params) + ");"

    def generate_global_alloc_wrapper(self, func) -> JITPyFunc[Any, _T]:
        params: List[tir.Var] = self.prim_func.params
        call_args = []
        for p in params:
            call_args.append(p.name)
        out_idx_set = self.out_idx_set()
        func_args = [arg for i, arg in enumerate(call_args) if i not in out_idx_set]
        closure = {}
        stmts = []
        closure["__tl_empty"] = torch.empty
        closure["__tl_kernel"] = func
        returns = []
        for allocs in self.global_allocs:
            shape_args = []
            for i, expr in enumerate(allocs.shape):
                name = f"__{allocs.buffer.name}_shape_{i}"
                if isinstance(expr, tir.Var):
                    idx = self.find_arg(expr)
                    shape_args.append(call_args[idx])
                elif isinstance(expr, (tir.IntImm, tir.FloatImm)):
                    closure[name] = expr.value
                    shape_args.append(name)
                elif isinstance(expr, (int, float)):
                    closure[name] = expr
                    shape_args.append(name)
                else:
                    raise RuntimeError(f"Unsupported shape expression type: {type(expr)}")
            dtype_name = f"__{allocs.buffer.name}_dtype"
            device_name = f"__{allocs.buffer.name}_device"
            closure[dtype_name] = cvt_tvm_dtype_to_torch(allocs.dtype)
            closure[device_name] = allocs.device or torch.device("cuda")
            arg_name = call_args[allocs.arg_idx]
            tensor_name = f"{arg_name}_tensor"
            stmts.append(f"{tensor_name} = __tl_empty(" + ",".join(shape_args) +
                         f", dtype={dtype_name}, device={device_name})")
            stmts.append(f"{arg_name} = {tensor_name}.data_ptr()")
        for out in self.outs:
            arg_name = call_args[out.arg_idx]
            returns.append(f"{arg_name}_tensor")
        closure["__stream_functor__"] = get_current_stream_functor()
        stmts.append("if __stream__ is None: __stream__ = __stream_functor__()")
        stmts.append("assert __tl_kernel(" + ",".join(call_args) +
                     ", __stream__) == 0, 'Kernel call failed'")

        source = ""
        source += "def __closure(" + ", ".join(closure.keys()) + "):\n"
        source += "  def wrapper(" + ", ".join(func_args) + ", __stream__):\n"
        source += "    " + "\n    ".join(stmts) + "\n"
        source += "    return " + ",".join(returns) + "\n"
        source += "  return wrapper"

        locs = {}
        line_cache_name = f"<tl_torch_wrapper_{id(source)}>"
        linecache.updatecache(line_cache_name, io.StringIO(source))
        code = compile(source, line_cache_name, "exec")
        exec(code, {}, locs)
        fn = locs["__closure"](**closure)
        fn.__tl_code__ = source
        return fn

    def parse_args(self, *args: _P.args, **kws: _P.kwargs) -> Tuple[Tuple, Tuple]:
        return self.arg_parser(*args, **kws)


@dataclass
class DSLBuilder:
    # base pass_config or compile_flags
    target: Target
    target_host: Target
    arg_parser: Optional[Callable]
    const_args: Tuple
    pass_configs: Dict[str, Any] = field(default_factory=dict)
    compile_flags: List[str] = field(default_factory=list)
    default_device: Optional[torch.device] = None

    global_allocs: List[Buffer_] = field(default_factory=list)
    outs: List[Buffer_] = field(default_factory=list)

    # used to store pending shape args, making handle arg first
    pending_args: List[(str, tir.Var)] = field(default_factory=list)
    arg_idx: int = 0

    var_map: Dict[str, tir.Var] = field(default_factory=dict)
    frames: List[Any] = field(default_factory=list)
    builder: IRBuilder = field(default_factory=IRBuilder)
    _params: List[tir.Var] = field(default_factory=list)

    def get(self) -> JITFunc:
        return JITFunc(
            target=self.target,
            target_host=self.target_host,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
            prim_func=self.builder.get(),
            global_allocs=self.global_allocs,
            arg_parser=self.arg_parser,
            const_args=self.const_args,
            outs=self.outs,
        )

    @contextmanager
    def with_frame(self, frame: ContextManager):
        n = len(self.frames)
        self.frames.append(frame)
        yield frame.__enter__()
        while len(self.frames) > n:
            self.frames.pop().__exit__(None, None, None)

    def push_frame(self, frame: ContextManager):
        self.frames.append(frame)
        return frame.__enter__()

    def prim_func(self):
        return self.with_frame(tb.prim_func())

    def func_name(self, name: str):
        tb.func_name(name)

    def get_param(self, name: str, ty: Any, new: bool = False) -> tir.Var:
        if name not in self.var_map:
            var = tir.Var(name, ty)
            self.var_map[name] = var
            if new:
                self.new_arg(name, var)
            else:
                self.pending_args.append((name, var))
        return self.var_map[name]

    def flush_pending_vars(self):
        for name, var in self.pending_args:
            self.new_arg(name, var)
        self.pending_args.clear()

    def new_arg(self, name: str, arg: tir.Var | tir.Buffer) -> Tuple[int, Any]:
        idx = self.arg_idx
        self.arg_idx += 1
        self._params.append(arg)
        return idx, tb.arg(name, arg)

    def _convert_shape(self, shape, shape_schema, name_hint) -> Tuple[Any, ...]:
        if shape_schema is None:
            return tuple(shape)
        else:
            shape = list(shape)
            assert len(shape) == len(shape_schema), "Expected a tensor with matching rank"
            for i, dim in enumerate(shape_schema):
                if isinstance(dim, DynSchema):
                    var_name = dim.name or f"{name_hint}_{i}"
                    shape[i] = self.get_param(var_name, cvt_dtype(int), new=False)
            return tuple(shape)

    def arg(self, name: str, expr: Any, annot: Any) -> Any:
        if isinstance(annot, DynSchema):
            return self.get_param(annot.name or name, ty=cvt_dtype(annot.ty))
        elif isinstance(annot, StridedTensorSchema):
            assert isinstance(expr, torch.Tensor), "Expected a tensor argument"
            ptr = tir.Var(f"{name}_handle", get_ptr_type(expr.dtype))
            if annot.shape is None:
                shape = tuple(expr.shape)
                stride = tuple(expr.stride())
            else:
                shape = self._convert_shape(expr.shape, annot.shape, name_hint=f"{name}_shape")
                stride = self._convert_shape(
                    expr.stride(), annot.stride, name_hint=f"{name}_stride")
            buffer = tir.decl_buffer(
                name=name,
                shape=shape,
                strides=stride,
                dtype=cvt_dtype(expr.dtype),
                data=ptr,
                scope="global",
            )
            _, arg = self.new_arg(name, buffer)
            self.flush_pending_vars()
            if self.default_device is None:
                self.default_device = expr.device
            return Buffer_(
                buffer=arg,
                shape=shape,
                stride=stride,
                dtype=arg.dtype,
                device=expr.device,
            )
        else:
            assert not isinstance(expr, torch.Tensor), "Tensor argument must be annotated"
            return expr

    def bind(self, name: str, expr: Any, annot: Any = None) -> Any:
        if isinstance(expr, MakeEmpty):
            ptr = tir.Var(f"{name}_handle", get_ptr_type(expr.dtype))
            buffer = tir.decl_buffer(
                name=name,
                shape=expr.shape,
                strides=expr.stride,
                dtype=cvt_dtype(expr.dtype),
                data=ptr,
                scope="global",
            )
            arg_idx, arg = self.new_arg(name, buffer)
            self.flush_pending_vars()
            result = Buffer_(
                buffer=arg,
                shape=expr.shape,
                stride=expr.stride,
                dtype=arg.dtype,
                arg_idx=arg_idx,
                device=expr.device or self.default_device,
            )
            self.global_allocs.append(result)
            return result
        elif isinstance(expr, (tir.Var, tir.IntImm, tir.FloatImm)):  # fast shortcut
            return expr
        elif isinstance(expr, tir.PrimExpr):
            var = tir.Var(name=name, dtype=expr.dtype)
            self.push_frame(tb.let(var, expr))
            return var
        else:
            return expr

    def eval(self, expr: Any):
        if isinstance(expr, tir.PrimExpr):
            tb.evaluate(expr)

    def ret(self, val: Any = None) -> Any:
        if isinstance(val, Buffer_):
            val = (val,)
        for v in val:
            assert isinstance(v, Buffer_), "Expected a buffer to return"
            assert v.arg_idx is not None, "Expected a make_empty buffer to return"
            self.outs.append(v)

    def assign(self, lval: Any, slice: Any, rval: Any) -> Any:
        if isinstance(slice, (tir.PrimExpr, int)):
            if isinstance(lval, Buffer_):
                lval = lval.buffer
            tb.buffer_store(lval, rval, slice)
        else:
            raise NotImplementedError()

    def ctx_if(self, cond: bool | tir.PrimExpr):
        return self.with_frame(tb.If(cond))

    def ctx_then(self):
        with self.with_frame(tb.Then()):
            yield

    def ctx_else(self):
        with self.with_frame(tb.Else()):
            yield

    def ctx_for(self, var_names, for_range):
        if isinstance(for_range, tb.frame.ForFrame):
            with self.with_frame(for_range):
                if len(for_range.vars) > 1:
                    results = []
                    for name, var in zip(var_names, for_range.vars):
                        v = tir.Var(name, var.dtype)
                        self.push_frame(tb.let(v, var))
                        results.append(v)
                    yield tuple(results)
                else:
                    v = tir.Var(var_names, for_range.vars[0].dtype)
                    self.push_frame(tb.let(v, for_range.vars[0]))
                    yield v
        else:
            return for_range

    def ctx(self, ctx):
        return ctx

    def logical_and(self, val, rclosure) -> bool:
        return (tir.And(val, rclosure()) if isinstance(val, tir.PrimExpr) else (val or rclosure()))

    def logical_or(self, val, rclosure) -> bool:
        return (tir.Or(val, rclosure()) if isinstance(val, tir.PrimExpr) else (val or rclosure()))


def _remove_leading_ident(source: str):
    lines = source.splitlines()
    if not lines:
        return source
    ident_size = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join([line[ident_size:] if len(line) >= ident_size else line for line in lines])


def make_prim_func_generator(func):
    source = inspect.getsource(func)
    source = _remove_leading_ident(source)
    tree = ast.parse(source)
    tree = DSLMutator().visit(tree)
    tree = ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    # print(source) to show the generated code
    line_cache_name = f"<{func.__name__}_{id(source)}>"
    linecache.updatecache(line_cache_name, io.StringIO(source))
    compiled = compile(source, filename=line_cache_name, mode="exec")
    locs = {}
    exec(compiled, func.__globals__, locs)
    fn = locs["__closure"]
    fn.__tl_code__ = source
    return fn


_thread_local_storage = threading.local()


def set_current_builder(builder=None):
    _thread_local_storage.builder = builder


def current_builder() -> DSLBuilder:
    return _thread_local_storage.builder


def set_pass_configs(configs: Dict[str, Any]):
    current_builder().pass_configs.update(configs)


def get_pass_configs() -> Dict[str, Any]:
    return current_builder().pass_configs


def set_compile_flags(flags: List[str]):
    current_builder().compile_flags = flags


def add_compile_flags(flags: List[str]):
    current_builder().compile_flags.extend(flags)


def get_compile_flags() -> List[str]:
    return current_builder().compile_flags


def get_target_host() -> Target:
    return current_builder().target_host


def get_target() -> Target:
    return current_builder().target


def get_params() -> List[tir.Var]:
    return current_builder()._params


def get_global_allocs() -> List[Buffer_]:
    return current_builder().global_allocs
