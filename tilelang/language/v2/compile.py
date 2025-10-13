from __future__ import annotations
from hashlib import sha256
import inspect
import torch
from dataclasses import dataclass, field
from tilelang.language.kernel import KernelLaunchFrame
from tvm import tir
import linecache
import io
from .ast_rewrite import DSLMutator, OpKind
from .lang import (DynSchema, StridedTensorSchema, ConstSchema, MakeEmpty, Place, _param)
from tilelang.language.dtypes import (get_tvm_dtype, get_torch_dtype, get_tvm_ptr_type,
                                      get_cffi_dtype, get_ctypes_dtype)
from tilelang.language.tir import assume as tl_assume
from tilelang.transform.pass_config import PassConfigKey
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
import tilelang.env as env
from tvm import DataType
from tvm.target import Target
from contextlib import contextmanager
import ast

_P = ParamSpec('_P')
_T = TypeVar("_T")


class JITPyFunc(Protocol[_P, _T]):

    def __call__(self, *args: _P.args, **kws: _P.kwargs) -> _T:
        ...

    __tl_code__: str


@dataclass(frozen=True, slots=True)
class JITArgParser(Generic[_P, _T]):
    parser: JITPyFunc[_P, Tuple[Tuple, Tuple]]
    const_arg_names: List[str]
    dyn_arg_names: List[str]
    sig: inspect.Signature


def disk_compile(source, name):
    cache_dir = env.TILELANG_CACHE_DIR
    if cache_dir is not None:
        import os
        save_dir = os.path.join(cache_dir, "py-cache")
        os.makedirs(save_dir, exist_ok=True)
        hash_sfx = sha256(source.encode('utf-8')).hexdigest()[:8]
        path = os.path.join(save_dir, f"{name}.{hash_sfx}.py")
        with open(path, 'w') as f:
            f.write(source)
    linecache.cache[path] = (len(source), None, source.splitlines(), path)
    return compile(source, path, "exec")


def generate_arg_parser(fn_name: str, func: Callable[_P, _T]) -> JITArgParser:
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
                    f", = {name}.shape")
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
                    f", = {name}.stride()")
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
    code = disk_compile(source, fn_name)
    exec(code, {}, locs)
    fn = locs["parse_args"](**closure)
    fn.__tl_code__ = source
    return JITArgParser(fn, tup_const, tup_dyn, sig)


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


@dataclass(frozen=True, slots=True)
class BufferSchema:
    name: Optional[str]
    shape: Tuple[int | tir.PrimExpr, ...]
    stride: Tuple[int | tir.PrimExpr, ...]
    dtype: DataType
    arg_idx: Optional[int] = None
    # device: Optional[torch.device] = None

    @classmethod
    def from_buffer(self, buffer: tir.Buffer) -> BufferSchema:
        if not hasattr(buffer, "arg_idx"):
            raise RuntimeError("Trying to return a local buffer")
        return BufferSchema(
            name=buffer.name,
            shape=buffer.shape,
            stride=buffer.strides,
            dtype=buffer.dtype,
            arg_idx=buffer.arg_idx,
        )


@dataclass(frozen=True, slots=True)
class JITFunc(Generic[_P, _T]):
    '''JITFunc is the IR generated from the kernel source
    It stores all information required to compile the kernel
    '''
    target: Target
    target_host: Target
    global_allocs: List[BufferSchema]
    outs: List[BufferSchema]
    pass_configs: Dict[PassConfigKey, Any]
    compile_flags: List[str]
    arg_parser: Callable[_P, Tuple[Tuple, Tuple]]
    const_args: Tuple[Any, ...]
    prim_func: tir.PrimFunc

    @property
    def out_idx(self) -> List[int]:
        return [i.arg_idx for i in self.outs]

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
                params.append(get_ctypes_dtype(x.dtype))
            else:
                raise RuntimeError(f"Unsupported argument type: {type(x)}")
        params.append(ctypes.c_void_p)
        return (ctypes.c_int, params)

    def get_cffi_sig(self) -> str:
        params = []
        for x in self.prim_func.params:
            if isinstance(x, tir.Var):
                cffi_type = get_cffi_dtype(x.dtype)
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
                name = f"__{allocs.name}_shape_{i}"
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
            dtype_name = f"__{allocs.name}_dtype"
            device_name = f"__{allocs.name}_device"
            closure[dtype_name] = get_torch_dtype(allocs.dtype)
            closure[device_name] = lambda: torch.device("cuda")
            arg_name = call_args[allocs.arg_idx]
            tensor_name = f"{arg_name}_tensor"
            stmts.append(f"{tensor_name} = __tl_empty(" + ",".join(shape_args) +
                         f", dtype={dtype_name}, device={device_name}())")
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
        code = disk_compile(source, func.__name__ + ".wrapper")
        exec(code, {}, locs)
        fn = locs["__closure"](**closure)
        fn.__tl_code__ = source
        return fn

    def parse_args(self, *args: _P.args, **kws: _P.kwargs) -> Tuple[Tuple, Tuple]:
        return self.arg_parser(*args, **kws)

    def repr_indent(self, ident: int = 0) -> str:
        ident_str = " " * ident
        prim_func = str(self.prim_func.script())
        return (f"JITFunc(\n"
                f"{ident_str}  target={repr(self.target)},\n"
                f"{ident_str}  target_host={repr(self.target_host)},\n"
                f"{ident_str}  global_allocs={repr(self.global_allocs)},\n"
                f"{ident_str}  outs={repr(self.outs)},\n"
                f"{ident_str}  pass_configs={repr(self.pass_configs)},\n"
                f"{ident_str}  compile_flags={repr(self.compile_flags)},\n"
                f"{ident_str}  arg_parser={repr(self.arg_parser)},\n"
                f"{ident_str}  const_args={repr(self.const_args)},\n"
                f"{ident_str}  prim_func=r'''{prim_func}''',\n"
                f"{ident_str})")

    def __repr__(self):
        return self.repr_indent()


_thread_local_storage = threading.local()


@dataclass(frozen=True, slots=True)
class MacroGuard:
    builder: DSLBuilder

    def __enter__(self):
        self.builder._macro_depth += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.builder._macro_depth -= 1


@dataclass(frozen=True, slots=True)
class ConstIfFrame:
    cond: bool

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


_interp_op_func = {
    'add': lambda lval, rval: lval + rval,
    'sub': lambda lval, rval: lval - rval,
    'mul': lambda lval, rval: lval * rval,
    'matmul': lambda lval, rval: lval @ rval,
    'div': lambda lval, rval: lval / rval,
    'mod': lambda lval, rval: lval % rval,
    'pow': lambda lval, rval: lval**rval,
    'lshift': lambda lval, rval: lval << rval,
    'rshift': lambda lval, rval: lval >> rval,
    'or': lambda lval, rval: lval | rval,
    'xor': lambda lval, rval: lval ^ rval,
    'and': lambda lval, rval: lval & rval,
    'floor_div': lambda lval, rval: lval // rval,
}


def _interp_op(op: OpKind, lval: Any, rval: Any) -> Any:
    return _interp_op_func[op](lval, rval)


def _interp_aug_assign(op: OpKind, lval: Any, slice: Any, rval: Any) -> Any:
    if op == 'add':
        lval[slice] += rval
    elif op == 'sub':
        lval[slice] -= rval
    elif op == 'mul':
        lval[slice] *= rval
    elif op == 'matmul':
        lval[slice] @= rval
    elif op == 'div':
        lval[slice] /= rval
    elif op == 'mod':
        lval[slice] %= rval
    elif op == 'pow':
        lval[slice] **= rval
    elif op == 'lshift':
        lval[slice] <<= rval
    elif op == 'rshift':
        lval[slice] >>= rval
    elif op == 'or':
        lval[slice] |= rval
    elif op == 'xor':
        lval[slice] ^= rval
    elif op == 'and':
        lval[slice] &= rval
    elif op == 'floor_div':
        lval[slice] //= rval
    else:
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class DSLBuilder:
    # base pass_config or compile_flags
    pass_configs: Dict[str, Any] = field(default_factory=dict)
    compile_flags: List[str] = field(default_factory=list)

    global_allocs: List[tir.Buffer] = field(default_factory=list)
    outs: List[tir.Buffer] = field(default_factory=list)

    # used to store pending shape args, making handle arg first
    pending_args: List[(str, tir.Var)] = field(default_factory=list)
    arg_idx: int = 0

    var_map: Dict[str, tir.Var] = field(default_factory=dict)
    frames: List[Any] = field(default_factory=list)
    builder: IRBuilder = field(default_factory=IRBuilder)
    _params: List[tir.Var] = field(default_factory=list)

    # used to bind tensor params, check whether shapes, strides are different
    _param_bind_map: Dict[str, Any] = field(default_factory=dict)

    _shortcut_expr_depth: int = 0
    _macro_depth: int = 0

    def __enter__(self):
        assert not hasattr(_thread_local_storage, "builder")
        _thread_local_storage.dslbuilder = self
        self.builder.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.builder.__exit__(exc_type, exc_value, traceback)
        del _thread_local_storage.dslbuilder

    def macro(self) -> MacroGuard:
        assert self._shortcut_expr_depth == 0, \
            "It is not supported to use macro inside shortcut expression, " \
            "don't use macro in `M if xxx else M`, `M and M`, `M or M`, " \
            "please use an if statement instead"
        return MacroGuard(self)

    @staticmethod
    def current() -> DSLBuilder:
        assert hasattr(_thread_local_storage,
                       "dslbuilder"), "Access DSLBuilder in non dsl builder scope"
        return _thread_local_storage.dslbuilder

    def get_global_allocs(self) -> List[BufferSchema]:
        return [BufferSchema.from_buffer(buf) for buf in self.global_allocs]
    def get_outs(self) -> List[BufferSchema]:
        return [BufferSchema.from_buffer(buf) for buf in self.outs]
    def get(self) -> tir.PrimFunc:
        return self.builder.get()
    def get_pass_configs(self) -> Dict[PassConfigKey, Any]:
        return self.pass_configs
    def get_compile_flags(self) -> List[str]:
        return self.compile_flags

    def is_inside_device_code(self) -> bool:
        num_tvm_frames = 0
        for frame in self.frames:
            if not isinstance(frame, ConstIfFrame):
                num_tvm_frames += 1
        return num_tvm_frames > 1 # ignore prim_func frame

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

    def new_arg(self, name: str, arg: tir.Var | tir.Buffer) -> Tuple[int, tir.Var]:
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
                    shape[i] = self.get_param(var_name, get_tvm_dtype(int), new=False)
            return tuple(shape)

    def arg(self, name: str, expr: Any, annot: Any) -> Any:
        if self._macro_depth > 0:
            return expr
        if isinstance(annot, DynSchema):
            return self.get_param(annot.name or name, ty=get_tvm_dtype(annot.ty))
        elif isinstance(annot, StridedTensorSchema):
            assert isinstance(expr, torch.Tensor | Place), "Expected a tensor argument"
            ptr = tir.Var(f"{name}_handle", get_tvm_ptr_type(expr.dtype))
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
                dtype=get_tvm_dtype(expr.dtype),
                data=ptr,
                scope="global",
            )
            _, arg = self.new_arg(name, buffer)
            self.flush_pending_vars()
            return buffer
        else:
            assert not isinstance(expr, torch.Tensor), "Tensor argument must be annotated"
            return expr

    def bind(self, name: str, expr: Any, annot: Any = None) -> Any:
        if isinstance(expr, _param):
            if name not in self._param_bind_map:
                self._param_bind_map[name] = expr.data
            old_val = self._param_bind_map[name]
            if isinstance(old_val, tir.Var) and isinstance(expr.data, tir.Var):
                tl_assume(old_val == expr.data)
                return old_val
            elif isinstance(old_val, tir.Var) and isinstance(expr.data, (tir.PrimExpr, int, float)):
                tl_assume(old_val == expr.data)
                self._param_bind_map[name] = expr.data
            elif isinstance(old_val, (tir.PrimExpr, int, float)) and isinstance(expr.data, tir.Var):
                tl_assume(old_val == expr.data)
                self._param_bind_map[name] = expr.data
            elif isinstance(old_val, (tir.PrimExpr, int, float)) and isinstance(expr.data, (tir.PrimExpr, int, float)):
                if isinstance(old_val, (int, float, tir.IntImm)) and isinstance(expr.data, (int, float, tir.IntImm)):
                    if old_val != expr.data:
                        raise RuntimeError(f"Param binding failed for '{name}', new binding {expr.data} is not compatible with old binding {old_val}")
                else:
                    tl_assume(old_val == expr.data)
                self._param_bind_map[name] = expr.data
            return self._param_bind_map[name]
        elif isinstance(expr, MakeEmpty):
            if self.is_inside_device_code(): # 1 is prim_func scope
                raise RuntimeError(
                    "Trying to allocate an empty buffer in device code"
                    f"frames: {self.frames}"
                )
            ptr = tir.Var(f"{name}_handle", get_tvm_ptr_type(expr.dtype))
            buffer = tir.decl_buffer(
                name=name,
                shape=expr.shape,
                strides=expr.stride,
                dtype=get_tvm_dtype(expr.dtype),
                data=ptr,
                scope="global",
            )
            arg_idx, arg = self.new_arg(name, buffer)
            self.flush_pending_vars()
            arg.arg_idx = arg_idx
            self.global_allocs.append(arg)
            return arg
        elif isinstance(expr, (tir.Var, tir.Buffer)):  # fast shortcut
            IRBuilder.name(name, expr)
            return expr
        elif isinstance(expr, (tir.IntImm, tir.FloatImm)):
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
        if self._macro_depth > 0:
            return val
        if self.is_inside_device_code():
            raise RuntimeError("ret is not supported in device code")
        if isinstance(val, tir.Buffer):
            val = (val,)
        for v in val:
            assert isinstance(v, tir.Buffer), "Expected a buffer to return"
            assert v.arg_idx is not None, "Expected a make_empty buffer to return"
            self.outs.append(v)

    def assign(self, lval: Any, slice: Any, rval: Any) -> Any:
        if isinstance(lval, tir.Buffer):
            tb.buffer_store(lval, rval, slice)
        else:
            lval[slice] = rval

    def aug_assign(self, *args) -> Any:
        if len(args) == 3:
            op, lval, rval = args
            if isinstance(lval, tir.Buffer):
                sl = slice(None)
                tb.buffer_store(lval, _interp_op(op, lval[sl], rval), sl)
                return lval
            else:
                return _interp_op(op, lval, rval)
        elif len(args) == 4:
            op, lval, sl, rval = args
            if isinstance(lval, tir.Buffer):
                tb.buffer_store(lval, _interp_op(op, lval[sl], rval), sl)
            else:
                _interp_aug_assign(op, lval, sl, rval)

    def ctx_if(self, cond: bool | tir.PrimExpr):
        if isinstance(cond, tir.PrimExpr):
            return self.with_frame(tb.If(cond))
        else:
            return self.with_frame(ConstIfFrame(cond))

    def ctx_then(self):
        if isinstance(self.frames[-1], ConstIfFrame):
            if self.frames[-1].cond:
                yield
        else:
            with self.with_frame(tb.Then()):
                yield

    def ctx_else(self):
        if isinstance(self.frames[-1], ConstIfFrame):
            if not self.frames[-1].cond:
                yield
        else:
            with self.with_frame(tb.Else()):
                yield

    def ctx_for(self, var_names, for_range):
        if isinstance(for_range, tb.frame.ForFrame):
            with self.with_frame(for_range):
                if len(for_range.vars) > 1:
                    for name, var in zip(var_names, for_range.vars):
                        IRBuilder.name(name, var)
                    yield tuple(for_range.vars)
                else:
                    IRBuilder.name(var_names, for_range.vars[0])
                    yield for_range.vars[0]
        else:
            for item in for_range:
                yield item

    def ctx(self, ctx):
        if isinstance(ctx, KernelLaunchFrame):
            return self.with_frame(ctx)
        else:
            return ctx

    def logical_and(self, val, rclosure) -> bool:
        if isinstance(val, tir.PrimExpr):
            if isinstance(val, tir.IntImm) and not val.value:
                res = tir.IntImm('bool', False)
            self._shortcut_expr_depth += 1
            res = tir.And(val, rclosure())
            self._shortcut_expr_depth -= 1
        else:
            res = val and rclosure()
        return res

    def logical_or(self, val, rclosure) -> bool:
        if isinstance(val, tir.PrimExpr):
            if isinstance(val, tir.IntImm) and val.value:
                res = tir.IntImm('bool', True)
            self._shortcut_expr_depth += 1
            res = tir.Or(val, rclosure())
            self._shortcut_expr_depth -= 1
        else:
            res = val or rclosure()
        return res

    def ifexp(self, cond, tclosure, fclosure):
        self._shortcut_expr_depth += 1
        if isinstance(cond, tir.PrimExpr):
            if isinstance(cond, tir.IntImm):
                return tclosure() if cond.value else fclosure()
            self._shortcut_expr_depth += 1
            res = tir.IfThenElse(cond, tclosure(), fclosure())
            self._shortcut_expr_depth -= 1
            return res
        else:
            res = tclosure() if cond else fclosure()
        self._shortcut_expr_depth -= 1
        return res

    def ctx_continue(self):
        if self.is_inside_device_code():
            raise RuntimeError("continue is not supported in device code")
        return True

    def ctx_break(self):
        if self.is_inside_device_code():
            raise RuntimeError("break is not supported in device code")
        return True


def _remove_leading_ident(source: str):
    lines = source.splitlines()
    if not lines:
        return source
    ident_size = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join([line[ident_size:] if len(line) >= ident_size else line for line in lines])


def make_ir_generator(func):
    _, start = inspect.getsourcelines(func)
    filename = inspect.getsourcefile(func) or inspect.getfile(func)
    source = inspect.getsource(func)
    source = _remove_leading_ident(source)
    tree = ast.parse(source, filename=filename)
    ast.increment_lineno(tree, start - 1)
    tree = DSLMutator().visit(tree)
    tree = ast.fix_missing_locations(tree)
    tl_source = ast.unparse(tree)
    linecache.cache[filename] = (len(source), None, source.splitlines(), filename)
    compiled = compile(tree, filename=filename, mode="exec")
    locs = {}
    exec(compiled, func.__globals__, locs)
    fn = locs["__closure"]
    fn.__tl_code__ = tl_source
    fn.__name__ = func.__name__
    return fn


def make_prim_func_generator(func):
    gen = make_ir_generator(func)
    name = func.__name__

    def inner(builder: DSLBuilder, *args, **kws):
        with builder, builder.prim_func():
            builder.func_name(name)
            return gen(builder)(*args, **kws)

    inner.__tl_code__ = gen.__tl_code__
    return inner


def make_macro_generator(func):
    gen = make_ir_generator(func)

    def inner(*args, **kws):
        builder = DSLBuilder.current()
        with builder.macro():
            return gen(builder)(*args, **kws)

    inner.__tl_code__ = gen.__tl_code__
    return inner


def set_pass_configs(configs: Dict[PassConfigKey, Any]):
    DSLBuilder.current().pass_configs.update(configs)


def get_pass_configs() -> Dict[PassConfigKey, Any]:
    return DSLBuilder.current().pass_configs


def set_compile_flags(flags: List[str]):
    DSLBuilder.current().compile_flags = flags


def add_compile_flags(flags: List[str]):
    DSLBuilder.current().compile_flags.extend(flags)


def get_compile_flags() -> List[str]:
    return DSLBuilder.current().compile_flags


def get_params() -> List[tir.Var]:
    return DSLBuilder.current()._params

