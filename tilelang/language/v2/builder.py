from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect

import torch
from tilelang.language.kernel import KernelLaunchFrame
from tvm.ffi.container import Map
from tvm.ir.base import Span
from .ast import BaseBuilder, eval_op, mutate
import tvm
from tvm.tir import Buffer
from tvm.script.ir_builder import tir, IRBuilder
from tvm.tir.expr import EqualOp, FloatImm, IntImm, NotEqualOp, PrimExpr, StringImm, Var
from typing import TYPE_CHECKING, Callable, ContextManager, Any, Generic, Hashable, ParamSpec, Self, TypeVar, ForwardRef
from .dtypes import get_tvm_dtype
from types import EllipsisType
import threading
import logging

logger = logging.getLogger(__name__)


def unwrap_expr(expr) -> PrimExpr | int | float:
    '''
    unwrap expr and convert it into PrimExpr like
    '''
    if isinstance(expr, tir.meta_var):
        expr = expr.value
    elif isinstance(expr, Buffer) and expr.scope() == 'local.var':
        expr = tir.BufferLoad(expr, indices=[0])
    elif isinstance(expr, (EqualOp, NotEqualOp)):
        expr = expr.asobject()
    return expr


def unwrap_cond(expr):
    '''
    unwrap expr and convert to bool condition
    '''
    expr = unwrap_expr(expr)
    if isinstance(expr, (IntImm, FloatImm, StringImm)):
        return bool(expr.value)
    elif isinstance(expr, PrimExpr):
        return expr
    elif isinstance(expr, Buffer):
        raise TypeError(f"Buffer `{expr}` cannot be used as condition directly.")
    elif isinstance(expr, (int, bool)) or expr is None:
        return bool(expr)
    else:
        logger.warning(
            f"Python expression `{expr}` is used as condition in TileLang, \n"
                 "this is treated as a constant expression. ", stack_info=True, stacklevel=3)
        return bool(expr)


thread_local_storage = threading.local()


class Frame:

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...


class MacroFrame(Frame):
    ...


class BoolOpFrame(Frame):
    ...


class ConstIfFrame(Frame):
    ...


class BlockFrame(Frame):
    ...


class ContinueFrame(Frame):
    ...

class BreakFrame(Frame):
    ...

ContinueOrBreak = ContinueFrame | BreakFrame
AnyFrame = tir.frame.IRBuilderFrame | Frame

TIR_CONTROL_FRAME = (
    tir.frame.WhileFrame,
    tir.frame.ForFrame,
    tir.frame.IfFrame,
    tir.frame.PrimFuncFrame,
)

TIR_VAR_SCOPE_FRAME = (
    tir.frame.WhileFrame,
    tir.frame.ForFrame,
    tir.frame.IfFrame,
    tir.frame.PrimFuncFrame,
    MacroFrame,
    KernelLaunchFrame,
)


def is_var(v: Any) -> bool:
    return isinstance(v, Buffer) and v.scope() == 'local.var'


class Builder(BaseBuilder):

    def __init__(self, arg_annot: dict[str, Any] = None):
        self.arg_annot = arg_annot
        self.frames: list[AnyFrame] = []
        self.ir_builder = IRBuilder()
        self.name_inside_frame: dict[str, AnyFrame] = {}

    @classmethod
    def current(cls) -> Self:
        builder = thread_local_storage.builder
        assert builder is not None, "No active Builder found in the current thread."
        return builder

    @contextmanager
    def prim_func(self, name):
        thread_local_storage.builder = self
        with self.ir_builder, self.with_frame(tir.prim_func()):
            tir.func_name(name)
            yield

    @contextmanager
    def macro(self, name=None):
        if self.find_frame_idx(BoolOpFrame) is not None:
            raise RuntimeError(
                f"Macro `{name}` is used inside boolean expressions, "
                "please use `if` to replace `M and M`, `M or M`, `M if xxx else M` constructs")
        save = self.name_inside_frame
        self.name_inside_frame = {}
        with self.with_frame(MacroFrame()):
            yield
        self.name_inside_frame = save

    def get(self):
        return self.ir_builder.get()

    def find_frame_idx(self, frame: type | tuple[type, ...], start=0) -> int | None:
        for idx in reversed(range(start, len(self.frames))):
            f = self.frames[idx]
            if isinstance(f, frame):
                return idx

    def enter_frame(self, frame: ContextManager):
        self.frames.append(frame)
        return frame.__enter__()

    def check_continue_break(self):
        idx = self.find_frame_idx(ContinueOrBreak)
        if idx is not None:
            logger.warning(
                'Writing code after continue/break may cause undefined behavior in tilelang.',
                stack_info=True,
                stacklevel=3
            )

    @contextmanager
    def with_frame(self, frame: ContextManager | None):
        pop_idx = len(self.frames)
        yield self.enter_frame(frame)
        while len(self.frames) > pop_idx:
            self.frames.pop().__exit__(None, None, None)

    class _has_if_frame:
        ...

    def ctx_if(self, cond):
        self.check_continue_break()
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            with self.with_frame(tir.If(cond)):
                yield self._has_if_frame
        else:
            with self.with_frame(ConstIfFrame()):
                yield cond

    def ctx_then(self, val):
        if val is self._has_if_frame:
            with self.with_frame(tir.Then()):
                yield
        else:
            with self.with_frame(BlockFrame()):
                if val:
                    yield

    def ctx_else(self, val):
        if val is self._has_if_frame:
            with self.with_frame(tir.Else()):
                yield
        else:
            with self.with_frame(BlockFrame()):
                if not val:
                    yield

    def eval(self, val: Any):
        val = unwrap_expr(val)
        if val is None:
            pass
        elif isinstance(val, tir.frame.IRBuilderFrame):
            self.enter_frame(val)
        elif isinstance(val, PrimExpr):
            tir.evaluate(val)
        elif isinstance(val, (int, bool)):
            tir.evaluate(tvm.tir.const(val))
        elif isinstance(val, str):
            pass
        elif isinstance(val, tvm.tir.stmt.BufferStore):
            tir.buffer_store(val.buffer, val.value, val.indices, val.predicate)
        else:
            raise TypeError(f"Unsupported eval value: {val} of type {type(val)}")

    def ctx_for(self, it):
        self.check_continue_break()
        it = unwrap_expr(it)
        if isinstance(it, range):
            assert it.step == 1, "Only step=1 is supported in range for now."
            it = tir.serial(it.start, it.stop)
        if not isinstance(it, tir.frame.ForFrame):
            raise TypeError(
                f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding")
        with self.with_frame(it) as v:
            yield v

    def ctx_continue(self):
        self.check_continue_break()
        self.enter_frame(ContinueFrame())
        raise RuntimeError("continue is not supported in TileLang builder")

    def ctx_break(self):
        self.check_continue_break()
        self.enter_frame(BreakFrame())
        raise RuntimeError("break is not supported in TileLang builder")

    def ctx_while(self, cond):
        self.check_continue_break()
        raise RuntimeError("while loops are not supported in TileLang builder")

    def bind(self, name, value, annot=BaseBuilder.empty):
        self.check_continue_break()
        locals = self.get_parent_locals()
        orig_value = locals.get(name, None)
        # annotation like tl.float32
        if callable(annot):
            annot_val = annot()
            if isinstance(annot_val, tir.Var):
                orig_value = tir.alloc_buffer((1,), dtype=annot_val.dtype, scope='local.var')
                IRBuilder.name(name, orig_value)
                if isinstance(value, EllipsisType) or value is self.empty:
                    return orig_value
                elif isinstance(value, (int, float, IntImm, FloatImm)):
                    tir.block_attr(
                        {'tl.local_var_init': {
                            orig_value.data: tvm.runtime.convert(value)
                        }})
                    return orig_value
        # if orig_value is a local.var, we use buffer_store to modify it immutably
        #   however, if rvalue is also a local.var, this is a new binding,
        #   we should not use buffer_store, and bind it instead
        #   ```py
        #   a = tl.alloc_var('float32')  # bind var `a`
        #   a = tl.alloc_var('float32')  # bind a new var `a_1`
        #   b = a                        # get value of var `b = a_1[0]``
        #   c = tl.alloc_var('float32')  # bind var `c`
        #   c = a                        # get and assign `c[0] = a_1[0]`
        #   ```
        if is_var(orig_value) and not is_var(value):
            tir.buffer_store(orig_value, value, 0)
            return orig_value
        res = self.bind_immutable(name, value)
        if name != '_':
            frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
            assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
            self.name_inside_frame[name] = self.frames[frame]
        return res

    def unwrap_value(self, value):
        value = unwrap_expr(value)
        # handle bx, by = tl.Kernel(128, 128), rval is frame
        if isinstance(value, tir.frame.IRBuilderFrame):
            return self.enter_frame(value)
        else:
            return value

    def bind_immutable(self, name, value):
        if isinstance(value, tir.meta_var):
            return value.value
        elif isinstance(value, tir.frame.IRBuilderFrame):
            return self.enter_frame(value)
        elif isinstance(value, (Buffer, tir.IterVar, tir.Var)):
            IRBuilder.name(name, value)
            return value
        elif isinstance(value, (tuple, list, tvm.ffi.Array)):
            return value
        else:
            try:
                value = tvm.runtime.convert(value)
            except TypeError:
                return value
            frame = tir.LetStmt(value)
            var = frame.var
            IRBuilder.name(name, var)
            return self.enter_frame(frame)

    def assign_slice(self, lval: Any, sl: slice, value: Any, annot=BaseBuilder.empty):
        self.check_continue_break()
        if annot is not self.empty:
            logger.warning(
                "Type annotation in slice assignment has no effect", stack_info=True, stacklevel=2)
        if isinstance(lval, Buffer):
            tir.buffer_store(lval, value, sl)
        else:
            return super().assign_slice(lval, sl, value)

    def aug_assign(self, op, target, aug_value):
        self.check_continue_break()
        if is_var(target):
            tir.buffer_store(target, eval_op(op, target[0], aug_value), 0)
        elif isinstance(target, Buffer):
            raise RuntimeError("Augmented assignment is not supported for Buffer")
        else:
            return super().aug_assign(op, target, aug_value)

    def aug_assign_slice(self, op, target, sl, aug_value):
        self.check_continue_break()
        if isinstance(target, Buffer):
            tir.buffer_store(target, eval_op(op, target[sl], aug_value), sl)
        else:
            return super().aug_assign_slice(op, target, sl, aug_value)

    def boolop(self, op, left, right):
        left = unwrap_cond(left)
        if isinstance(left, PrimExpr):
            with self.with_frame(BoolOpFrame()):
                if op == 'And':
                    return tir.And(left, right())
                if op == 'Or':
                    return tir.Or(left, right())
            raise RuntimeError(f"Unsupported boolean operator: {op}")
        else:
            return super().boolop(op, left, right)

    def ifexp(self, cond, then, otherwise):
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            with self.with_frame(BoolOpFrame()):
                return tir.if_then_else(cond, then(), otherwise())
        else:
            return super().ifexp(cond, then, otherwise)

    def ret(self, value):
        self.check_continue_break()
        # handle return T.alloc_var()
        value = self.unwrap_value(value)
        last_macro = self.find_frame_idx(MacroFrame)
        if last_macro is not None:
            frame = self.find_frame_idx(TIR_CONTROL_FRAME, start=last_macro)
            if frame is not None:
                raise NotImplementedError(
                    "Return from control flow is not supported yet. \n"
                    "You should allocate a var before the control flow, assign value inside the blocks, \n"
                    "and return the var after the control flow. i.e.\n"
                    "```\n"
                    "@T.macro\n" \
                    "def my_macro(cond):\n"
                    "    a: T.float16 = ...\n"
                    "    if cond:\n"
                    "        a = 1.0\n"
                    "    return a\n"
                    "```"
                )
        return value

    def ctx_with(self, ctx):
        self.check_continue_break()
        if isinstance(ctx, tir.frame.IRBuilderFrame):
            return self.with_frame(ctx)
        else:
            return super().ctx_with(ctx)

    def assert_expr(self, cond, msg):
        self.check_continue_break()
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            self.enter_frame(tir.Assert(cond, msg))
        else:
            super().assert_expr(cond, msg)

    def rval(self, name: str, value: Any) -> Any:
        if name in self.name_inside_frame:
            frame = self.name_inside_frame[name]
            if frame not in self.frames:
                raise RuntimeError(
                    f"Use immutable variable `{name}` outside its defining region, did you forget **alloc_var**?\n"
                    f"variable `{name}` is defined in frame: {frame}, current frames: {self.frames}."
                )
        return self.unwrap_value(value)

    def arg(self, name, value):
        if self.find_frame_idx(MacroFrame) is not None:
            return value
        if isinstance(value, (Buffer, Var)):
            return tir.arg(name, value)
        elif hasattr(value, '__tl_arg__'):
            return value.__tl_arg__(name, self)
        elif isinstance(value, Hashable):
            return value
        else:
            raise TypeError(f"Unsupported argument type: {type(value)} for argument `{name}`.")

    def override(self, name: str):
        if name == 'range':
            return tir.serial
        raise ValueError(f'Unknown override: {name}')


def __torch_tensor_tl_arg__(self: torch.Tensor, name: str, builder: Builder):
    buffer = tir.buffer(
        self.shape, get_tvm_dtype(self.dtype), strides=self.stride(), scope='global')
    return tir.arg(name, buffer)


torch.Tensor.__tl_arg__ = __torch_tensor_tl_arg__

_P = ParamSpec('_P')
_T = TypeVar('_T')


@dataclass
class IRGenerator(Generic[_P, _T]):
    gen: Callable[[BaseBuilder], Callable[_P, _T]]
    source: str


if TYPE_CHECKING:

    class PrimFunc(Generic[_P, _T], tvm.tir.PrimFunc):
        params: list[tvm.tir.Var | tvm.tir.Buffer]
        body: tvm.tir.Stmt
        ret_type: tvm.ir.Type
        buffer_map: Map[tvm.tir.Var, tvm.tir.Buffer]
        attrs: tvm.Attrs | None
        span: Span | None
        ir_gen: IRGenerator[_P, _T] | None
        source: str | None
        orig_func: Callable[_P, _T] | None
else:
    PrimFunc = tvm.tir.PrimFunc


@dataclass
class Macro(Generic[_P, _T]):
    name: str
    orig_func: Callable[_P, _T]
    ir_gen: IRGenerator[_P, _T]

    @property
    def source(self) -> str:
        return self.ir_gen.source

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        builder = Builder.current()
        with builder.macro(self.name):
            res = self.ir_gen.gen(builder)(*args, **kwargs)
        return res


def build_ir_generator(func: Callable[_P, _T]) -> IRGenerator[_P, _T]:
    ir_gen = mutate(func)
    ir_gen = IRGenerator(gen=ir_gen, source=ir_gen.__source__)
    return ir_gen


def macro(func: Callable[_P, _T]) -> Macro[_P, _T]:
    return Macro(name=func.__name__, orig_func=func, ir_gen=build_ir_generator(func))


from typing import _eval_type


def get_type_hints(func):
    annot = getattr(func, '__annotations__', None)
    if annot is None:
        raise TypeError(f'Failed to get function type hints, {func} is not a function')
    hints = {}
    type_params = getattr(func, "__type_params__", ())
    globalns = getattr(func, '__globals__', {})
    localns = globalns
    for name, value in annot.items():
        if isinstance(value, tvm.DataType):
            hints[name] = value
            continue
        if value is None:
            value = type(None)
        if isinstance(value, str):
            value = ForwardRef(value, is_argument=True, is_class=False)

        hints[name] = _eval_type(value, globalns=globalns, localns=localns, type_params=type_params)
    return hints


def prim_func(func: Callable[_P, _T]) -> PrimFunc[_P, _T]:
    sig = inspect.signature(func)
    annot = get_type_hints(func)
    args = []
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.annotation is not param.empty:
            if callable(param.annotation):
                value = param.annotation()
            else:
                value = param.annotation
        elif param.default is not param.empty:
            value = param.default
        else:
            value = Builder.empty
        if param.kind == param.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    ir_gen = build_ir_generator(func)
    builder = Builder(annot)
    with builder.prim_func(func.__name__):
        ir_gen.gen(builder)(*args, **kwargs)
    res = builder.get()
    res.ir_gen = ir_gen
    res.source = ir_gen.source
    res.orig_func = func
    return res
