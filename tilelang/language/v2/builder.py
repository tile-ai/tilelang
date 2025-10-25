from __future__ import annotations

from contextlib import contextmanager
import functools
from tilelang.language.kernel import KernelLaunchFrame
from tvm.ffi.container import Map
from tvm.ir.base import Span
from .ast import BaseBuilder, eval_op, mutate
import tvm
from tvm.tir import Buffer
from tvm.script.ir_builder import tir, IRBuilder
from tvm.tir.expr import EqualOp, NotEqualOp, PrimExpr
from typing import Callable, ContextManager, Any, Generic, ParamSpec, Self, TypeVar
import threading
import logging

logger = logging.getLogger(__name__)


def unwrap_expr(expr) -> PrimExpr | int | float:
    if isinstance(expr, tir.meta_var):
        expr = expr.value
    elif isinstance(expr, Buffer) and expr.scope() == 'local.var':
        expr = tir.BufferLoad(expr, indices=[0])
    elif isinstance(expr, (EqualOp, NotEqualOp)):
        expr = expr.asobject()
    elif isinstance(expr, tir.IntImm) and expr.dtype == 'int32':
        expr = expr.value
    return expr


def unwrap_cond(expr):
    expr = unwrap_expr(expr)
    if isinstance(expr, PrimExpr):
        return expr
    elif isinstance(expr, Buffer):
        raise TypeError(f"Buffer `{expr}` cannot be used as condition directly.")
    elif isinstance(expr, (int, bool, tuple, list)):
        return expr
    else:
        logger.warning(f"Python expression `{expr}` is used in TileLang. ", stack_info=True)
        return expr


thread_local_storage = threading.local()


class DummyFrame:

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...


class MacroFrame(DummyFrame):
    ...


class BoolOpFrame(DummyFrame):
    ...


class ConstIfFrame(DummyFrame):
    ...


class BlockFrame(DummyFrame):
    ...


AnyFrame = tir.frame.IRBuilderFrame | DummyFrame

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


class Builder(BaseBuilder):

    def __init__(self, arg_annot: dict[str, Any]):
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

    @contextmanager
    def with_frame(self, frame: ContextManager | None):
        pop_idx = len(self.frames)
        yield self.enter_frame(frame)
        while len(self.frames) > pop_idx:
            self.frames.pop().__exit__(None, None, None)

    class _has_if_frame:
        ...

    def ctx_if(self, cond):
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
            self.enter_frame(tir.evaluate(tvm.tir.const(val)))
        elif isinstance(val, str):
            pass
        elif isinstance(val, tvm.tir.stmt.BufferStore):
            self.enter_frame(tir.buffer_store(val.buffer, val.value, val.indices, val.predicate))
        else:
            raise TypeError(f"Unsupported eval value: {val} of type {type(val)}")

    def ctx_for(self, it):
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
        raise RuntimeError("continue is not supported in TileLang builder")

    def ctx_break(self):
        raise RuntimeError("break is not supported in TileLang builder")

    def ctx_while(self, cond):
        raise RuntimeError("while loops are not supported in TileLang builder")

    def bind(self, name, value):
        if name == '_':
            return value
        locals = self.get_parent_locals()
        orig_value = locals.get(name, None)
        # handle var
        if isinstance(orig_value, Buffer) and orig_value.scope() == 'local.var':
            tir.buffer_store(orig_value, value, 0)
            return orig_value
        res = self.bind_immutable(name, value)
        frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
        assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
        self.name_inside_frame[name] = self.frames[frame]
        return res

    def bind_immutable(self, name, value):
        if isinstance(value, tir.meta_var):
            return value.value
        elif isinstance(value, tir.frame.IRBuilderFrame):
            return self.enter_frame(value)
        elif isinstance(value, (Buffer, tir.IterVar, tir.Var)):
            IRBuilder.name(name, value)
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

    def assign_slice(self, lval: Any, sl: slice, value: Any):
        if isinstance(lval, Buffer):
            tir.buffer_store(lval, value, sl)
        else:
            return super().assign_slice(lval, sl, value)

    def aug_assign(self, op, target, aug_value):
        if isinstance(target, Buffer) and target.scope() == 'local.var':
            tir.buffer_store(target, eval_op(op, target, aug_value), 0)
        if isinstance(target, Buffer):
            raise RuntimeError("Augmented assignment is not supported for Buffer")
        else:
            return super().aug_assign(op, target, aug_value)

    def aug_assign_slice(self, op, target, sl, aug_value):
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
        last_macro = self.find_frame_idx(MacroFrame)
        if last_macro is not None:
            frame = self.find_frame_idx(TIR_CONTROL_FRAME, start=last_macro)
            if frame is not None:
                raise NotImplementedError(
                    "Return from control flow is not supported yet. "
                    "You can't return inside `if`, `for`, `while` blocks in a macro. "
                    "You should allocate a var before the control flow, assign value inside the blocks, "
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
        return super().ret(value)

    def ctx_with(self, ctx):
        if isinstance(ctx, tir.frame.IRBuilderFrame):
            return self.with_frame(ctx)
        else:
            return super().ctx_with(ctx)

    def assert_expr(self, cond, msg):
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
                    f"Use variable `{name}` outside its defining region, defined in frame: {frame}, current frames: {self.frames}."
                )
        if isinstance(value, tir.IntImm):
            return value.value
        if isinstance(value, Buffer) and value.scope() == 'local.var':
            return tir.BufferLoad(value, indices=[0])
        return super().rval(name, value)

    def arg(self, name, value):
        if self.find_frame_idx(MacroFrame) is not None:
            return value
        else:
            annot = self.arg_annot[name]
            if callable(annot):
                annot = annot()
            return tir.arg(name, annot)

    def override(self, name: str):
        if name == 'range':
            return tir.serial
        raise ValueError(f'Unknown override: {name}')


_P = ParamSpec('_P')
_T = TypeVar('_T')


class PrimFunc(Generic[_P, _T], tvm.tir.PrimFunc):
    params: list[tvm.tir.Var | tvm.tir.Buffer]
    body: tvm.tir.Stmt
    ret_type: tvm.ir.Type
    buffer_map: Map[tvm.tir.Var, tvm.tir.Buffer]
    attrs: tvm.Attrs | None
    span: Span | None


def macro(func: Callable[_P, _T]) -> PrimFunc[_P, _T]:
    ir_gen = mutate(func)

    @functools.wraps(func)
    def macro_wrapper(*args, **kwargs):
        builder = Builder.current()
        with builder.macro(func.__name__):
            res = ir_gen(builder)(*args, **kwargs)
        return res

    return macro_wrapper


def prim_func(func: Callable[_P, _T]) -> PrimFunc[_P, _T]:
    # hints = get_type_hints(func)
    hints = func.__annotations__
    ir_gen = mutate(func)
    builder = Builder(hints)
    with builder.prim_func(func.__name__):
        ir_gen(builder)(*hints)
    res = builder.get()
    res.ir_gen = ir_gen
    return res
