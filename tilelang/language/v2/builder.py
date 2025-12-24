from __future__ import annotations
from contextlib import contextmanager, AbstractContextManager
from dataclasses import dataclass
import inspect

from tilelang.language.kernel import KernelLaunchFrame
from tvm_ffi.container import Map
from tvm.ir.base import Span
from tvm.ir.expr import Range
from tvm.tir.stmt import BufferRegion
from .ast import BaseBuilder, IRGenerator, eval_op, mutate
from .utils import construct_strides
import tvm
from tvm.tir import Buffer
from tvm.script.ir_builder import tir, IRBuilder

from tvm.tir.expr import BufferLoad, EqualOp, FloatImm, IntImm, NotEqualOp, PrimExpr, StringImm, Var
from typing import TYPE_CHECKING, Callable, Any, Generic, TypeVar, ForwardRef, Union
from collections.abc import Sequence
from .annot import FuncAnnot, ArgVarTable, Annot
import pprint

# Python 3.9 compatibility for ParamSpec and Self
try:
    from typing import ParamSpec, Self
except ImportError:  # Python < 3.11 for Self, < 3.10 for ParamSpec
    from typing_extensions import ParamSpec, Self
from . import dtypes as dt
from . import utils
import threading
import logging

logger = logging.getLogger(__name__)


def unwrap_expr(expr) -> PrimExpr | int | float:
    """
    unwrap expr and convert it into PrimExpr like
    """
    if isinstance(expr, tir.meta_var):
        expr = expr.value
    elif isinstance(expr, Ref):
        return expr.load()
    elif is_var(expr):
        expr = tir.BufferLoad(expr, indices=[0])
    elif isinstance(expr, (EqualOp, NotEqualOp)):
        expr = expr.asobject()
    return expr


def unwrap_cond(expr):
    """
    unwrap expr and convert to bool condition
    """
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
            f"Python expression `{expr}` is used as condition in TileLang, \nthis is treated as a constant expression. ",
            stack_info=True,
            stacklevel=3,
        )
        return bool(expr)


thread_local_storage = threading.local()


class Frame:
    """
    Frame are virtual context managers used in frontend only
    They do not have any runtime representation in the generated TIR.
    """

    def __enter__(self): ...

    def __exit__(self, exc_type, exc_value, traceback): ...


class MacroFrame(Frame): ...


class ExitedMacroFrame(Frame): ...


class BoolOpFrame(Frame): ...


class ContinueFrame(Frame): ...


class BreakFrame(Frame): ...


@dataclass
class SerialForWithStep:
    start: PrimExpr
    stop: PrimExpr
    step: PrimExpr
    annotations: dict[str, Any] | None = None


@dataclass
class OutTensor:
    shape: Sequence[PrimExpr]
    dtype: dt.dtype

    @property
    def strides(self):
        return construct_strides(tuple(self.shape))


@dataclass
class Ref:
    bufload: BufferLoad

    @property
    def buffer(self):
        return self.bufload.buffer

    def store(self, value):
        tir.buffer_store(self.bufload.buffer, value, self.bufload.indices)

    def load(self):
        return self.bufload


class UnrollForWithStep(SerialForWithStep): ...


# Python 3.9 compatibility: avoid PEP 604 unions at runtime
# Use tuple for isinstance checks and typing.Union for annotations/aliases
ContinueOrBreak = (ContinueFrame, BreakFrame)
AnyFrame = Union[tir.frame.IRBuilderFrame, Frame]

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
    return isinstance(v, Buffer) and v.scope() == "local.var"


class Builder(BaseBuilder):
    def __init__(self, func_annot: FuncAnnot = None):
        self.frames: list[AnyFrame] = []
        self.ir_builder = IRBuilder()
        self.name_inside_frame: dict[str, AnyFrame] = {}
        self.macro_arg_annot = {}
        self.func_annot = func_annot
        self.out_idx = []
        self.out_tensor_cnt = 0
        self.arg_vt = ArgVarTable()

    @classmethod
    def current(cls) -> Self:
        builder = getattr(thread_local_storage, "builder", None)
        return builder

    @contextmanager
    def prim_func(self, name):
        thread_local_storage.builder = self
        with self.ir_builder, self.with_frame(tir.prim_func()):
            tir.func_name(name)
            yield
        if len(self.out_idx) != self.out_tensor_cnt:
            raise RuntimeError("Not all tensor allocated from `T.empty` are returned")

    @contextmanager
    def macro(self, name=None, annotations=None):
        if self.find_frame_idx(BoolOpFrame) is not None:
            raise RuntimeError(
                f"Macro `{name}` is used inside boolean expressions, "
                "please use `if` to replace `M and M`, `M or M`, `M if xxx else M` constructs"
            )
        save = self.name_inside_frame, self.macro_arg_annot
        self.name_inside_frame = {}
        self.macro_arg_annot = annotations or {}
        pos = len(self.frames)
        # here we add a ExitedMacroFrame to preserve the frame stack inside macro
        # because macro may bind some variable, and return it
        #
        # ```py
        # @T.macro
        # def foo(x):
        #    y = x + 1
        #    return y
        # @T.prim_func
        # def bar():
        #    c = foo(1) # macro generates let y = x + 1
        #    d = c # d = c should lay inside frame of `let y = x + 1`
        self.frames.append(MacroFrame())
        yield
        self.frames[pos] = ExitedMacroFrame()
        self.name_inside_frame, self.macro_arg_annot = save

    def get(self):
        return self.ir_builder.get()

    def find_frame_idx(self, frame: type | tuple[type, ...], start=0) -> int | None:
        for idx in reversed(range(start, len(self.frames))):
            f = self.frames[idx]
            if isinstance(f, frame):
                return idx

    def enter_frame(self, frame: AbstractContextManager[Any]):
        self.frames.append(frame)
        return frame.__enter__()

    def check_continue_break(self):
        idx = self.find_frame_idx(ContinueOrBreak)
        if idx is not None:
            logger.warning("Writing code after continue/break may cause undefined behavior in tilelang.", stack_info=True, stacklevel=3)

    @contextmanager
    def with_frame(self, frame: AbstractContextManager[Any] | None):
        pop_idx = len(self.frames)
        yield self.enter_frame(frame)
        while len(self.frames) > pop_idx:
            self.frames.pop().__exit__(None, None, None)

    class _has_if_frame: ...

    def ctx_if(self, cond):
        self.check_continue_break()
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            with self.with_frame(tir.If(cond)):
                yield self._has_if_frame
        else:
            yield cond

    def ctx_then(self, val):
        if val is self._has_if_frame:
            with self.with_frame(tir.Then()):
                yield
        else:
            if val:
                yield

    def ctx_else(self, val):
        if val is self._has_if_frame:
            with self.with_frame(tir.Else()):
                yield
        else:
            if not val:
                yield

    def eval(self, val: Any):
        val = unwrap_expr(val)
        if val is None:
            pass
        elif isinstance(val, tir.frame.IRBuilderFrame):
            if isinstance(val, tir.frame.ForFrame):
                logger.warning(
                    "Evaluating a for frame may cause undefined behavior in tilelang.",
                    stack_info=True,
                    stacklevel=1,
                )
            self.enter_frame(val)
        elif isinstance(val, PrimExpr):
            tir.evaluate(val)
        elif isinstance(val, (int, bool)):
            tir.evaluate(tvm.tir.const(val))
        elif isinstance(val, str):
            pass
        elif isinstance(val, tvm.tir.stmt.BufferStore):
            tir.buffer_store(val.buffer, val.value, val.indices, val.predicate)
        elif isinstance(val, (Buffer, Var)):
            pass
        else:
            logger.warning(f"Unused return value: {val}({type(val)})", stack_info=True, stacklevel=2)

    def ctx_for(self, it):
        self.check_continue_break()
        it = unwrap_expr(it)
        if isinstance(it, (SerialForWithStep, UnrollForWithStep)):
            # Validate and compute the trip count before constructing the frame
            if isinstance(it.step, (int, IntImm)):
                step_value = it.step if isinstance(it.step, int) else it.step.value
                if step_value == 0:
                    raise ValueError("Invalid stepped serial: step must be non-zero")
                if step_value > 0:
                    real_stop = tir.ceildiv(it.stop - it.start, step_value)
                else:
                    real_stop = tir.ceildiv(it.start - it.stop, -step_value)
            else:
                logger.warning(f"Using a non-constant step `{it.step}` in stepped serial may lead to undefined behavior in tilelang")
                real_stop = tir.ceildiv(it.stop - it.start, it.step)
            if isinstance(it, UnrollForWithStep):
                real_frame = tir.unroll(real_stop, annotations=it.annotations)
            elif isinstance(it, SerialForWithStep):
                real_frame = tir.serial(real_stop, annotations=it.annotations)
            else:
                raise TypeError(
                    f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                    "range, T.serial, T.unroll, T.grid, T.parallel, T.vectorized, T.thread_binding"
                )
            with self.with_frame(real_frame) as v:
                IRBuilder.name("_tmp", v)
                yield it.start + v * it.step
        else:
            if not isinstance(it, tir.frame.ForFrame):
                raise TypeError(
                    f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                    "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding"
                )
            with self.with_frame(it) as v:
                yield v

    def ctx_continue(self):
        self.check_continue_break()
        # add a dummy frame for checking code after continue/break
        self.enter_frame(ContinueFrame())
        tir.evaluate(tir.continue_loop())

    def ctx_break(self):
        self.check_continue_break()
        # add a dummy frame for checking code after continue/break
        self.enter_frame(BreakFrame())
        tir.evaluate(tir.break_loop())

    def ctx_while(self, cond):
        self.check_continue_break()
        cond_v = cond()
        cond_v_unwrap = unwrap_cond(cond_v)
        if not isinstance(cond_v_unwrap, PrimExpr):
            if cond_v_unwrap:
                raise RuntimeError(
                    f"Infinite while loop detected in TileLang\n"
                    f"Condition: {cond_v}({type(cond_v)}) => {cond_v_unwrap}({type(cond_v_unwrap)})\n"
                )
            else:
                logger.warning(
                    "While loop with constant false condition detected in Tilelang, the loop body will never be executed.\n",
                    f"Condition: {cond_v}({type(cond_v)}) => {cond_v_unwrap}({type(cond_v_unwrap)})\n",
                    stack_info=True,
                    stacklevel=2,
                )
        with self.with_frame(tir.While(cond_v_unwrap)):
            yield None

    def bind(self, name, value, annot=BaseBuilder.empty):
        self.check_continue_break()
        locals = self.get_parent_locals()
        orig_value = locals.get(name, None)
        # if orig_value is a local.var, we use buffer_store to modify it immutably
        #   however, if rvalue is not a PrimExpr, such as buffer,
        #   we should not use buffer_store, and bind it instead
        #   ```py
        #   a = tl.alloc_var('float32')  # bind var `a`
        #   a = tl.alloc_var('float32')  # bind a new var `a_1`
        #   a = tl.alloc_shared((1,), T.float32) # bind a to new buffer
        #   b = a                        # get value of var `b = a_1[0]``
        #   c = tl.alloc_var('float32')  # bind var `c`
        #   c = a                        # get and assign `c[0] = a_1[0]`
        #   ```
        if isinstance(orig_value, Ref) and isinstance(value, (int, float, PrimExpr)):
            orig_value.store(value)
            return orig_value
        if is_var(orig_value) and isinstance(value, (int, float, PrimExpr)):
            tir.buffer_store(orig_value, value, 0)
            return orig_value

        # 2. Quick return for trivil types
        if isinstance(value, (tuple, list, tvm.ffi.Array, int, float, str)):
            return value
        if isinstance(value, tir.IntImm) and value.dtype == "int32":
            return value.value
        if isinstance(value, (Var, Buffer)):
            # Bind TVM Var/Buffer names and also record scope so reusing the same
            # Python name (e.g., loop vars like `i`) across different for-frames
            # works without triggering out-of-scope errors.
            IRBuilder.name(name, value)
            if name != "_":
                frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
                assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
                self.name_inside_frame[name] = self.frames[frame]
            return value

        # 3. Bind immutable tilelang objects
        res = self.bind_immutable(name, value)

        # 4. Check variable scope and shadowing
        if name != "_":
            frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
            assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
            if name in self.name_inside_frame and self.name_inside_frame[name] in self.frames:
                logger.warning(
                    f"Variable `{name}` is declared twice, are you looking for a T.alloc_var?",
                    stack_info=True,
                    stacklevel=2,
                )
            self.name_inside_frame[name] = self.frames[frame]
        return res

    def unwrap_value(self, value):
        """
        Unwrap some tilelang objects to get their inner value
        """
        value = unwrap_expr(value)
        # handle bx, by = tl.Kernel(128, 128), rval is frame
        if isinstance(value, tir.frame.IRBuilderFrame):
            return self.enter_frame(value)
        else:
            return value

    def bind_immutable(self, name, value):
        """
        Bind an immutable tilelang objects.
        The immutability means the result is usually not changed or re-assigned in a python block.
        """
        if name == "_":
            # use _tmp to make the generated tir more readable
            name = "_tmp"
        if isinstance(value, tir.meta_var):
            return value.value
        elif isinstance(value, tir.frame.IRBuilderFrame):
            if isinstance(value, tir.frame.ForFrame):
                logger.warning(
                    "Binding a for frame to variable may cause undefined behavior in tilelang.",
                    stack_info=True,
                    stacklevel=2,
                )
            return self.enter_frame(value)
        elif isinstance(value, OutTensor):
            arg = tir.arg(
                name,
                tir.buffer(
                    shape=value.shape,
                    dtype=value.dtype,
                    strides=value.strides,
                ),
            )
            arg._out_idx = self.out_tensor_cnt
            self.out_tensor_cnt += 1
            return arg
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

    def assign_slice(self, lval: Any, sl: slice, value: Any, annot=BaseBuilder.empty):
        self.check_continue_break()
        if annot is not self.empty:
            logger.warning("Type annotation in slice assignment has no effect", stack_info=True, stacklevel=2)
        if isinstance(lval, Buffer):
            tir.buffer_store(lval, value, sl)
        else:
            return super().assign_slice(lval, sl, value)

    def aug_assign(self, op, target, aug_value):
        self.check_continue_break()
        if isinstance(target, Ref):
            target.store(eval_op(op, target.bufload, aug_value))
            return target
        elif is_var(target):
            tir.buffer_store(target, eval_op(op, target[0], aug_value), 0)
            return target
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

    def boolop(self, op, left, right=None):
        left = unwrap_cond(left)
        if isinstance(left, PrimExpr):
            with self.with_frame(BoolOpFrame()):
                if op == "And":
                    return tir.And(left, right())
                if op == "Or":
                    return tir.Or(left, right())
                if op == "Not":
                    return tir.Not(left)
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

    def ret(self, value=None):
        self.check_continue_break()
        # handle return T.alloc_var()
        if value is None:
            value = tuple()
        elif isinstance(value, tuple):
            value = tuple(self.unwrap_value(v) for v in value)
        else:
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
                    "@T.macro\n"
                    "def my_macro(cond):\n"
                    "    a = T.alloc_var(T.float16)\n"
                    "    if cond:\n"
                    "        a = 1.0\n"
                    "    return a\n"
                    "```"
                )
            return value
        else:
            if not isinstance(value, tuple):
                value = (value,)
            for v in value:
                if not isinstance(v, Buffer) or not hasattr(v, "_out_idx"):
                    raise RuntimeError(f"Only tensor allocated from `T.empty` can be returned in a prim_func, got {v}({type(v)})")
                # convert 0, 1, 2 => -3, -2, -1 as the out tensor index
                self.out_idx.append(v._out_idx - self.out_tensor_cnt)
            if len(self.out_idx) != self.out_tensor_cnt:
                raise RuntimeError(f"Not all tensor from `T.empty` are returned, only got {value}")
            return NotImplemented

    def ctx_with(self, ctx):
        self.check_continue_break()
        if isinstance(ctx, tir.frame.IRBuilderFrame):
            return self.with_frame(ctx)
        else:
            return super().ctx_with(ctx)

    def assert_expr(self, cond, msg=None):
        self.check_continue_break()
        cond = unwrap_cond(cond)
        if msg is None:
            msg = "Assertion failed"
        if isinstance(cond, PrimExpr):
            self.enter_frame(tir.Assert(cond, msg))
        elif not cond:
            raise AssertionError(msg)

    def rval(self, name: str, value: Any) -> Any:
        if name in self.name_inside_frame:
            frame = self.name_inside_frame[name]
            if frame not in self.frames:
                raise RuntimeError(
                    f"Use immutable variable `{name}` outside its defining region, did you forget **alloc_var**?\n"
                    f"variable `{name}` is defined in frame: {frame}, current frames: {self.frames}."
                )
        return self.unwrap_value(value)

    def macro_arg(self, name, value):
        annot_value = self.macro_arg_annot.get(name, None)
        if annot_value is Var or annot_value is Ref:
            if annot_value is Var:
                logger.warning("Use `T.Var` as macro annotations is deprecated, please use `T.Ref`")
            if isinstance(value, BufferLoad):
                if is_var(value.buffer):
                    return value.buffer
                idx = [self.bind("_", idx) for idx in value.indices]
                # indices = self.bind(f'_', value.indices)
                return Ref(BufferLoad(value.buffer, indices=idx))
            if isinstance(value, BufferRegion):
                region = [Range(self.bind("_", x.begin), end=self.bind("_", x.end) if x.end is not None else None) for x in value.region]
                return BufferRegion(value.buffer, region=region)
            raise ValueError(
                f"To pass as reference, argument `{name}` is expected to be a variable or a buffer region, but got {value}({type(value)})"
            )
        elif isinstance(value, (PrimExpr, int, float)):
            return self.bind(name, value)
        else:
            return value

    def prim_func_arg(self, name, value):
        return self.func_annot.create_argument(name, value, self.arg_vt)
        # if isinstance(value, (Buffer, Var)):
        #     return tir.arg(name, value)
        # elif value is self.empty:
        #     raise ValueError(f'Argument `{name}` is not annotated')
        # else:
        #     raise TypeError(
        #         f"Unsupported argument type: {value}({type(value)}) for argument `{name}`.")

    def arg(self, name, value):
        if self.find_frame_idx(MacroFrame) is not None:
            return self.macro_arg(name, value)
        else:
            return self.prim_func_arg(name, value)

    def override(self, name: str):
        from tilelang.language import serial

        if name == "range":
            return serial
        raise ValueError(f"Unknown override: {name}")


_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass
class PrimFuncCreater(Generic[_P, _T]):
    func_annot: FuncAnnot
    ir_gen: IRGenerator[_P, _T]
    orig_func: Callable[_P, _T]

    @property
    def annot(self) -> dict[str, Annot]:
        return self.func_annot.annots

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> PrimFunc[_P, _T]:
        builder = Builder(self.func_annot)
        with builder.prim_func(self.orig_func.__name__):
            self.ir_gen.gen(builder)(*args, **kwargs)
        res: PrimFunc = builder.get()
        res.ir_gen = self.ir_gen
        res.orig_func = self.orig_func
        res.func_annot = self.func_annot
        res.out_idx_override = builder.out_idx or None
        return res

    def __repr__(self):
        fmt = pprint.pformat({"annot": self.func_annot.annots, "ir_gen": self.ir_gen, "orig_func": self.orig_func}, indent=2)
        return f"{self.__class__.__name__}(\n{fmt}\n)"


if TYPE_CHECKING:

    class PrimFunc(Generic[_P, _T], tvm.tir.PrimFunc):
        params: list[tvm.tir.Var | tvm.tir.Buffer]
        body: tvm.tir.Stmt
        ret_type: tvm.ir.Type
        buffer_map: Map[tvm.tir.Var, tvm.tir.Buffer]
        attrs: tvm.Attrs | None
        span: Span | None
        ir_gen: IRGenerator[_P, _T] | None
        orig_func: Callable[_P, _T] | None
        func_annot: FuncAnnot | None
        out_idx_override: list[int] | None

else:
    PrimFunc = tvm.tir.PrimFunc


@dataclass
class Macro(Generic[_P, _T]):
    name: str
    orig_func: Callable[_P, _T]
    ir_gen: IRGenerator[_P, _T]
    annotations: dict[str, Any]

    @property
    def source(self) -> str:
        return self.ir_gen.source

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        builder = Builder.current() or Builder()
        with builder.macro(self.name, self.annotations):
            res = self.ir_gen.gen(builder)(*args, **kwargs)
        return res

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)


def macro(func: Callable[_P, _T] = None) -> Macro[_P, _T]:
    """
    Decorator that converts a Python function into a TileLang macro.
    TileLang macro is very similar to PrimFunc, it can be used in prim_func or another macro.
    Parameters
    ----------
    func : Callable[_P, _T]
        The Python function to be converted into a macro. This function will be analyzed
        and transformed into an IR generation function. The function can take any parameters
        (_P) and return any type (_T).
    Returns
    -------
    Macro[_P, _T]
        A Macro object that wraps the original function with IR generation capabilities.
        The returned Macro preserves the original function's signature (parameters _P and
        return type _T) while adding metaprogramming capabilities.
    Example:
    --------
        >>> @macro
        ... def my_macro(x: T.int32) -> T.int32:
        ...    return x ** 2
        >>> @prim_func
        ... def my_func(A: T.Tensor((10,), T.int32), B: T.Tensor((10,), T.int32)):
        ...    with T.Kernel(1) as _:
        ...        for i in T.serial(10):
        ...            B[i] = my_macro(A[i])
    See Also
    --------
    Macro : The class that wraps macro functions
    mutate : The function that transforms Python code into IR generators
    """

    def impl(func: Callable[_P, _T]) -> Macro[_P, _T]:
        annotations = get_type_hints(func)
        return Macro(name=func.__name__, orig_func=func, ir_gen=mutate(func), annotations=annotations)

    return impl(func) if func is not None else impl


from typing import _eval_type


def get_type_hints(func):
    annot = getattr(func, "__annotations__", None)
    if annot is None:
        raise TypeError(f"Failed to get function type hints, {func} is not a function")
    hints = {}
    # Build eval namespaces from function globals plus captured closure variables
    # This lets annotations reference symbols like `n`, `h`, or dtype vars
    # defined in the outer scope of a nested function.
    globalns = func.__globals__
    # Here we add nonlocals into localns, to capture the parameters declared in the parent function
    # ```py
    # def foo():
    #   n = 128 # n is nonlocal
    #   def bar(
    #       A: T.Tensor(n, T.float32) # we add nonlocal in its eval context
    #   ):
    #      for i in range(n): ...
    # ```
    #
    # This is incomplete and buggy
    #   the only bug scenario the function body doesn't use the the parameters
    #   but such define-no-use scenario is very rare in writing kernels
    #
    # ```py
    # def foo():
    #   n = 128
    #   def bar(A: T.Tensor((n,), T.float32)):
    #     ... # empty function, do not use `n`
    localns = utils.get_func_nonlocals(func)
    for name, value in annot.items():
        if name == "return":
            continue
        if isinstance(value, tvm.DataType):
            hints[name] = value
            continue
        if value is None:
            value = type(None)
        if isinstance(value, str):
            # if the annotation is string, is can be: (i) a T.float32 like annotations, (ii) a ForwardRef object
            # typing doesn't handle (i), it will try to interpret T.float32
            #    typing see: T.float32 is str('float32'), and there is no object named `flaot32` and give a NameError
            # here we manually interpret it to return T.float32 object
            try:
                _, v = value.split(".", maxsplit=1)
            except ValueError:
                v = value
            if v in dt._all_dtypes:
                try:
                    hints[name] = eval(value, globalns, localns)
                    continue
                except Exception:
                    pass
            value = ForwardRef(value, is_argument=True, is_class=False)
            hints[name] = _eval_type(value, globalns=globalns, localns=localns)
        else:
            hints[name] = value
    return hints


def prim_func(func: Callable[_P, _T] = None, *, generator: bool = False) -> PrimFunc[_P, _T] | PrimFuncCreater[_P, _T]:
    """
    Decorator to create a primitive function (PrimFunc) for TileLang IR generation.
    This decorator transforms a Python function into a TileLang primitive function by analyzing
    its type annotations and generating intermediate representation (IR) code. It supports both
    immediate construction (when all parameters are statically annotated) and generator mode
    (for dynamic construction).
    Parameters
    ----------
    func : Callable[_P, _T], optional
        The function to be decorated. Can be None when using decorator with arguments.
    generator : bool, default=False
        If True, returns a generator function that creates PrimFunc instances on demand.
        If False, attempts to create a PrimFunc immediately using type annotations.
    Returns
    -------
    PrimFunc[_P, _T] | Callable[_P, PrimFunc[_P, _T]]
        - If `generator=False` and all parameters are statically annotated: returns a PrimFunc instance
        - If `generator=True`: returns a callable that generates PrimFunc instances when invoked
        - If used without parentheses: returns the decorator implementation function
    Examples
    --------
    Static annotation mode (immediate construction):
    >>> @prim_func
    ... def add_kernel(A: T.Buffer((128,), T.float32),
    ...                B: T.Buffer((128,), T.float32)):
    ...     for i in T.grid(128):
    ...         B[i] = A[i] + 1.0
    Generator mode (dynamic construction):
    >>> @prim_func(generator=True)
    ... def dynamic_kernel(A=T.Tensor((128,), T.float32)):
    ...     # function body
    ...     pass
    >>> kernel_instance = dynamic_kernel()
    With custom parameters:
    >>> @prim_func(generator=True)
    ... def parameterized_kernel(size: int = 128):
    ...     # function body using size parameter
    ...     pass
    >>> kernel = parameterized_kernel(size=256)
    See Also
    --------
    Builder : The IR builder class used for constructing primitive functions
    mutate : Function used to generate IR from the decorated function
    """

    def impl(func: Callable[_P, _T]) -> PrimFunc[_P, _T] | Callable[_P, PrimFunc[_P, _T]]:
        sig = inspect.signature(func)
        annot = get_type_hints(func)

        func_annot = FuncAnnot.from_sig_annots(sig, annot)
        ir_gen = mutate(func)

        prim_func_generator = PrimFuncCreater(func_annot, ir_gen, orig_func=func)

        if func_annot.is_all_static():
            args = func_annot.get_all_static_args()
            return prim_func_generator(**args)
        else:
            if generator is False:
                unknown_args = func_annot.get_compile_time_unknown_args()
                raise ValueError(
                    f"Cannot create PrimFunc for `{func.__name__}`, some arguments are not compile-time known, \n"
                    f"Annotations:\n{func_annot.annots}"
                    f"Unknown Args: {unknown_args}"
                )
            return prim_func_generator

    return impl(func) if func is not None else impl
