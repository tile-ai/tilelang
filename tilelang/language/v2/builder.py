from __future__ import annotations

from contextlib import contextmanager, AbstractContextManager
from dataclasses import dataclass
import inspect

from tilelang.language.kernel import KernelLaunchFrame
from tvm_ffi.container import Map
from tvm.ir.base import Span
from .ast import BaseBuilder, IRGenerator, eval_op, mutate
import tvm
from tvm.tir import Buffer
from tvm.script.ir_builder import tir, IRBuilder
from tvm.tir.expr import EqualOp, FloatImm, IntImm, NotEqualOp, PrimExpr, StringImm, Var
from typing import TYPE_CHECKING, Callable, Any, Generic, TypeVar, ForwardRef, Union
# Python 3.9 compatibility for ParamSpec and Self
try:
    from typing import ParamSpec, Self
except ImportError:  # Python < 3.11 for Self, < 3.10 for ParamSpec
    from typing_extensions import ParamSpec, Self
from . import dtypes as dt
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
            "this is treated as a constant expression. ",
            stack_info=True,
            stacklevel=3)
        return bool(expr)


thread_local_storage = threading.local()


class Frame:
    '''
    Frame are virtual context managers used in frontend only
    They do not have any runtime representation in the generated TIR.
    '''

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


@dataclass
class SerialForWithStep:
    start: PrimExpr
    stop: PrimExpr
    step: PrimExpr
    annotations: dict[str, Any] | None = None


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
    return isinstance(v, Buffer) and v.scope() == 'local.var'


class Builder(BaseBuilder):

    def __init__(self):
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

    def enter_frame(self, frame: AbstractContextManager[Any]):
        self.frames.append(frame)
        return frame.__enter__()

    def check_continue_break(self):
        idx = self.find_frame_idx(ContinueOrBreak)
        if idx is not None:
            logger.warning(
                'Writing code after continue/break may cause undefined behavior in tilelang.',
                stack_info=True,
                stacklevel=3)

    @contextmanager
    def with_frame(self, frame: AbstractContextManager[Any] | None):
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
            if isinstance(val, tir.frame.ForFrame):
                logger.warning(
                    'Evaluating a for frame may cause undefined behavior in tilelang.',
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
        elif not isinstance(val, tvm.tir.Buffer):
            raise TypeError(f"Unsupported eval value: {val} of type {type(val)}")

    def ctx_for(self, it):
        self.check_continue_break()
        it = unwrap_expr(it)
        if isinstance(it, SerialForWithStep):
            # Validate and compute the trip count before constructing the frame
            if isinstance(it.step, (int, IntImm)):
                step_value = it.step if isinstance(it.step, int) else it.step.value
                if step_value == 0:
                    raise ValueError('Invalid stepped serial: step must be non-zero')
                if step_value > 0:
                    real_stop = tir.ceildiv(it.stop - it.start, step_value)
                else:
                    real_stop = tir.ceildiv(it.start - it.stop, -step_value)
            else:
                logger.warning(
                    f'Using a non-constant step `{it.step}` in stepped serial may lead to undefined behavior in tilelang'
                )
                real_stop = tir.ceildiv(it.stop - it.start, it.step)
            real_frame = tir.serial(real_stop, annotations=it.annotations)
            with self.with_frame(real_frame) as v:
                IRBuilder.name('_tmp', v)
                yield it.start + v * it.step
        else:
            if not isinstance(it, tir.frame.ForFrame):
                raise TypeError(
                    f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                    "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding")
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
        raise RuntimeError("while loops are not supported in TileLang builder")

    def bind(self, name, value, annot=BaseBuilder.empty):
        self.check_continue_break()
        locals = self.get_parent_locals()
        orig_value = locals.get(name, None)
        # annotation like tl.float32
        # temporarily disable annotation based var declaration, for better pull request separation
        # if callable(annot):
        #     annot_val = annot()
        #     if isinstance(annot_val, tir.Var):
        #         orig_value = tir.alloc_buffer((1,), dtype=annot_val.dtype, scope='local.var')
        #         IRBuilder.name(name, orig_value)
        #         if isinstance(value, EllipsisType) or value is self.empty:
        #             return orig_value
        #         elif isinstance(value, (int, float, IntImm, FloatImm)):
        #             tir.block_attr(
        #                 {'tl.local_var_init': {
        #                     orig_value.data: tvm.runtime.convert(value)
        #                 }})
        #             return orig_value
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
            if name in self.name_inside_frame and self.name_inside_frame[name] in self.frames:
                logger.warning(
                    f'Variable `{name}` shadows another declared value, Are you forgetting to allocate it as a var?',
                    stack_info=True,
                    stacklevel=2,
                )
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
        if name == '_':
            # use _tmp to make the generated tir more readable
            name = "_tmp"
        if isinstance(value, tir.meta_var):
            return value.value
        elif isinstance(value, tir.frame.IRBuilderFrame):
            if isinstance(value, tir.frame.ForFrame):
                logger.warning(
                    'Binding a for frame to variable may cause undefined behavior in tilelang.',
                    stack_info=True,
                    stacklevel=2,
                )
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
                    "    a = T.alloc_var(T.float16)\n"
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

    def arg(self, name, value):
        if self.find_frame_idx(MacroFrame) is not None:
            if isinstance(value, (PrimExpr, int, float)):
                return self.bind(name, value)
            else:
                return value
        if isinstance(value, (Buffer, Var)):
            return tir.arg(name, value)
        elif value is self.empty:
            raise ValueError(f'Argument `{name}` is not annotated')
        # elif isinstance(value, Hashable):
        #     return value
        else:
            raise TypeError(
                f"Unsupported argument type: {value}({type(value)}) for argument `{name}`.")

    def override(self, name: str):
        from tilelang.language import serial
        if name == 'range':
            return serial
        raise ValueError(f'Unknown override: {name}')


_P = ParamSpec('_P')
_T = TypeVar('_T')

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
        return Macro(name=func.__name__, orig_func=func, ir_gen=mutate(func))

    return impl(func) if func is not None else impl


from typing import _eval_type


def get_type_hints(func):
    annot = getattr(func, '__annotations__', None)
    if annot is None:
        raise TypeError(f'Failed to get function type hints, {func} is not a function')
    hints = {}
    # Build eval namespaces from function globals plus captured closure variables
    # This lets annotations reference symbols like `n`, `h`, or dtype vars
    # defined in the outer scope of a nested function.
    globalns = dict(getattr(func, '__globals__', {}))
    localns = dict(globalns)
    try:
        freevars = getattr(func.__code__, 'co_freevars', ())
        cells = getattr(func, '__closure__', ()) or ()
        closure_bindings = {
            name: cell.cell_contents for name, cell in zip(freevars, cells) if name not in localns
        }
        if closure_bindings:
            localns.update(closure_bindings)
            # Also update globals so ForwardRef eval sees them uniformly
            globalns.update(closure_bindings)
    except Exception:
        # Be permissive: absence or access issues with closure shouldn't crash
        pass

    for name, value in annot.items():
        if name == 'return':
            continue
        if isinstance(value, tvm.DataType):
            hints[name] = value
            continue
        if value is None:
            value = type(None)
        if isinstance(value, str):
            # Handle simple dtype aliases like T.float32 appearing as strings
            # Evaluate directly only when it matches known dtypes
            try:
                _, v = value.split('.', maxsplit=1)
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
    return hints


def _is_static_annot(annot: Any) -> bool:
    return isinstance(annot, (dt.dtype, Buffer, Var))


def prim_func(func: Callable[_P, _T] = None,
              *,
              generator: bool = False) -> PrimFunc[_P, _T] | Callable[_P, PrimFunc[_P, _T]]:
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

        for k in annot:
            if callable(annot[k]):
                annot[k] = annot[k]()

        # check whether all arguments are annotated
        all_arg_annotated = all([x in annot for x in sig.parameters])
        # check whether all annotations are Buffer/Var/dtype
        all_annot_are_static = all([_is_static_annot(x) for x in annot.values()])
        ir_gen = mutate(func)

        def prim_func_generator(*args, **kwargs):
            builder = Builder()
            with builder.prim_func(func.__name__):
                ir_gen.gen(builder)(*args, **kwargs)
            res = builder.get()
            res.ir_gen = ir_gen
            res.source = ir_gen.source
            res.orig_func = func
            return res

        prim_func_generator.ir_gen = ir_gen
        prim_func_generator.source = ir_gen.source
        prim_func_generator.orig_func = func

        if generator:
            return prim_func_generator

        if all_arg_annotated and all_annot_are_static:
            return prim_func_generator(**annot)
        else:
            raise ValueError(
                "Some arguments are not supported or statically annotated, \n"
                "please check the annotations or set generator=True to get a prim_func generator.\n"
                f"Argument Annotations: {annot}\n"
                "Example usage of generator:\n"
                "```py\n"
                "@prim_func(generator=True)\n"
                "def my_func(a=T.Tensor((128,), T.float32)): ...\n"
                "return my_func()\n"
                "```")

    return impl(func) if func is not None else impl
