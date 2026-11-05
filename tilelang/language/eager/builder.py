from __future__ import annotations
from tilelang.language import _native
from contextlib import contextmanager, AbstractContextManager
from dataclasses import dataclass
import inspect
import builtins
from functools import partial

from tilelang._analyzer import create_analyzer
from tilelang import env
from tilelang.language.kernel import KernelLaunchFrame
from tvm_tirx.ir import Location as Span
from tvm_tirx.ir import Range
from tilelang.language._native import BufferRegion
from .ast import BaseBuilder, IRGenerator, eval_op, has_internal_prim_func, mutate
from .utils import construct_strides
from tilelang.language.frame import clear_let_values, register_let_value
from tilelang.language._native import side_effect
import tvm_tirx as tvm
import tvm_ffi
from tilelang.language._native import Buffer, tensor_strides, dtype_of
from tilelang.language import _native as tirx
from tvm_tirx.script.ir_builder import IRBuilder

from tilelang.language._native import BufferLoad, CallEffectKind, EqualOp, FloatImm, IntImm, NotEqualOp, PrimExpr, StringImm, Var
from typing import TYPE_CHECKING, Any, Generic, TypeVar, ForwardRef, Literal, get_origin, ParamSpec
from collections.abc import Callable
from typing_extensions import Self
from collections.abc import Hashable
from collections.abc import Sequence

from .. import dtypes as dt
from . import utils
from tilelang.jit.exceptions import JITNoBuilderError, EagerJITBuildError
import threading
import logging

logger = logging.getLogger(__name__)


def _is_runtime_parameter_annotation(value):
    """Recognize source and native parameter annotations before meta values."""
    if isinstance(value, tvm.ir.Type):
        if not isinstance(value, (tvm.ir.PrimType, tvm.ir.PtrType, tvm.tirx.TensorType)):
            raise TypeError(f"A native TileLang runtime parameter requires PrimType, PtrType or TensorType, got {type(value).__name__}")
        return True
    return isinstance(value, (Buffer, Var))


# Guard iterator/container construction, not reductions (use explicit T.* ops
# for device reductions). Identity keys avoid hashing or comparing user callables.
_BUILTIN_ITERABLE_ARG_MAP = {
    id(func): (func, select_iterables)
    for funcs, select_iterables in (
        ((builtins.enumerate,), lambda args, kwargs: args[:1] or ((kwargs["iterable"],) if "iterable" in kwargs else ())),
        ((builtins.zip,), lambda args, kwargs: args),
        ((builtins.map,), lambda args, kwargs: args[1:]),
        ((builtins.filter,), lambda args, kwargs: args[1:2]),
        # iter(callable, sentinel) does not take an iterable.
        ((builtins.iter,), lambda args, kwargs: args if len(args) == 1 else ()),
        (
            (
                builtins.reversed,
                builtins.list,
                builtins.tuple,
                builtins.set,
                builtins.frozenset,
                builtins.dict,
                builtins.sorted,
            ),
            lambda args, kwargs: args[:1],
        ),
    )
    for func in funcs
}


def unwrap_expr(expr) -> PrimExpr | int | float:
    """
    unwrap expr and convert it into PrimExpr like
    """
    if isinstance(expr, tirx.meta_var):
        expr = expr.value
    elif isinstance(expr, Ref):
        return expr.load()
    elif is_var(expr):
        expr = tirx.BufferLoad(expr, indices=[0])
    elif isinstance(expr, (EqualOp, NotEqualOp)):
        expr = expr.__tvm_ffi_object__()
    return expr


def _dtype_total_bits(dtype: dt.dtype) -> int:
    return dtype.bits * dtype.lanes


def _packed_storage_factor(buffer: Buffer, value) -> int:
    if isinstance(value, Buffer):
        return 1

    logical_bits = _dtype_total_bits(dtype_of(buffer))
    if logical_bits >= 8:
        return 1

    storage_dtype = dt.dtype(dtype_of(value))
    return _dtype_total_bits(storage_dtype) // logical_bits


def _logical_shape_value(value, buffer: Buffer, dim: int):
    shape = value.shape[dim]
    if dim == len(buffer.shape) - 1:
        shape *= _packed_storage_factor(buffer, value)
    return shape


def _logical_stride_value(value, buffer: Buffer, dim: int):
    if isinstance(value, Buffer):
        return tensor_strides(value)[dim]

    stride = value.stride()[dim]
    if dim != len(tensor_strides(buffer)) - 1:
        stride *= _packed_storage_factor(buffer, value)
    return stride


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
            f"Python expression `{expr}` is used as condition in TileLang, this is treated as a constant expression.",
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


class PythonLoopFrame(Frame): ...


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
        return self.bufload.source

    def store(self, value):
        tirx.buffer_store(self.bufload.source, value, self.bufload.indices)

    def load(self):
        return self.bufload


@dataclass
class PointerRef(Ref):
    """A DSL mutable pointer backed by an actual uint64 scalar tensor.

    Native tensors contain primitive values, so a pointer cannot be a tensor
    element. The existing native reinterpret operation converts a pointer to
    its 64-bit address for storage and restores its pointee type and address
    space for each read. This wrapper only participates in source construction;
    the resulting IR contains TensorLoad/Store and typed reinterpret calls.
    """

    pointer_type: tvm.ir.PtrType
    ir_builder: IRBuilder

    @property
    def dtype(self):
        return tvm_ffi.dtype("handle")

    def store(self, value):
        value = unwrap_expr(value)
        if isinstance(value, int) and not isinstance(value, bool):
            address = _native.const(value, "uint64")
        elif isinstance(value, tvm.ir.Expr) and isinstance(value.ty, tvm.ir.PtrType):
            if value.ty.scope != self.pointer_type.scope:
                raise TypeError("A mutable handle assignment must preserve its pointer address space")
            address = tvm.tirx.reinterpret(value, ty=tvm.ir.PrimType("uint64"), loc=_native._loc())
        else:
            raise TypeError("A mutable handle requires a pointer expression or an integer address")
        tirx.buffer_store(self.buffer, address, self.bufload.indices)

    def load(self):
        value = tvm.tirx.reinterpret(self.bufload, ty=self.pointer_type, loc=_native._loc())
        # Construction-only provenance. Compiler consumers use the actual
        # PtrType on value and never rely on this Python attribute.
        value._tilelang_pointer_ref = self
        return value

    @staticmethod
    def _check_index(index):
        if isinstance(index, IntImm):
            index = int(index.value)
        if not isinstance(index, int) or isinstance(index, bool) or index != 0:
            raise IndexError("A mutable handle has one element, at index 0")

    def __getitem__(self, index):
        self._check_index(index)
        return self.load()

    def __setitem__(self, index, value):
        self._check_index(index)
        return self.store(value)


class UnrollForWithStep(SerialForWithStep): ...


# Runtime frame checks use tuple form; AnyFrame is kept as an annotation alias.
ContinueOrBreak = (ContinueFrame, BreakFrame)
AnyFrame = tirx.frame.IRBuilderFrame | Frame

TIR_CONTROL_FRAME = (
    tirx.frame.WhileFrame,
    tirx.frame.ForFrame,
    tirx.LoopFrame,
    tirx.frame.IfFrame,
    tirx.frame.PrimFuncFrame,
)

TIR_VAR_SCOPE_FRAME = (
    tirx.frame.WhileFrame,
    tirx.frame.ForFrame,
    tirx.LoopFrame,
    tirx.frame.IfFrame,
    tirx.frame.PrimFuncFrame,
    MacroFrame,
    KernelLaunchFrame,
    tirx.BlockFrame,
    tirx.InitFrame,
)


def register_var_scope_frame(frame_type: type) -> None:
    """Register a backend-specific frame as a variable scope."""
    global TIR_VAR_SCOPE_FRAME
    if frame_type not in TIR_VAR_SCOPE_FRAME:
        TIR_VAR_SCOPE_FRAME += (frame_type,)


def is_var(v: Any) -> bool:
    return isinstance(v, Buffer) and str(v.scope()) == "local.var"


# phase1: eager jit obtain function signature
# phase2: eager jit elaborate function
# none: not inside eager jit, i.e. it is lazyjit
EagerJITStage = Literal["phase1", "phase2", "none"]


class Builder(BaseBuilder):
    def __init__(self):
        self.frames: list[AnyFrame] = []
        self.ir_builder = IRBuilder()
        self.name_inside_frame: dict[str, AnyFrame] = {}
        self.macro_arg_annot = {}
        self.out_idx = []
        self.out_tensor_cnt = 0
        self.constexpr_var = set()
        self.eager_jit: EagerJITStage = "none"
        self.eager_jit_subs: dict[str, PrimExpr] = {}
        self.func_pass_configs: dict[str, Any] | None = None
        self.func_compile_flags: list[str] | str | None = None
        self.current_file = "<unknown>"
        self.current_line = 0
        self.current_macro_name = "<unknown-macro>"
        # stack to record caller fileline, not callee fileline
        self.macro_fileline_stack: list[tuple[str, int, str]] = []
        self._spans_enabled = env.is_span_enable()
        # Pending leaf segment: (stmt_frame, stmt_count, file, line) recorded
        # by set_fileline; stmts appended to the frame afterwards belong to
        # that source statement until the next set_fileline / frame exit.
        self._span_pending: tuple[Any, int, str, int] | None = None
        self._span_first_fileline: tuple[str, int] | None = None
        # Body annotations can appear in a different order than the Python
        # signature. Register their actual native types when known, then retain
        # the original argument order when assembling the completed Function.
        self._parameter_order: dict[str, Var | None] = {}
        self._deferred_parameters: dict[str, Var] = {}
        self._matched_parameters: dict[Var, Var] = {}
        self._matched_parameter_storage: dict[Var, Var] = {}
        self._matched_parameter_specs: dict[Var, _native.TensorSpec] = {}
        self._tensor_parameter_specs: dict[Var, dict[str, Any]] = {}

    @classmethod
    def current(cls) -> Self:
        builder = getattr(thread_local_storage, "builder", None)
        return builder

    @contextmanager
    def prim_func(self, name, *, private=False):
        thread_local_storage.builder = self
        clear_let_values()
        try:
            with self.ir_builder, self.with_frame(tirx.prim_func(is_private=private)):
                tirx.func_name(name)
                yield
            if self.eager_jit != "phase1" and len(self.out_idx) != self.out_tensor_cnt:
                raise RuntimeError("Not all tensor allocated from `T.empty` are returned")
        finally:
            clear_let_values()
            del thread_local_storage.builder

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
        self.macro_fileline_stack.append((self.current_file, self.current_line, self.current_macro_name))
        self.frames.append(MacroFrame())
        yield
        self.frames[pos] = ExitedMacroFrame()
        self.macro_fileline_stack.pop()
        self.name_inside_frame, self.macro_arg_annot = save

    @property
    def current_location(self):
        if not self._spans_enabled or self.current_line <= 0:
            return tvm.ir.UnknownLoc()
        loc = tvm.ir.SourceLoc(tvm.ir.SourceName(self.current_file), self.current_line, 0, self.current_line, 0)
        for file, line, _ in self.macro_fileline_stack:
            loc = tvm.ir.CallSiteLoc(loc, tvm.ir.SourceLoc(tvm.ir.SourceName(file), line, 0, line, 0))
        return loc

    def get(self) -> PrimFunc:
        if self._spans_enabled:
            self._flush_span_pending()
        func = self.ir_builder.get()
        if self._deferred_parameters or self._matched_parameters:
            missing = [name for name in self._deferred_parameters if self._parameter_order[name] is None]
            if missing:
                raise ValueError(f"Missing tensor/pointer annotation for parameter(s): {', '.join(missing)}")
            ordered = list(self._parameter_order.values())
            seen = set(ordered) | set(self._matched_parameters) | set(self._matched_parameter_storage)
            ordered.extend(param for param in func.params if param not in seen)
            func = _native.Function(ordered, func.body, func.ret_type, func.attrs, func.loc)
        if self._matched_parameters:
            func = self._finalize_matched_parameters(func)
        func = _native.attach_parameter_tensor_specs(func, self._tensor_parameter_specs)
        func = _native.attach_size_var_defaults(func)
        if self._spans_enabled and self._span_first_fileline is not None:
            file, line = self._span_first_fileline
            self.ir_builder._attach_loc(func, tvm.ir.SourceLoc(tvm.ir.SourceName(file), line, 0, line, 0))
        return func

    def match_parameter_buffer(self, value, spec):
        """Materialize the tensor contract of a source handle parameter.

        Shape symbols introduced in the body belong to the matched parameter
        type. They are bound from the runtime tensor by the native ABI. The
        temporary handle is dropped when the final parameter order is rebuilt;
        no extra public argument or detached shape-variable table is created.
        """
        if not isinstance(value, Var) or not isinstance(value.ty, tvm.ir.PtrType):
            return None
        if value in self._matched_parameters:
            previous = self._matched_parameters[value]
            name = next(name for name, parameter in self._parameter_order.items() if parameter is not None and parameter.same_as(previous))
            self._tensor_parameter_specs.pop(previous, None)
        else:
            name = next(
                (name for name, parameter in self._parameter_order.items() if parameter is not None and parameter.same_as(value)), None
            )
        if name is None:
            return None
        parameter = tirx.arg(name, spec)
        # The last match supplies the input ABI, as in the original builder.
        # Equal types alone never identify storage. Only explicit data= links
        # earlier views to the last match's physical pointer.
        storage = parameter if spec.data is None else _native.tensor_storage_source(spec.data)
        if not isinstance(storage, Var):
            raise TypeError("match_buffer data must be a pointer variable or a tensor data projection")
        storage = self._matched_parameter_storage.get(storage, storage)
        self._matched_parameter_storage[parameter] = storage
        self._matched_parameter_specs[parameter] = spec
        self._parameter_order[name] = parameter
        self._matched_parameters[value] = parameter
        return parameter

    def _finalize_matched_parameters(self, func):
        """Materialize descriptor and explicit data aliases in native IR.

        Temporary construction records do not survive the Function. Its last
        tensor parameters, entry Bind statements and pointer-equality checks
        carry the entire runtime contract through serialization and lowering.
        """
        ir = tvm.ir
        native = tvm.tirx
        used = set()
        for root in (func.body, func.attrs):
            if root is not None:
                tvm_ffi.structural_walk(root, (Var, used.add))

        prefix = []
        representatives = {}
        for parameter in self._matched_parameters.values():
            storage = self._matched_parameter_storage[parameter]
            if storage in representatives:
                other = representatives[storage]
                condition = native.reinterpret(native.tensor_data_ptr(parameter), ty="uint64") == native.reinterpret(
                    native.tensor_data_ptr(other), ty="uint64"
                )
                prefix.append(
                    ir.AssertStmt(
                        _native._expr(condition),
                        ir.StringImm("RuntimeError"),
                        [ir.StringImm("match_buffer parameters sharing data must have the same physical pointer")],
                        func.loc,
                    )
                )
            else:
                representatives[storage] = parameter

        local_specs = {}
        public_parameters = set(func.params)
        for tensor, storage in self._matched_parameter_storage.items():
            if tensor in public_parameters or tensor not in used:
                continue
            if storage not in representatives:
                raise ValueError(
                    f"Earlier match_buffer tensor `{tensor.name}` has no storage definition after a later match; "
                    "use explicit data= to share its storage"
                )
            source = native.tensor_data_ptr(representatives[storage], loc=tensor.loc)
            declaration = ir.Call("tirx.decl_tensor", [source], ty_args=[tensor.ty], loc=tensor.loc)
            prefix.append(ir.Bind(tensor, declaration, tensor.loc))
            metadata = self._matched_parameter_specs[tensor].contract_metadata()
            if metadata is not None:
                local_specs[tensor] = metadata

        for storage, parameter in representatives.items():
            if storage in self._matched_parameter_storage:
                continue
            source = native.tensor_data_ptr(parameter, loc=storage.loc)
            if not isinstance(storage.ty, ir.PtrType):
                raise ValueError("match_buffer data must originate in an input or a pointer variable")
            if storage in public_parameters:
                condition = native.reinterpret(storage, ty="uint64") == native.reinterpret(source, ty="uint64")
                prefix.append(
                    ir.AssertStmt(
                        _native._expr(condition),
                        ir.StringImm("RuntimeError"),
                        [ir.StringImm("match_buffer data must equal its explicit pointer parameter")],
                        storage.loc,
                    )
                )
            elif storage in used:
                # An explicit free data Var is defined from the final input;
                # it is neither an extra argument nor a DLTensor descriptor.
                prefix.append(ir.Bind(storage, native.reinterpret(source, ty=storage.ty, loc=storage.loc), storage.loc))

        for handle, parameter in self._matched_parameters.items():
            if handle in used:
                descriptor = _native.tensor_descriptor_ptr(parameter, loc=handle.loc)
                if not tvm_ffi.structural_equal(handle.ty, descriptor.ty):
                    descriptor = native.reinterpret(descriptor, ty=handle.ty, loc=handle.loc)
                prefix.append(ir.Bind(handle, descriptor, handle.loc))
        if local_specs:
            func = _native.attach_parameter_tensor_specs(func, local_specs)
        if prefix:
            func = func.with_body(ir.SeqStmt([*prefix, func.body], loc=func.loc), loc=func.loc)
        return func

    # ------------------------------------------------------------------
    # Source span injection (TILELANG_ENABLE_IR_SPAN)
    # ------------------------------------------------------------------

    def with_buffer_span(self, buffer: Buffer) -> Buffer:
        """Stamp an unspanned buffer with the current user source location."""
        if not self._spans_enabled or self.current_line <= 0:
            return buffer

        if isinstance(buffer.loc, tvm.ir.UnknownLoc):
            self.ir_builder._attach_loc(buffer, self.current_location)
        return buffer

    def _current_stmt_frame(self):
        """The innermost open frame whose `stmts` accumulates emitted stmts."""
        for frame in reversed(self.frames):
            if isinstance(frame, (tirx.frame.TIRFrame, tirx.LoopFrame, tirx.BlockFrame, tirx.InitFrame, KernelLaunchFrame)):
                f = frame
                # KernelLaunchFrame nests launch/block frames entered C++-side;
                # leaf statements accumulate in its innermost sub-frame.
                while isinstance(f, KernelLaunchFrame) and len(f.frames) > 0 and isinstance(f.frames[-1], tirx.frame.TIRFrame):
                    f = f.frames[-1]
                return f
        return None

    def _flush_span_pending(self):
        """Stamp stmts appended since the last recorded source statement."""
        if self._span_pending is None:
            return

        frame, start, file, line = self._span_pending
        self._span_pending = None
        if line <= 0:
            return
        try:
            stmts = frame.stmts
        except Exception:
            return
        span = tvm.ir.SourceLoc(tvm.ir.SourceName(file), line, 0, line, 0)
        for i in range(start, len(stmts)):
            if isinstance(stmts[i].loc, tvm.ir.UnknownLoc):
                self.ir_builder._attach_loc(stmts[i], span)

    def _stamp_new_parent_stmts(self, parent, start: int, file: str, line: int):
        """Stamp stmts produced by an exited frame into its parent frame.

        Frames like KernelLaunchFrame assemble a whole subtree (launch grid
        loops, block realize) whose intermediate nodes never pass through
        `with_frame`; stamp every still-unspanned stmt in the produced
        subtree with the frame's entry line. Nodes already stamped with their
        own statement line keep their span.
        """
        if parent is None or line <= 0:
            return

        try:
            stmts = parent.stmts
        except Exception:
            return
        if len(stmts) <= start:
            return
        span = tvm.ir.SourceLoc(tvm.ir.SourceName(file), line, 0, line, 0)

        def stamp(node):
            if isinstance(node, tvm.ir.Stmt) and isinstance(node.loc, tvm.ir.UnknownLoc):
                self.ir_builder._attach_loc(node, span)

        for i in range(start, len(stmts)):
            tvm_ffi.structural_walk(stmts[i], (tvm.ir.Stmt, stamp), order="post")

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
            logger.warning("Statements after continue/break have no effect and will be ignored.", stacklevel=3)

    @contextmanager
    def with_frame(self, frame: AbstractContextManager[Any] | None):
        pop_idx = len(self.frames)
        span_entry = None
        if self._spans_enabled and frame is not None:
            # Record the enclosing stmt container and the entry position (the
            # `with`/`for` line), so stmts this frame produces into its parent
            # can be stamped after it exits.
            parent = self._current_stmt_frame()
            span_entry = (
                parent,
                len(parent.stmts) if parent is not None else 0,
                self.current_file,
                self.current_line,
            )
        try:
            yield self.enter_frame(frame)
        except BaseException as exc:
            # Unwind Python frame state without finalizing incomplete TIR.
            while len(self.frames) > pop_idx:
                self.frames.pop().__exit__(type(exc), exc, exc.__traceback__)
            raise
        if self._spans_enabled:
            # Flush leaf stmts of the frame being exited before its __exit__
            # moves them into the produced node.
            self._flush_span_pending()
        while len(self.frames) > pop_idx:
            self.frames.pop().__exit__(None, None, None)
        if self._spans_enabled and span_entry is not None:
            self._stamp_new_parent_stmts(*span_entry)

    class _has_if_frame: ...

    def ctx_if(self, cond):
        self.check_continue_break()
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            with self.with_frame(tirx.If(cond)):
                yield self._has_if_frame
        else:
            yield cond

    def ctx_then(self, val):
        if val is self._has_if_frame:
            with self.with_frame(tirx.Then()):
                yield
        else:
            if val:
                yield

    def ctx_else(self, val):
        if val is self._has_if_frame:
            with self.with_frame(tirx.Else()):
                yield
        else:
            if not val:
                yield

    def eval(self, val: Any):
        val = unwrap_expr(val)
        if val is None:
            pass
        elif isinstance(val, (tirx.frame.IRBuilderFrame, tirx.LoopFrame, KernelLaunchFrame)):
            if isinstance(val, tirx.frame.ForFrame):
                logger.warning(
                    "A for-loop frame is being evaluated as a standalone expression. Did you mean to use it in a `for` statement?",
                    stacklevel=2,
                )
            self.enter_frame(val)
        elif isinstance(val, tirx.AlreadyEmitted):
            return val
        elif isinstance(val, (PrimExpr, tvm.ir.Call, tvm.ir.Return)):
            # A native Call can carry MissingType until the owning builder
            # infers its result. In particular, handle-returning barrier calls
            # must still become statements; a type predicate must not discard
            # their effects before inference has run.
            tirx.evaluate(val)
        elif isinstance(val, (int, bool)):
            tirx.evaluate(tirx.const(val))
        elif isinstance(val, str):
            pass
        elif isinstance(val, tirx.BufferStore):
            tirx.buffer_store(val.dest, val.value, val.indices, None)
        elif isinstance(val, (Buffer, Var)):
            pass
        else:
            logger.warning(f"Return value `{val}` ({type(val)}) is unused and will be discarded.", stacklevel=2)

    def ctx_for(self, it):
        self.check_continue_break()
        it = unwrap_expr(it)
        if isinstance(it, range):
            from tilelang.language import serial

            it = serial(it.start, it.stop, it.step)
        if not isinstance(it, (SerialForWithStep, UnrollForWithStep, tirx.frame.ForFrame, tirx.LoopFrame)):
            # Python iterables expand the body at IR construction time. TIR
            # values must be excluded explicitly: Var carries an __iter__ shim
            # for single-binding unpacking and Buffer.__getitem__ never raises
            # IndexError, so both would iterate instead of failing.
            self.python_iterable(it, "Invalid for loop")
            try:
                iterator = iter(it)
            except TypeError:
                raise TypeError(
                    f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                    "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding, "
                    "or a Python iterable"
                ) from None
            pos = len(self.frames)
            self.frames.append(PythonLoopFrame())
            try:
                # Keep user-owned generators resumable after a loop break.
                for value in iterator:  # noqa: UP028
                    yield value
            finally:
                # Python loops do not introduce a lexical scope. Keep emitted
                # lets/allocations alive for later iterations and following code.
                self.frames.pop(pos)
            return
        if isinstance(it, (SerialForWithStep, UnrollForWithStep)):
            # Validate and compute the trip count before constructing the frame
            if isinstance(it.step, (int, IntImm)):
                step_value = it.step if isinstance(it.step, int) else it.step.value
                if step_value == 0:
                    raise ValueError("Invalid stepped serial: step must be non-zero")
                if step_value > 0:
                    real_stop = tirx.ceildiv(it.stop - it.start, step_value)
                else:
                    real_stop = tirx.ceildiv(it.start - it.stop, -step_value)
            else:
                logger.warning(
                    f"Non-constant step `{it.step}` in serial range may produce unexpected results. Consider using a constant step if possible.",
                    stacklevel=2,
                )
                real_stop = tirx.ceildiv(it.stop - it.start, it.step)
            if isinstance(it, UnrollForWithStep):
                real_frame = tirx.unroll(real_stop, annotations=it.annotations)
            elif isinstance(it, SerialForWithStep):
                real_frame = tirx.serial(real_stop, annotations=it.annotations)
            else:
                raise TypeError(
                    f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                    "range, T.serial, T.unroll, T.grid, T.parallel, T.vectorized, T.thread_binding"
                )
            with self.with_frame(real_frame) as v:
                IRBuilder.name("_tmp", v)
                yield it.start + v * it.step
        else:
            if not isinstance(it, (tirx.frame.ForFrame, tirx.LoopFrame)):
                raise TypeError(
                    f"Invalid for loop, got {it}({type(it)}), expect one of the following: "
                    "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding"
                )
            with self.with_frame(it) as v:
                yield v

    def _is_python_loop_control(self):
        idx = self.find_frame_idx((PythonLoopFrame, tirx.frame.ForFrame, tirx.LoopFrame, tirx.frame.WhileFrame, MacroFrame))
        if idx is None or not isinstance(self.frames[idx], PythonLoopFrame):
            return False
        if self.find_frame_idx(tirx.frame.IfFrame, start=idx + 1) is not None:
            raise NotImplementedError(
                "Cannot lower break/continue targeting a compile-time Python iterable loop under a device-side condition: "
                "the expanded loop has no runtime control-flow target. Use T.serial over runtime-indexable data, "
                "or a compile-time condition."
            )
        return True

    def ctx_continue(self):
        self.check_continue_break()
        if self._is_python_loop_control():
            raise self.PythonLoopContinue
        # add a dummy frame for checking code after continue/break
        self.enter_frame(ContinueFrame())
        tirx.continue_loop()

    def ctx_break(self):
        self.check_continue_break()
        if self._is_python_loop_control():
            raise self.PythonLoopBreak
        # add a dummy frame for checking code after continue/break
        self.enter_frame(BreakFrame())
        tirx.break_loop()

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
                    "While loop condition is always false; the loop body will be skipped.\n"
                    f"Condition: {cond_v} ({type(cond_v)}) => {cond_v_unwrap} ({type(cond_v_unwrap)})\n",
                    stacklevel=2,
                )
        with self.with_frame(tirx.While(cond_v_unwrap)):
            yield None

    def bind(self, name, value, annot=BaseBuilder.empty, *, loop_target=False):
        self.check_continue_break()

        # in prim func, before T.match_buffer
        # user may write some shape size expression like
        #   ```py
        #   M = T.const('M')
        #   M_2 = M * 2
        #   A = T.match_buffer(A, (M, M_2))
        #   ```
        # If not deal properly, M_2 will be treated as a LetStmt, and causes error in match_buffer
        # here we do a quick check in prim_func_frame, if the value is pure expr, we directly return it
        if (
            isinstance(value, PrimExpr)
            and isinstance(self.frames[-1], tirx.frame.PrimFuncFrame)
            and side_effect(value) <= CallEffectKind.Pure.value
        ):
            return value

        locals = self.get_parent_locals()

        # Handle type annotation
        if value is self.empty:
            orig_value = locals.get(name, value)
            pending = self._deferred_parameters.get(name)
            if pending is not None and isinstance(orig_value, Var) and orig_value.same_as(pending):
                if not _is_runtime_parameter_annotation(annot):
                    raise TypeError(f"Parameter `{name}` requires a runtime tensor, pointer or scalar annotation, got {annot}")
                parameter = tirx.arg(name, annot)
                self._parameter_order[name] = parameter
                return self.with_buffer_span(parameter) if isinstance(parameter, Buffer) else parameter
            if isinstance(annot, Buffer) and str(annot.scope()) == "global":
                from tilelang.language import match_buffer

                return self.with_buffer_span(
                    IRBuilder.name(
                        name,
                        tirx.bind(
                            match_buffer(
                                orig_value,
                                annot.shape,
                                _native.dtype_of(annot),
                                strides=tensor_strides(annot),
                            )
                        ),
                    )
                )
            else:
                return orig_value

        # Only a live Ref/alloc_var is a store target; loop targets and the
        # rewriter's `_` temporaries always bind fresh.
        if loop_target or name == "_" or self.binding_expired(name):
            orig_value = self.empty
        else:
            orig_value = locals.get(name, self.empty)

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
        # Convert deferred comparison ops to PrimExpr so that Ref/alloc_var
        # store paths below can recognise them as assignable values.
        if isinstance(value, (EqualOp, NotEqualOp)):
            value = value.__tvm_ffi_object__()

        if isinstance(orig_value, Ref) and isinstance(value, (int, float, PrimExpr)):
            orig_value.store(value)
            return orig_value
        if is_var(orig_value) and isinstance(value, (int, float, PrimExpr)):
            tirx.buffer_store(orig_value, value, 0)
            return orig_value

        # 2. Quick return for trivil types
        if isinstance(value, (tuple, list, tvm_ffi.Array, int, float, str)):
            self.name_inside_frame.pop(name, None)
            return value
        if isinstance(value, tirx.IntImm) and dtype_of(value) == "int32":
            return value.value
        if isinstance(value, (Var, Buffer)):
            # Bind TVM Var/Buffer names and also record scope so reusing the same
            # Python name (e.g., loop vars like `i`) across different for-frames
            # works without triggering out-of-scope errors.
            if isinstance(value, Buffer):
                self.with_buffer_span(value)
            if value not in self._matched_parameters.values():
                IRBuilder.name(name, value)
            if name != "_":
                frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
                assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
                self.name_inside_frame[name] = self.frames[frame]
            return value

        # 3. Bind immutable tilelang objects
        res = self.bind_immutable(name, value)
        if isinstance(res, Buffer):
            self.with_buffer_span(res)

        # 4. Check variable scope
        if name != "_":
            frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
            assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
            self.name_inside_frame[name] = self.frames[frame]
        return res

    def binding_expired(self, name: str | None) -> bool:
        """Whether `name` was last bound inside a TIR region that has since closed."""
        frame = self.name_inside_frame.get(name)
        return frame is not None and frame not in self.frames

    def unpack_value(self, value):
        value = self.unwrap_value(value)
        return (value,) if isinstance(value, Var) else value

    def unwrap_value(self, value):
        """
        Unwrap some tilelang objects to get their inner value
        """
        value = unwrap_expr(value)
        # handle bx, by = tl.Kernel(128, 128), rval is frame
        if isinstance(value, (tirx.frame.IRBuilderFrame, tirx.LoopFrame, KernelLaunchFrame)):
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
        if isinstance(value, tirx.meta_var):
            return value.value
        elif isinstance(value, (tirx.frame.IRBuilderFrame, tirx.LoopFrame, KernelLaunchFrame)):
            return self.enter_frame(value)
        elif isinstance(value, OutTensor):
            arg = tirx.arg(
                name,
                tirx.buffer(
                    shape=value.shape,
                    dtype=dtype_of(value),
                    strides=value.strides,
                ),
            )
            arg._out_idx = self.out_tensor_cnt
            self.out_tensor_cnt += 1
            return arg
        elif isinstance(value, (Buffer, tirx.Var)):
            IRBuilder.name(name, value)
            return value
        elif isinstance(value, BufferRegion):
            # Regions are native indexing expressions. They carry the tensor
            # and slice directly and have no separately nameable definition.
            return value
        elif isinstance(value, tvm.ir.Expr):
            # Native calls may return a tensor, tuple or function. Every such
            # result needs a typed Bind before consumers use its variable;
            # primitive-expression membership alone misses tensor views.
            var = tirx.bind(value)
            register_let_value(var, value)
            IRBuilder.name(name, var)
            return var
        else:
            return value

    @staticmethod
    def _check_buffer_slice_step(value, indices):
        # This is the public TileLang Buffer slicing contract, independent of
        # the direct native IR subscript API. Ordinary Python values retain
        # their own slicing and __getitem__ behavior.
        if isinstance(value, Buffer):
            indices = indices if isinstance(indices, (tuple, list)) else (indices,)
            if any(isinstance(index, slice) and index.step is not None for index in indices):
                raise RuntimeError("Buffer slicing with step is not supported.")

    def getitem(self, value: Any, sl: slice):
        self._check_buffer_slice_step(value, sl)
        return super().getitem(value, sl)

    def assign_slice(self, lval: Any, sl: slice, value: Any, annot=BaseBuilder.empty):
        self.check_continue_break()
        if annot is not self.empty:
            logger.warning("Type annotation on slice assignment is not supported and will be ignored.", stacklevel=2)
        if isinstance(lval, Buffer):
            # Unlike a Buffer read, the original store builder accepted a
            # slice as a vector index. Preserve its Ramp/element contract.
            indices = sl if isinstance(sl, (tuple, list)) else (sl,)
            if any(isinstance(index, slice) for index in indices):
                normalized = []
                analyzer = create_analyzer()
                for index in indices:
                    if isinstance(index, slice):
                        step = 1 if index.step is None else index.step
                        lanes = analyzer.simplify((index.stop - index.start + step - 1) // step)
                        normalized.append(index.start if int(lanes) == 1 else tirx.Ramp(index.start, step, int(lanes)))
                    else:
                        normalized.append(index)
                sl = normalized
            tirx.buffer_store(lval, value, sl)
        else:
            return super().assign_slice(lval, sl, value)

    def aug_assign(self, op, target, aug_value, name: str | None = None):
        self.check_continue_break()
        if isinstance(target, Ref):
            target.store(eval_op(op, target.bufload, aug_value))
            return target
        elif is_var(target):
            tirx.buffer_store(target, eval_op(op, target[0], aug_value), 0)
            return target
        elif isinstance(target, Buffer):
            raise RuntimeError(
                f"Attempting to update buffer `{target}` using augmented assignment.\n"
                "Please use slice assignment, e.g. `buf[0] += value` instead."
            )
        elif isinstance(target, Var):
            # Treat augmented assignment on immutable vars (SSA) as re-binding:
            #   x -= y  ==>  x = x - y
            name = name or getattr(target, "orig_name", None) or target.name  # type: ignore[attr-defined]
            res = eval_op(op, target, aug_value)

            # Mirror the `bind` fast-path: if we're at the prim_func frame and the
            # expression is pure, keep it as a raw PrimExpr to avoid creating
            # LetStmts before match_buffer.
            if (
                isinstance(res, PrimExpr)
                and isinstance(self.frames[-1], tirx.frame.PrimFuncFrame)
                and side_effect(res) <= CallEffectKind.Pure.value
            ):
                return res

            res = self.bind_immutable(name, res)
            if name != "_":
                frame = self.find_frame_idx(TIR_VAR_SCOPE_FRAME)
                assert frame is not None, f"Variable `{name}` is not defined inside any control flow."
                self.name_inside_frame[name] = self.frames[frame]
            return res
        else:
            return super().aug_assign(op, target, aug_value, name=name)

    def aug_assign_slice(self, op, target, sl, aug_value):
        self.check_continue_break()
        self._check_buffer_slice_step(target, sl)
        if isinstance(target, Buffer):
            tirx.buffer_store(target, eval_op(op, target[sl], aug_value), sl)
        else:
            return super().aug_assign_slice(op, target, sl, aug_value)

    def boolop(self, op, left, right=None):
        left = unwrap_cond(left)
        if isinstance(left, PrimExpr):
            with self.with_frame(BoolOpFrame()):
                if op == "And":
                    return tirx.And(left, right())
                if op == "Or":
                    return tirx.Or(left, right())
                if op == "Not":
                    return tirx.Not(left)
            raise RuntimeError(f"Unsupported boolean operator: {op}")
        else:
            return super().boolop(op, left, right)

    def unaryop(self, op, operand):
        if op == "UAdd" and isinstance(operand, PrimExpr):
            # PrimExpr overloads unary minus and invert but not unary plus.
            # Unary plus is the identity on numbers, so hand back the operand.
            return operand
        return super().unaryop(op, operand)

    def ifexp(self, cond, then, otherwise):
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            with self.with_frame(BoolOpFrame()):
                return tirx.if_then_else(cond, then(), otherwise())
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
                    "In tilelang macro, return from control flow is not supported yet. \n"
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
            if self.eager_jit == "phase1":
                return NotImplemented
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

    def ctx_with(self, ctx, *, unpack=False):
        self.check_continue_break()
        if isinstance(ctx, (tirx.frame.IRBuilderFrame, tirx.LoopFrame, tirx.BlockFrame, tirx.InitFrame, KernelLaunchFrame)):
            ctx = self.with_frame(ctx)
        return super().ctx_with(ctx, unpack=unpack)

    def assert_expr(self, cond, msg=None):
        self.check_continue_break()
        cond = unwrap_cond(cond)
        if msg is None:
            msg = "Assertion failed"
        if isinstance(cond, PrimExpr):
            tirx.assert_(cond, msg)
        elif not cond:
            raise AssertionError(msg)

    def rval(self, name: str | None, value: Any) -> Any:
        if self.binding_expired(name):
            raise RuntimeError(
                f"Immutable variable `{name}` is used outside its defining region!\n"
                f"variable `{name}` is defined in frame: {self.name_inside_frame[name]}, current frames: {self.frames}."
            )
        return self.unwrap_value(value)

    def macro_arg(self, name, value):
        annot_value = self.macro_arg_annot.get(name, None)
        if annot_value is Var or annot_value is Ref:
            if annot_value is Var:
                logger.warning("Use `T.Var` as macro annotations is deprecated, please use `T.Ref`")
            if isinstance(value, BufferLoad):
                if is_var(value.source):
                    return value.source
                idx = [idx if isinstance(idx, Var) else self.bind("_", idx) for idx in value.indices]
                # indices = self.bind(f'_', value.indices)
                return Ref(BufferLoad(value.source, indices=idx))
            if isinstance(value, BufferRegion):
                region = [
                    Range.from_min_extent(
                        x.begin if isinstance(x.begin, Var) else self.bind("_", x.begin),
                        x.extent if isinstance(x.extent, Var) else self.bind("_", x.extent),
                    )
                    for x in value.region
                ]
                return BufferRegion(value.source, region=region)
            raise ValueError(
                f"To pass as reference, argument `{name}` is expected to be a variable or a buffer region, but got {value}({type(value)})"
            )
        elif isinstance(value, Var):
            # Macro parameters borrow the caller's definition. Naming them
            # mutates that existing Var (and its owner Function) in place.
            # Keep the identity so captures, tensor types and layout references
            # still refer to the caller's definition after macro expansion.
            return value
        elif isinstance(value, (PrimExpr, int, float)):
            return self.bind(name, value)
        else:
            return value

    def prim_func_arg(self, name, value):
        if _is_runtime_parameter_annotation(value):
            if isinstance(value, Var) and getattr(value, "_tilelang_deferred_parameter", False):
                placeholder = Var(name, value.ty, self.current_location)
                self._deferred_parameters[name] = placeholder
                self._parameter_order[name] = None
                return placeholder
            arg = tirx.arg(name, value)
            self._parameter_order[name] = arg
            # Stamp parameter buffers with the signature line (SpanAttacher
            # emits set_fileline with the parameter's own line before each
            # `__tb.arg(...)` call), so diagnostics involving only argument
            # buffers (e.g. T.copy(A, B)) also carry a source location.
            if isinstance(arg, Buffer):
                self.with_buffer_span(arg)
            return arg
        elif value is self.empty:
            raise ValueError(f"Argument `{name}` is not annotated")
        elif isinstance(value, Hashable):
            return value
        else:
            raise TypeError(f"Unsupported argument type: {value}({type(value)}) for argument `{name}`.")

    def arg(self, name, value):
        if self.find_frame_idx(MacroFrame) is not None:
            return self.macro_arg(name, value)
        else:
            return self.prim_func_arg(name, value)

    def iter_call(self, func, /, *args, **kwargs):
        from tilelang.language import serial

        if func is builtins.range:
            return serial(*args, **kwargs)
        return func(*args, **kwargs)

    def python_iterable(self, value, context):
        """Reject device values before Python can acquire an iterator over them."""
        if isinstance(value, (PrimExpr, Buffer, BufferRegion, Ref, EqualOp, NotEqualOp)):
            raise TypeError(
                f"{context}: a TIR expression or buffer is not iterable (got {type(value).__name__}). "
                "Use range(n) or T.serial(n) in a for statement to loop over a runtime extent, "
                "or iterate over a Python container such as [expr] or buffer.shape."
            )
        if isinstance(value, (SerialForWithStep, tirx.frame.IRBuilderFrame, tirx.LoopFrame)):
            raise TypeError(
                f"{context}: a TileLang loop or IR builder frame is not a compile-time Python iterable. "
                "Use TileLang loop constructors directly in a for statement; "
                "use range with compile-time bounds in a comprehension."
            )
        return value

    def resolve_call(self, func):
        entry = _BUILTIN_ITERABLE_ARG_MAP.get(id(func))
        if entry is None:
            return func
        return partial(self._call_python_builtin, *entry)

    def _call_python_builtin(self, func, select_iterables, /, *args, **kwargs):
        """Check iterable operands before a builtin hides or consumes them."""
        for value in select_iterables(args, kwargs):
            self.python_iterable(value, f"{func.__name__}()")
        return func(*args, **kwargs)

    def comprehension_filter(self, cond):
        cond = unwrap_cond(cond)
        if isinstance(cond, PrimExpr):
            raise TypeError("Comprehension filters must be evaluable at compile time; use a serial loop with an if statement instead.")
        return cond

    def constexpr(self, name: str, dtype: str = "int32") -> Var:
        var = tirx.Var(name, dtype)
        self.constexpr_var.add(var)
        var.orig_name = name
        return var

    def set_fileline(self, filename: str, lineno: int, name: str):
        self.current_file = filename
        self.current_line = lineno
        self.current_macro_name = name
        if self._spans_enabled:
            self._flush_span_pending()
            frame = self._current_stmt_frame()
            if frame is not None:
                self._span_pending = (frame, len(frame.stmts), filename, lineno)
                if self._span_first_fileline is None and lineno > 0:
                    self._span_first_fileline = (filename, lineno)

    def get_fileline_stack(self, stacklevel=1):
        stack = self.macro_fileline_stack + [(self.current_file, self.current_line, self.current_macro_name)]
        return stack[: len(stack) - stacklevel + 1]

    def skip_kernel_ctx(self):
        return self.eager_jit == "phase1"


_P = ParamSpec("_P")
_T = TypeVar("_T")


if TYPE_CHECKING:

    class PrimFunc(Generic[_P, _T], _native.Function):
        params: list[tvm.ir.Var | Buffer]
        body: tvm.ir.Stmt
        ret_type: tvm.ir.Type
        attrs: tvm.ir.DictAttrs | None
        span: Span | None
        ir_gen: IRGenerator[_P, _T] | None
        orig_func: Callable[_P, _T] | None

else:
    PrimFunc = _native.Function


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
        builder = Builder.current()
        if builder is None:
            raise JITNoBuilderError("T.macro can only be used inside @tilelang.jit")

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
import re


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
    #   the only bug scenario the function body doesn't use the parameters
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
        if isinstance(value, tvm_ffi.dtype):
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
            value = ForwardRef(value, module=func.__module__)
            hints[name] = _eval_type(value, globalns=globalns, localns=localns)
        else:
            hints[name] = value
    return hints


def const(name: str, dtype: str = "int32") -> Var | tuple[Var, ...]:
    """
    Declare constexpr variables for dynamic tensor dimensions (eager mode only).

    In eager mode, use T.const() to declare shape dimensions that will be
    inferred from actual tensor arguments at runtime.

    Example::

        @tilelang.jit
        def kernel(A, B):
            M, N = T.const("M, N")
            A: T.Tensor[[M, N], T.float32]
            ...
    """
    builder = Builder.current()
    # assert builder is not None, "T.const() can only be used inside @tilelang.jit (eager mode)"
    # assert builder.eager_jit, "T.const() can only be used inside @tilelang.jit (eager mode)"
    if builder is None or builder.eager_jit == "none":
        raise JITNoBuilderError("T.const() can only be used inside @tilelang.jit (eager mode)")

    if builder.eager_jit == "phase1":
        # in stage 1, we create constexpr variables
        if "," in name:
            names = re.split(r"\s*,\s*", name)
            return tuple(builder.constexpr(n, dtype) for n in names)
        if " " in name:
            names = re.split(r"\s+", name)
            return tuple(builder.constexpr(n, dtype) for n in names)
        else:
            return builder.constexpr(name, dtype)
    elif builder.eager_jit == "phase2":
        # in stage 2, we substitute constexpr variables with actual values
        if "," in name:
            names = re.split(r"\s*,\s*", name)
            return tuple(builder.eager_jit_subs[n] for n in names)
        if " " in name:
            names = re.split(r"\s+", name)
            return tuple(builder.eager_jit_subs[n] for n in names)
        else:
            return builder.eager_jit_subs[name]


def annotate_compile_flags(flags: list[str] | str) -> None:
    """
    Annotate additional device compile flags inside a function body.

    The flags will be merged with any externally provided compile_flags
    at compilation time. Can be placed before or after tensor type annotations.

    Example::

        @tilelang.jit
        def kernel(A, B):
            T.annotate_compile_flags(["--use_fast_math"])
            ...
    """
    builder = Builder.current()
    if builder is None:
        raise JITNoBuilderError("T.annotate_compile_flags() can only be used inside @tilelang.jit or @T.prim_func")
    if builder.eager_jit == "phase1":
        return
    builder.func_compile_flags = flags


def annotate_pass_configs(configs: dict[str, Any]) -> None:
    """
    Annotate pass configuration inside a function body.

    The configs will be merged with any externally provided pass_configs
    at compilation time (function-level configs take lower priority, i.e.
    external configs override). Can be placed before or after tensor type annotations.

    Example::

        @tilelang.jit
        def kernel(A, B):
            T.annotate_pass_configs({
                PassConfigKey.TL_ENABLE_FAST_MATH: True})
            ...
    """
    builder = Builder.current()
    if builder is None:
        raise JITNoBuilderError("T.annotate_pass_configs() can only be used inside @tilelang.jit or @T.prim_func")
    if builder.eager_jit == "phase1":
        return
    builder.func_pass_configs = configs


def _patch_prim_func_attrs(pf: PrimFunc, builder: Builder) -> PrimFunc:
    """Attach function-level out_idx, pass_configs and compile_flags as PrimFunc attrs."""
    if builder.out_idx:
        pf = pf.with_attr("tilelang_out_idx", builder.out_idx)
    if builder.func_pass_configs is not None:
        pf = pf.with_attr("tilelang_pass_configs", builder.func_pass_configs)
    if builder.func_compile_flags is not None:
        flags = builder.func_compile_flags
        if isinstance(flags, str):
            flags = [flags]
        pf = pf.with_attr("tilelang_compile_flags", flags)
    return pf


@dataclass
class TirTemplate(Generic[_P, _T]):
    """
    Template for generating TIR PrimFunc with dynamic shape substitution.

    For lazy-style functions, the PrimFunc is used directly without substitution.
    For eager-style functions, constexpr variables are substituted based on
    actual tensor shapes at runtime.
    """

    name: str
    prim_func: PrimFunc[_P, _T]
    matcher: dict[Var, tuple[tvm.ir.Var, str, int, str]] | None = None
    constexprs: set[Var] = None
    is_lazy_style: bool = False  # True if from lazy-style (returns PrimFunc directly)
    ir_gen: IRGenerator[_P, _T] | None = None
    private: bool = False

    @classmethod
    def create(
        cls,
        name: str,
        prim_func: PrimFunc[_P, _T],
        constexpr: set[Var],
        ir_gen: IRGenerator[_P, _T] | None = None,
        *,
        private: bool = False,
    ) -> TirTemplate[_P, _T]:
        matcher = {}
        for v in prim_func.params:
            if not tirx.is_tensor(v):
                continue
            k = v
            for i, s in enumerate(v.shape):
                if s in constexpr and s not in matcher:
                    matcher[s] = (k.name, "shape", i, s.name)
            for i, s in enumerate(tensor_strides(v)):
                if s in constexpr and s not in matcher:
                    matcher[s] = (k.name, "stride", i, s.name)
        for s in constexpr:
            if s not in matcher:
                shapes = {v.name: v.shape for v in prim_func.params if tirx.is_tensor(v)}
                strides = {v.name: tensor_strides(v) for v in prim_func.params if tirx.is_tensor(v)}
                raise RuntimeError(
                    f"Constexpr variable `{s}` is not used in any buffer shape or stride.\n"
                    "At least one **DIRECT** usage is required. Please check:\n"
                    "(1) the variable is not used\n"
                    f"(2) all uses are indirect, e.g. {s} * 2, {s} * 3. (you can replace them with separate constexpr variables)\n"
                    f"Buffer shapes: {shapes}\n"
                    f"Buffer strides: {strides}"
                )
        matcher = {k: matcher[k] for k in constexpr}
        return cls(
            name=name,
            prim_func=prim_func,
            matcher=matcher,
            constexprs=constexpr,
            is_lazy_style=False,
            ir_gen=ir_gen,
            private=private,
        )

    @classmethod
    def from_lazy_style(cls, name: str, prim_func: PrimFunc[_P, _T]) -> TirTemplate[_P, _T]:
        """Create template from lazy-style function that returns PrimFunc directly."""
        return cls(name=name, prim_func=prim_func, is_lazy_style=True)

    def _parse_phase2_key(self, **kwargs):
        if self.matcher is None:
            return ()
        result = []
        buffers = {param.name: param for param in self.prim_func.params if tirx.is_tensor(param)}
        for k, ty, i, name in self.matcher.values():
            if name in kwargs:
                result.append(kwargs.get(name))
            elif k in kwargs:
                value = kwargs[k]
                buffer = buffers[k]
                if ty == "shape":
                    result.append(_logical_shape_value(value, buffer, i))
                elif ty == "stride":
                    result.append(_logical_stride_value(value, buffer, i))
            else:
                raise ValueError(
                    f"Cannot find value for constexpr variable `{name}`\n"
                    f"Please provide it as a keyword argument, e.g. `{name}=<value>`\n"
                    f"Or provide the corresponding tensor argument `{k}`."
                )
        return tuple(result)

    def get_tir(self, tensor_args, given_tensor_args, kwargs):
        if self.is_lazy_style:
            return self.prim_func
        values = self._parse_phase2_key(**given_tensor_args, **kwargs)
        subs = {name.orig_name: value for name, value in zip(self.matcher, values)}
        builder = Builder()
        builder.eager_jit = "phase2"
        builder.eager_jit_subs = subs
        with builder.prim_func(self.name, private=self.private):
            self.ir_gen.gen(builder)(**tensor_args, **kwargs)
        pf = builder.get()
        pf = _patch_prim_func_attrs(pf, builder)
        return pf


_MISSING_ARG = object()
_ExtraJITKwargsKey = tuple[tuple[str, Any], ...]


def _make_jit_key(values: tuple[Any, ...], *, context: str = "JIT compile-time arguments") -> tuple[Any, ...]:
    """Validate compile-time values without changing their cache-key semantics."""

    try:
        hash(values)
    except TypeError as exc:
        raise TypeError(
            f"{context} must be hashable: {exc}. Use immutable, hashable values such as tuples instead of lists or dicts."
        ) from exc
    return values


@dataclass(frozen=True)
class _BoundJITArgs:
    """JIT arguments split into phase-1 key, runtime tensors, and compile kwargs."""

    p1_key: tuple[Any, ...]
    tensor_args: dict[str, Any]
    compile_kwargs: dict[str, Any]


class _JITArgumentBinder:
    """Bind Python JIT calls into TileLang's phase-1 cache inputs.

    The binder keeps three pieces of data separate:

    - phase-1 key values: compile-time arguments that identify the TIR template
    - tensor args: runtime tensors used later for phase-2 shape/stride matching
    - compile kwargs: raw values passed back to the user's Python JIT function

    The common path avoids inspect.Signature.bind() because this code runs before
    every cache lookup. Signatures with *args or **kwargs still use Python's
    binder because their call semantics are harder to reproduce safely.
    """

    def __init__(
        self,
        signature: inspect.Signature,
        tensor_arg_names: set[str],
    ):
        self.signature = signature
        self.tensor_arg_names = tensor_arg_names
        self.params = tuple(signature.parameters.values())
        for param in self.params:
            if param.name not in tensor_arg_names and param.default is not inspect.Parameter.empty:
                _make_jit_key((param.default,), context=f"Default value for JIT compile-time parameter {param.name!r}")
        self.param_names = tuple(param.name for param in self.params)
        self.param_index = {name: i for i, name in enumerate(self.param_names)}
        self.defaults = tuple(param.default if param.default is not inspect.Parameter.empty else _MISSING_ARG for param in self.params)
        self.required_indices = tuple(i for i, value in enumerate(self.defaults) if value is _MISSING_ARG)
        self.positional_param_count = sum(
            1 for param in self.params if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        self.use_signature_bind = any(
            param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for param in self.params
        )

    def bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _BoundJITArgs:
        if self.use_signature_bind:
            return self._bind_with_signature(args, kwargs)
        return self._bind_fast(args, kwargs)

    def bind_no_tensor_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        if self.tensor_arg_names:
            raise ValueError("bind_no_tensor_key is only valid for JIT functions without tensor arguments")
        if self.use_signature_bind:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return self._pack_no_tensor_key(bound.arguments)
        return self._bind_fast_no_tensor_key(args, kwargs)

    @staticmethod
    def _extra_key(extra_kwargs: dict[str, Any]) -> _ExtraJITKwargsKey:
        # Extra kwargs are compile-time values for explicit T.const bindings.
        # Sort names so f(M=1, N=2) and f(N=2, M=1) share one key.
        return tuple(sorted(extra_kwargs.items()))

    def _pack_no_tensor_key(self, arguments: dict[str, Any]) -> tuple[Any, ...]:
        """Pack a lazy no-tensor call directly into a phase-1 key."""

        p1_values: list[Any] = []
        for param in self.params:
            value = arguments[param.name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # Signature.bind() stores **kwargs as one dict under the
                # VAR_KEYWORD parameter. Flatten it so each keyword remains a
                # normal compile-time key component.
                if value:
                    p1_values.append(self._extra_key(value))
            else:
                p1_values.append(value)
        return _make_jit_key(tuple(p1_values))

    def _pack_bound_arguments(
        self,
        arguments: dict[str, Any],
        extra_kwargs: dict[str, Any] | None = None,
    ) -> _BoundJITArgs:
        p1_values: list[Any] = []
        tensor_args: dict[str, Any] = {}
        compile_kwargs: dict[str, Any] = {}
        extra_kwargs = extra_kwargs or {}
        for param in self.params:
            name = param.name
            value = arguments[name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # Keep **kwargs flattened both in the key and in compile_kwargs;
                # passing {"kwargs": {...}} would change the user's call form.
                if value:
                    p1_values.append(self._extra_key(value))
                    compile_kwargs.update(value)
                continue
            if name in self.tensor_arg_names:
                # Tensor args are runtime inputs. They do not identify the TIR
                # template, but they are needed for phase-2 shape/stride keys.
                tensor_args[name] = value
            else:
                p1_values.append(value)
                compile_kwargs[name] = value
        if extra_kwargs:
            p1_values.append(self._extra_key(extra_kwargs))
            compile_kwargs.update(extra_kwargs)
        return _BoundJITArgs(_make_jit_key(tuple(p1_values)), tensor_args, compile_kwargs)

    def _bind_with_signature(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _BoundJITArgs:
        """Slow path for signatures that need Python's full binding rules."""

        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return self._pack_bound_arguments(bound.arguments)

    def _bind_fast(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _BoundJITArgs:
        """Fast binder for normal positional/keyword-only JIT signatures."""

        if len(args) > self.positional_param_count:
            raise TypeError(
                f"too many positional arguments for JIT function: expected at most {self.positional_param_count}, got {len(args)}"
            )

        values = list(self.defaults)
        extra_kwargs: dict[str, Any] = {}
        for i, value in enumerate(args):
            values[i] = value

        for name, value in kwargs.items():
            index = self.param_index.get(name)
            if index is None:
                # TileLang allows extra compile-time kwargs for explicit T.const bindings.
                extra_kwargs[name] = value
                continue
            if index < len(args):
                raise TypeError(f"got multiple values for argument '{name}'")
            values[index] = value

        for index in self.required_indices:
            if values[index] is _MISSING_ARG and self.param_names[index] not in self.tensor_arg_names:
                raise TypeError(f"missing a required argument: '{self.param_names[index]}'")

        p1_values: list[Any] = []
        tensor_args: dict[str, Any] = {}
        compile_kwargs: dict[str, Any] = {}
        for name, value in zip(self.param_names, values):
            if name in self.tensor_arg_names:
                if value is not _MISSING_ARG:
                    tensor_args[name] = value
            else:
                p1_values.append(value)
                compile_kwargs[name] = value
        if extra_kwargs:
            p1_values.append(self._extra_key(extra_kwargs))
            compile_kwargs.update(extra_kwargs)
        return _BoundJITArgs(_make_jit_key(tuple(p1_values)), tensor_args, compile_kwargs)

    def _bind_fast_no_tensor_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        """Fastest path for lazy factories where every argument is compile-time."""

        if len(args) > self.positional_param_count:
            raise TypeError(
                f"too many positional arguments for JIT function: expected at most {self.positional_param_count}, got {len(args)}"
            )

        if not kwargs:
            # Common lazy cache-hit form: positional values, no tensor args, no
            # kwargs. Fill omitted defaults without allocating _BoundJITArgs.
            if len(args) == len(self.param_names):
                return _make_jit_key(args)
            for index in self.required_indices:
                if index >= len(args):
                    raise TypeError(f"missing a required argument: '{self.param_names[index]}'")
            return _make_jit_key(args + self.defaults[len(args) :])

        values = list(self.defaults)
        extra_kwargs: dict[str, Any] = {}
        for i, value in enumerate(args):
            values[i] = value

        for name, value in kwargs.items():
            index = self.param_index.get(name)
            if index is None:
                # TileLang allows extra compile-time kwargs for explicit T.const bindings.
                extra_kwargs[name] = value
                continue
            if index < len(args):
                raise TypeError(f"got multiple values for argument '{name}'")
            values[index] = value

        for index in self.required_indices:
            if values[index] is _MISSING_ARG:
                raise TypeError(f"missing a required argument: '{self.param_names[index]}'")

        if extra_kwargs:
            values.append(self._extra_key(extra_kwargs))
        return _make_jit_key(tuple(values))


@dataclass
class JITFunc(Generic[_P, _T]):
    """
    Internal wrapper for JIT-compiled functions.

    This class handles both lazy and eager execution styles:

    - **lazy style**: Function explicitly returns a PrimFunc. The original function
      is called directly to obtain the TIR.

    - **eager style**: Function uses the DSL builder pattern with tensor type
      annotations. The TIR is constructed by tracing the function body through
      the Builder.

    The style is determined by `_is_lazy_style()` which checks if calling the
    original function returns a PrimFunc directly.
    """

    orig_func: Callable[_P, _T]
    signature: inspect.Signature
    arg_names: list[str]
    tensor_args: dict[str, Buffer | Var | tvm.ir.Type]
    tensor_args_defaults: dict[str, Any]
    ir_gen: IRGenerator[_P, _T]
    mode: Literal["auto", "lazy", "eager"] = "auto"
    private: bool = False

    def __post_init__(self):
        # we don't want it to show up in the constructor
        self.p1_cache: dict[Any, TirTemplate[_P, _T]] = {}
        self._argument_binder = _JITArgumentBinder(self.signature, set(self.tensor_args))

    def _parse_phase1_key(self, *args, **kwargs):
        bound = self._argument_binder.bind(args, kwargs)
        return bound.p1_key, bound.tensor_args, bound.compile_kwargs

    def _is_lazy_style(self, *args, **kwargs) -> bool:
        """
        Check if the function uses lazy style (explicitly returns PrimFunc).

        Lazy style functions define an inner @T.prim_func and return it:
            @jit
            def foo(M, N):
                @T.prim_func
                def kernel(...): ...
                return kernel  # <- returns PrimFunc

        Eager style functions use the builder pattern with type annotations:
            @jit
            def foo(A, B):
                A: T.Tensor[...]
                with T.Kernel(...): ...
                # no return
        """
        if has_internal_prim_func(self.orig_func):
            return True
        try:
            inspect.signature(self.orig_func).bind(*args, **kwargs)
        except TypeError:
            return False
        bound = self._argument_binder.bind(args, kwargs)
        try:
            prim_func = self.orig_func(*args, **kwargs)
            # lazy jit must return PrimFunc
            if isinstance(prim_func, PrimFunc):
                if self.private:
                    prim_func = prim_func.without_attr("global_symbol")
                self.p1_cache[bound.p1_key] = TirTemplate.from_lazy_style(self.orig_func.__name__, prim_func)
                return True
            return False
        except (JITNoBuilderError, EagerJITBuildError):
            # In eager mode, we construct AST directly without prim_func,
            # so there's no Builder available when the function is called.
            # When eager-only features like T.const() or T.Kernel() are used,
            # they raise JITNoBuilderError because no Builder exists yet.
            # This indicates the function is eager-style, not lazy-style.
            return False

    def _build_tir_template(self, *args, **kwargs) -> TirTemplate[_P, _T]:
        """Build TIR template based on the execution mode."""
        if self.mode == "lazy":
            # lazy: function returns PrimFunc directly
            prim_func = self.orig_func(*args, **kwargs)
            if self.private:
                prim_func = prim_func.without_attr("global_symbol")
            return TirTemplate.from_lazy_style(self.orig_func.__name__, prim_func)
        elif self.mode == "eager":
            # eager: trace function body through Builder to construct TIR
            builder = Builder()
            builder.eager_jit = "phase1"
            with builder.prim_func(self.orig_func.__name__, private=self.private):
                self.ir_gen.gen(builder)(**self.tensor_args, **kwargs)
            pf = builder.get()
            pf.orig_func = self.orig_func
            return TirTemplate.create(self.orig_func.__name__, pf, builder.constexpr_var, self.ir_gen, private=self.private)
        else:
            raise ValueError(f"Invalid jit mode: {self.mode}, expected 'lazy' or 'eager'")

    def parse_args(self, *args, **kwargs):
        """Parse arguments and return cache key and tensor args."""
        if not self.tensor_args:
            p1_key = self._argument_binder.bind_no_tensor_key(args, kwargs)
            return (p1_key, None), {}

        bound = self._argument_binder.bind(args, kwargs)
        tir_temp = self.p1_cache.get(bound.p1_key, None)
        if tir_temp is None:
            # mode should be set by JITImpl before calling parse_args
            tir_temp = self._build_tir_template(**bound.compile_kwargs)
            self.p1_cache[bound.p1_key] = tir_temp
        p2_key = tir_temp._parse_phase2_key(**bound.tensor_args, **bound.compile_kwargs)
        return (bound.p1_key, p2_key), bound.tensor_args

    def get_tir(self, *args, **kwargs):
        bound = self._argument_binder.bind(args, kwargs)
        if bound.p1_key not in self.p1_cache:
            # Lazy factories retain their positional arguments.
            if self.mode == "lazy":
                template = self._build_tir_template(*args, **kwargs)
            else:
                template = self._build_tir_template(**bound.compile_kwargs)
            self.p1_cache[bound.p1_key] = template
        return self.p1_cache[bound.p1_key].get_tir(
            self.tensor_args,
            bound.tensor_args,
            bound.compile_kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.get_tir(*args, **kwargs)

    def set_mode(self, mode: Literal["lazy", "eager"]):
        """Set the JIT execution mode (internal use only)."""
        self.mode = mode

    # Proxy function attributes for compatibility with autotuner and inspect.
    # These attributes are needed by autotuner to extract closure variables
    # and generate cache keys.
    _PROXIED_ATTRS = frozenset({"__closure__", "__code__", "__name__", "__globals__", "__wrapped__"})

    def __getattr__(self, name):
        if name in JITFunc._PROXIED_ATTRS:
            if name == "__wrapped__":
                return self.orig_func
            return getattr(self.orig_func, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def substitute_primfunc(prim_func, vmap):
    """Substitute typed variables while preserving the function's calling convention.

    Shape and layout symbols may become scalar expressions. Explicit parameters
    and local variable definitions remain Vars, so this helper neither drops
    arguments nor renumbers output indices. Eager JIT uses AST reconstruction
    in ``TirTemplate.get_tir`` rather than this directly callable helper.
    """
    ir = tvm.ir
    if not isinstance(prim_func, tvm.tirx.Function):
        raise TypeError("substitute_primfunc expects a native Function")
    replacements = {}
    parameters = set(prim_func.params)
    for variable, value in vmap.items():
        if not isinstance(variable, ir.Var):
            raise TypeError("substitution keys must be native Vars")
        if not isinstance(value, ir.Expr):
            value = unwrap_expr(value)
            if not isinstance(value, ir.Expr):
                if not isinstance(variable.ty, ir.PrimType):
                    raise TypeError("non-scalar substitutions require an expression with a matching type")
                value = ir.const(value, variable.ty.dtype)
        if variable in parameters and not isinstance(value, ir.Var):
            raise TypeError("a function parameter must remain a Var after substitution")
        replacements[variable] = value

    analyzer = create_analyzer()

    def replace_variable(variable, mutator):
        cached = mutator.var_remap_get(variable)
        if isinstance(cached, ir.Expr):
            return cached
        value = replacements.get(variable)
        if value is None:
            return mutator.default_mutate(variable)
        if mutator.def_region_kind() == tvm_ffi.DefRegionKind.DEF_SIMPLE and not isinstance(value, ir.Var):
            raise TypeError("a local definition must remain a Var after substitution")
        mapped = mutator.default_mutate(variable)
        if isinstance(value, ir.Var):
            # Rebuild a replacement tensor's dependent type with the same
            # substitutions. Never mutate the caller's Var or TensorType.
            value_type = mutator.mutate(value.ty)
            if not value_type.same_as(value.ty):
                source = value
                value = ir.Var(value.name, value_type, value.loc)
                if tvm_ffi.get_global_func("tl.HasSizeVarDefault")(source):
                    tvm_ffi.get_global_func("tl.MarkSizeVarDefault")(value)
        if not tvm_ffi.structural_equal(mapped.ty, value.ty):
            raise TypeError(f"substitution for {variable.name!r} changes its type: {mapped.ty} -> {value.ty}")
        # Custom callbacks return final values. Record the original identity as
        # well so later uses and default typed-definition hooks share one Var.
        mutator.var_remap_set(variable, value)
        return value

    def replace_expression(expression, mutator):
        mapped = mutator.default_mutate(expression)
        return analyzer.simplify(mapped) if ir.is_prim_expr(mapped) else mapped

    result = tvm_ffi.structural_mutate(prim_func, [(ir.Var, replace_variable), (ir.Expr, replace_expression)])
    tvm.tirx.analysis.check_well_formed(result)
    return result


def prim_func(func: Callable[_P, _T] = None, *, eager_jit: bool = False, private: bool = False) -> PrimFunc[_P, _T] | JITFunc[_P, _T]:
    """Build a native Function; private helpers do not acquire an export symbol."""

    def impl(func: Callable[_P, _T]) -> PrimFunc[_P, _T] | Callable[_P, PrimFunc[_P, _T]]:
        sig = inspect.signature(func)
        ir_gen = mutate(func)
        func_annot = get_type_hints(func)
        annot = {}
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_ONLY:
                raise TypeError(f"PrimFunc does not support positional-only parameters: `{param.name}`")
            if param.name in ir_gen.extra_type_hints:
                annot[param.name] = ir_gen.extra_type_hints[param.name]
            elif param.name in func_annot:
                annot[param.name] = func_annot[param.name]
        for k in annot:
            # Factories construct annotations. Native expressions are already
            # annotations: Expr.__call__ builds an IR call, including on pointer
            # Vars returned by T.handle(...), and must not run as a factory.
            # Typing generics such as Optional[int] cannot be instantiated either.
            if not isinstance(annot[k], (type, tvm.ir.Expr, tvm.ir.Type)) and callable(annot[k]) and get_origin(annot[k]) is None:
                annot[k] = annot[k]()

        if eager_jit:
            arg_names = list(sig.parameters.keys())
            tensor_args = {k: v for k, v in annot.items() if _is_runtime_parameter_annotation(v)}
            tensor_args_defaults = {
                k: sig.parameters[k].default for k in tensor_args if sig.parameters[k].default is not sig.parameters[k].empty
            }
            return JITFunc(func, sig, arg_names, tensor_args, tensor_args_defaults, ir_gen, private=private)
        else:
            try:
                builder = Builder()
                with builder.prim_func(func.__name__, private=private):
                    ir_gen.gen(builder)(**annot)
                prim_func = builder.get()
                prim_func = _patch_prim_func_attrs(prim_func, builder)
                prim_func.orig_func = func
                return prim_func
            except Exception as e:
                logger.fatal(f"Failed to build prim_func from {func.__name__}\nargs={annot}\nsource={ir_gen.source}")
                raise e

    return impl(func) if func is not None else impl
