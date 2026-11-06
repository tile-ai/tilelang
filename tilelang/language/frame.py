"""Loop frames and tracked immutable bindings used by the eager DSL builder."""

from __future__ import annotations
import builtins
from contextlib import ExitStack
import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.ir import prim as _prim
from tvm_tirx.ir import Var, Expr, TensorLoad, TensorRegion
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as _builder

from ._construction import as_expr, bind, current_location, dtype_of
from ._annotation_codec import attr
from tvm_tirx.ir import Range
from tvm_ffi import dtype as DataType
from tvm_tirx.script.ir_builder.frame import StmtFrame as TIRFrame
from collections import deque
import threading


class FrameStack:
    """A stack-like container for managing TIR frame objects and their variable bindings.

    This class implements a stack data structure using a deque and maintains a mapping
    of variables to their values. It provides methods for stack operations and variable
    value lookups.
    """

    def __init__(self):
        """Initialize an empty frame stack and variable mapping."""
        self._stack = deque()
        self._var_value_map = {}

    def push(self, item):
        """Push an item onto the stack and update variable mapping if applicable.

        Args:
            item: The frame object to push onto the stack
        """
        self._stack.append(item)
        if hasattr(item, "var") and hasattr(item, "value"):
            self._var_value_map[item.var] = item.value

    def pop(self):
        """Remove and return the top item from the stack.

        Returns:
            The top frame object from the stack

        Raises:
            IndexError: If the stack is empty
        """
        if self._stack:
            item = self._stack.pop()
            if hasattr(item, "var"):
                self._var_value_map.pop(item.var, None)
            return item
        raise IndexError(f"{self.__class__.__name__} is empty")

    def get_value(self, var):
        """Retrieve the value associated with a variable.

        Args:
            var: The variable to look up

        Returns:
            The value associated with the variable, or None if not found
        """
        return self._var_value_map.get(var)

    def set_value(self, var, value):
        """Record a variable binding that is not represented by a scoped frame."""
        self._var_value_map[var] = value

    def clear_values(self):
        """Clear all tracked variable bindings."""
        self._var_value_map.clear()

    def has_value(self, var):
        """Check if a variable has an associated value.

        Args:
            var: The variable to check

        Returns:
            bool: True if the variable has an associated value, False otherwise
        """
        return var in self._var_value_map

    def top(self):
        """Return the top item of the stack without removing it.

        Returns:
            The top frame object from the stack

        Raises:
            IndexError: If the stack is empty
        """
        if self._stack:
            return self._stack[-1]
        raise IndexError(f"{self.__class__.__name__} is empty")

    def __len__(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __bool__(self):
        """
        Allows truthy checks on the stack object itself,
        e.g., 'if stack: ...'
        """
        return bool(self._stack)


# Use thread local to store the stack
# This is to avoid the cross-thread interference
_local_let = threading.local()


def _get_let_stack() -> FrameStack:
    if not hasattr(_local_let, "let_frame_stack"):
        _local_let.let_frame_stack = FrameStack()
    return _local_let.let_frame_stack


def register_let_value(var: Var, value: Expr | TensorRegion):
    """Record the value behind a flat tirx Bind.

    The legacy LetFrame path updated FrameStack through __enter__/__exit__.
    tirx emits flat Bind statements instead, so callers that need to recover
    TensorRegion aliases must explicitly register the Python-side mapping.
    """
    _get_let_stack().set_value(var, value)


def clear_let_values():
    """Clear all Python-side bindings recorded for the current thread."""
    _get_let_stack().clear_values()


class LetFrame(TIRFrame):
    """A TIR frame for let bindings that manages variable scope and value tracking.

    This frame type extends TIRFrame to provide variable binding functionality and
    maintains a global stack of active bindings.
    """

    def __enter__(self) -> Var:
        """Enter the let frame scope and process buffer loads.

        Returns:
            Var: The variable bound in this frame
        """
        super().__enter__()
        if isinstance(self.value, TensorLoad):
            indices = self.value.indices
            is_block_load = False
            for index in indices[:-1]:
                if DataType(dtype_of(index)).lanes > 1:
                    is_block_load = True
                    break
            if is_block_load:
                self.value = TensorRegion(
                    self.value.source, [Range(x.base, x.lanes) for x in indices], ir.TensorRegionType(), current_location()
                )

        _get_let_stack().push(self)
        return self.var

    def __exit__(self, ptype, value, trace):
        """Exit the let frame scope and clean up the stack.

        Args:
            ptype: Exception type if an exception occurred
            value: Exception value if an exception occurred
            trace: Exception traceback if an exception occurred
        """
        stack = _get_let_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> LetFrame:
        """Get the current (topmost) let frame.

        Returns:
            LetFrame: The current let frame

        Raises:
            IndexError: If there are no active let frames
        """
        return _get_let_stack().top()

    @staticmethod
    def get_value(var: Var):
        """Get the value bound to a variable in any active frame.

        Args:
            var (Var): The variable to look up

        Returns:
            The value bound to the variable, or None if not found
        """
        return _get_let_stack().get_value(var)

    @staticmethod
    def has_value(var: Var) -> bool:
        """Check if a variable has a binding in any active frame.

        Args:
            var (Var): The variable to check

        Returns:
            bool: True if the variable has a binding, False otherwise
        """
        return _get_let_stack().has_value(var)


def has_let_value(var: Var) -> bool:
    """Check if a variable has a binding in the current let frame stack.

    Args:
        var (Var): The variable to check

    Returns:
        bool: True if the variable has a binding, False otherwise
    """
    return _get_let_stack().has_value(var)


def get_let_value(var: Var) -> Expr | None:
    """Get the value bound to a variable in the current let frame stack.

    Args:
        var (Var): The variable to look up

    Returns:
        Optional[Expr]: The bound value if found, None otherwise
    """
    return _get_let_stack().get_value(var)


def _has_expr_annotation(value):
    if isinstance(value, ir.Expr):
        return True
    if isinstance(value, dict | tvm_ffi.Map):
        return any(_has_expr_annotation(key) or _has_expr_annotation(item) for key, item in value.items())
    if isinstance(value, list | tuple | tvm_ffi.Array):
        return any(_has_expr_annotation(item) for item in value)
    return not isinstance(value, (str, builtins.int, builtins.float, builtins.bool, type(None)))


def _normalize_pipeline_annotations(annotations):
    """Store compile-time scheduling integers as native host attributes.

    Source programs commonly spell these integers with ``T.int32``. Their
    type is irrelevant to the scheduling hint, and putting them in an operand
    region hides the stage/order fields from ordinary loop consumers.
    """

    def integer(value):
        if isinstance(value, ir.IntImm):
            return builtins.int(value.value)
        if isinstance(value, builtins.int):
            return value
        if isinstance(value, (tuple, list, tvm_ffi.Array)):
            return [integer(item) for item in value]
        raise ValueError("Pipeline annotations require compile-time integers")

    return {
        key: integer(value)
        if key.startswith("software_pipeline_")
        or key in ("num_stages", "tl_pipelined_num_stages", "tl_pipeline_stage", "tl_pipeline_order")
        else value
        for key, value in (annotations or {}).items()
    }


class LoopFrame:
    """Own a loop nest and optional typed metadata in the native frame stack."""

    def __init__(self, frames, annotations=None):
        self.frames = list(frames)
        self.vars = [var for current in self.frames for var in getattr(current, "vars", getattr(current, "body_params", ()))]
        self.annotations = annotations
        self._stack = None

    @property
    def stmts(self):
        return self.frames[-1].stmts

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.__enter__()
        try:
            if self.annotations:
                self._stack.enter_context(attr(None, "tl.loop.annotations", self.annotations))
            for current in self.frames:
                self._stack.enter_context(current)
            return self.vars[0] if len(self.vars) == 1 else self.vars
        except BaseException:
            import sys

            self._stack.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class PersistentLoopFrame(LoopFrame):
    """Expand grouped persistent work into native loops, binds and tail guards.

    Coordinate binds are inside the wave loop, where their induction variable
    is defined. The public coordinates keep their domain dtypes; the generated
    condition guards both an incomplete group and an incomplete worker wave.
    """

    def __init__(self, domain, wave_size, index, group_size=8, num_stages=0, annotations=None):
        if not domain:
            raise ValueError("T.Persistent requires a nonempty domain")
        self.domain = [as_expr(extent) for extent in domain]
        self.wave_size = as_expr(wave_size)
        self.index = as_expr(index)
        if isinstance(self.wave_size, ir.IntImm) and self.wave_size.value <= 0:
            raise ValueError("T.Persistent wave_size must be positive")
        if any(isinstance(extent, ir.IntImm) and extent.value < 0 for extent in self.domain):
            raise ValueError("T.Persistent domain extents must be nonnegative")
        last = self.domain[-1]
        self.group_size = _prim.max(1, _prim.min(as_expr(8 if group_size is None else group_size), last))
        self.grouped_domain = [_prim.ceildiv(last, self.group_size), *self.domain[:-1], self.group_size]
        self.padded_size = self.grouped_domain[0]
        for extent in self.grouped_domain[1:]:
            self.padded_size = self.padded_size * extent
        waves = _prim.ceildiv(self.padded_size, self.wave_size)
        annotations = dict(annotations or {})
        if num_stages > 0:
            annotations["num_stages"] = num_stages
        annotations = _normalize_pipeline_annotations(annotations)
        dynamic = _has_expr_annotation(annotations)
        current = _builder.serial(waves, annotations={} if dynamic else annotations)
        super().__init__([current], annotations if dynamic else None)

    def __enter__(self):
        wave = super().__enter__()
        try:
            linear = wave * self.wave_size + self.index
            remainder = linear
            indices = [None] * len(self.grouped_domain)
            for axis in range(len(self.grouped_domain) - 1, 0, -1):
                # Empty dimensions produce zero waves. Keep the unused decode
                # well-formed as well, instead of constructing division by zero.
                extent = _prim.max(1, self.grouped_domain[axis])
                indices[axis] = _prim.truncmod(remainder, extent)
                remainder = _prim.truncdiv(remainder, extent)
            last = remainder * self.group_size + indices[-1]
            coordinates = [*indices[1:-1], last]
            self.vars = [bind(_prim.Cast(extent.ty.dtype, value)) for extent, value in zip(self.domain, coordinates)]
            condition = _prim.And(linear < self.padded_size, last < self.domain[-1])
            for current in (_builder.if_(condition, loc=current_location()), _builder.then_(loc=current_location())):
                self._stack.enter_context(current)
                self.frames.append(current)
            return self.vars[0] if len(self.vars) == 1 else self.vars
        except BaseException:
            import sys

            self._stack.__exit__(*sys.exc_info())
            raise


class ThreadBindingFrame(LoopFrame):
    """Bind a source loop index to an actual target thread region parameter."""

    def __init__(self, begin, end, thread, annotations=None, variable=None):
        self.begin, self.end = as_expr(begin), as_expr(end)
        self.annotations = annotations
        self.thread = str(thread)
        current = _builder.launch_thread(
            self.thread,
            self.end - self.begin,
            body_params=None if variable is None else [variable],
        )
        self.frames = [current]
        self.vars = list(current.body_params)
        self.doms = [ir.Range(self.begin, self.end)]
        self._stack = None

    def __enter__(self):
        value = super().__enter__()
        if isinstance(self.begin, ir.IntImm) and self.begin.value == 0:
            return value
        return bind(value + self.begin)


def thread_binding(start, stop=None, thread="", *, annotations=None):
    """Retain ``for``/concise thread syntax using ``tirx.launch_thread``."""
    if isinstance(stop, str) and not thread:
        thread, stop = stop, None
    if stop is None:
        start, stop = 0, start
    if not thread:
        raise ValueError("A thread binding requires a nonempty thread tag")
    return ThreadBindingFrame(start, stop, thread, annotations)


def env_thread(thread_tag, dtype="int32"):
    value = ir.Var(thread_tag.replace(".", "_"), ir.PrimType(dtype))
    value._tilelang_thread_tag = str(thread_tag)
    return value


def launch_thread(thread, extent):
    if isinstance(thread, str):
        return ThreadBindingFrame(0, extent, thread)
    tag = getattr(thread, "_tilelang_thread_tag", None)
    if tag is None:
        raise TypeError("launch_thread expects a thread tag or T.env_thread variable")
    return ThreadBindingFrame(0, extent, tag, variable=thread)


def _loop(constructor, start, stop=None, *, step=None, annotations=None):
    if not IRBuilder.is_in_scope():
        from tilelang.jit.exceptions import JITNoBuilderError

        raise JITNoBuilderError("TileLang loops require an active @tilelang.jit, @T.prim_func, or IRBuilder context.")
    annotations = _normalize_pipeline_annotations(annotations)
    dynamic = _has_expr_annotation(annotations)
    frame = constructor(start, stop, step=step, annotations={} if dynamic else annotations)
    return LoopFrame([frame], annotations) if dynamic else frame


def parallel_nest(*extents, annotations=None):
    if not extents:
        raise ValueError("T.Parallel requires at least one extent")
    annotations = _normalize_pipeline_annotations(annotations)
    dynamic = _has_expr_annotation(annotations)
    frames = [
        _builder.parallel(extent, annotations=annotations if index == 0 and not dynamic else {}) for index, extent in enumerate(extents)
    ]
    return LoopFrame(frames, annotations if dynamic else None)


def serial(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.serial, start, stop, step=step, annotations=annotations)


def parallel(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.parallel, start, stop, step=step, annotations=annotations)


def unroll(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.unroll, start, stop, step=step, annotations=annotations)


def vectorized(start, stop=None, *, step=None, annotations=None):
    return _loop(_builder.vectorized, start, stop, step=step, annotations=annotations)
