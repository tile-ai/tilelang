"""Override the LetFrame to print a message when entering the frame."""

from __future__ import annotations
from tvm.ffi import register_object as _register_object
from tvm.tir import Var, PrimExpr, BufferLoad, BufferRegion
from tvm.ir import Range
from tvm import DataType
from tvm.script.ir_builder.tir.frame import TIRFrame
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


@_register_object("script.ir_builder.tir.LetFrame")
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
        if isinstance(self.value, BufferLoad):
            indices = self.value.indices
            is_block_load = False
            for index in indices[:-1]:
                if DataType(index.dtype).lanes > 1:
                    is_block_load = True
                    break
            if is_block_load:
                self.value = BufferRegion(self.value.buffer, [Range(x.base, x.lanes) for x in indices])

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


def get_let_value(var: Var) -> PrimExpr | None:
    """Get the value bound to a variable in the current let frame stack.

    Args:
        var (Var): The variable to look up

    Returns:
        Optional[PrimExpr]: The bound value if found, None otherwise
    """
    return _get_let_stack().get_value(var)
