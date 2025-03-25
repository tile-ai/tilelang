# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Override the LetFrame to print a message when entering the frame."""

from tvm._ffi import register_object as _register_object
from tvm.tir import Var, PrimExpr, BufferLoad, BufferRegion
from tvm.ir import Range
from tvm import DataType
from tvm.script.ir_builder.tir.frame import TIRFrame
from collections import deque
from typing import Optional


class FrameStack:
    """
    A stack-like wrapper around a deque that provides push, pop, and top methods,
    along with a var-value mapping functionality.
    """

    def __init__(self):
        self._stack = deque()
        self._var_value_map = {}

    def push(self, item):
        """Pushes an item onto the top of the stack."""
        self._stack.append(item)
        # Store the var-value mapping if it's a LetFrame
        if hasattr(item, 'var') and hasattr(item, 'value'):
            self._var_value_map[item.var] = item.value

    def pop(self):
        """
        Pops and returns the top of the stack, or returns None
        if the stack is empty.
        """
        if self._stack:
            item = self._stack.pop()
            # Clean up the var-value mapping if it's a LetFrame
            if hasattr(item, 'var'):
                self._var_value_map.pop(item.var, None)
            return item
        raise IndexError(f"{self.__class__.__name__} is empty")

    def get_value(self, var):
        """Get the value associated with a variable."""
        return self._var_value_map.get(var)

    def has_value(self, var):
        """Check if a variable has an associated value."""
        return var in self._var_value_map

    def top(self):
        """
        Returns the item on the top of the stack without removing it,
        or None if the stack is empty.
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


# Global stack for LetFrame instances
_let_frame_stack = FrameStack()


@_register_object("script.ir_builder.tir.LetFrame")
class LetFrame(TIRFrame):

    def __enter__(self) -> Var:
        super().__enter__()
        if isinstance(self.value, BufferLoad):
            indices = self.value.indices
            is_block_load = False
            for index in indices[:-1]:
                if DataType(index.dtype).lanes > 1:
                    is_block_load = True
                    break
            if is_block_load:
                self.value = BufferRegion(self.value.buffer,
                                          [Range(x.base, x.lanes) for x in indices])

        _let_frame_stack.push(self)
        return self.var

    def __exit__(self, ptype, value, trace):
        if _let_frame_stack.top() is self:
            _let_frame_stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> "LetFrame":
        """
        Returns the topmost (current) LetFrame from the stack if it exists,
        or raises IndexError if the stack is empty.
        """
        return _let_frame_stack.top()

    @staticmethod
    def get_value(var: Var):
        """
        Get the value associated with a variable.
        Returns None if the variable is not found.
        """
        return _let_frame_stack.get_value(var)

    @staticmethod
    def has_value(var: Var) -> bool:
        """
        Check if a variable has an associated value.
        """
        return _let_frame_stack.has_value(var)


def has_let_value(var: Var) -> bool:
    """
    Check if a variable has an associated value in the let frame stack.
    """
    return _let_frame_stack.has_value(var)


def get_let_value(var: Var) -> Optional[PrimExpr]:
    """
    Get the value associated with a variable from the let frame stack.
    Returns None if the variable is not found.
    """
    return _let_frame_stack.get_value(var)
