from __future__ import annotations
import threading
from typing import Any

# Use thread local to store the stack
# This is to avoid the cross-thread interference
_local = threading.local()


class CaptureStack:
    """
    A simple stack implementation for capturing items in a thread-local context.
    Used to manage a stack of items (e.g., input tensors) for auto-tuning capture.
    """

    def __init__(self):
        # Initialize an empty list to use as the stack
        self.stack = []

    def push(self, item):
        """
        Push an item onto the top of the stack.

        Args:
            item: The item to be pushed onto the stack.
        """
        self.stack.append(item)

    def pop(self):
        """
        Pop and return the top item from the stack.

        Returns:
            The item at the top of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        return self.stack.pop()

    def top(self):
        """
        Return the item at the top of the stack without removing it.

        Returns:
            The item at the top of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        return self.stack[-1]

    def size(self):
        """
        Return the number of items in the stack.

        Returns:
            int: The size of the stack.
        """
        return len(self.stack)

    def __len__(self):
        """
        Return the number of items in the stack (len operator support).

        Returns:
            int: The size of the stack.
        """
        return len(self.stack)

    def __bool__(self):
        """
        Return True if the stack is not empty, False otherwise.

        Returns:
            bool: Whether the stack contains any items.
        """
        return bool(self.stack)


def _get_current_stack() -> CaptureStack:
    if not hasattr(_local, "capture_stack"):
        _local.capture_stack = CaptureStack()
    return _local.capture_stack


class AutotuneInputsCapture:
    __slots__ = "tensors"

    def __init__(self, tensors: list[Any]):
        self.tensors = tensors

    def __enter__(self) -> None:
        _get_current_stack().push(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _get_current_stack().pop()


def set_autotune_inputs(*args) -> AutotuneInputsCapture:
    """Set input tensors for auto-tuning.

    This function creates a context manager for capturing input tensors
    during the auto-tuning process. It supports both:
        set_autotune_inputs(a, b, c)
        set_autotune_inputs([a, b, c])

    Args:
        *args: Either a single list/tuple of tensors, or multiple tensor arguments.

    Returns:
        AutotuneInputsCapture: A context manager for auto-tuning inputs.
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        tensors = list(args[0])
    else:
        tensors = list(args)
    return AutotuneInputsCapture(tensors)


def get_autotune_inputs() -> list[Any] | None:
    """
    Get the current autotune inputs from the stack.
    """
    stack = _get_current_stack()
    return stack.top().tensors if stack else None
