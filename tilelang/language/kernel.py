"""Kernel launching language interface in TileLang."""

from __future__ import annotations
from collections import deque
from tvm import tir
from tvm.tir import Var
from tvm.script.ir_builder.tir.frame import TIRFrame, BlockFrame
from tvm.ffi import register_object
from tilelang import _ffi_api
import threading

# Ensure single-dimension kernel bindings can be unpacked like iterables.
# especially for issue https://github.com/tile-ai/tilelang/issues/830
if not hasattr(Var, "__iter__"):

    def _var_iter(self):
        yield self

    Var.__iter__ = _var_iter  # type: ignore[attr-defined]

if not hasattr(Var, "__len__"):
    Var.__len__ = lambda self: 1  # type: ignore[attr-defined]


class FrameStack:
    """
    A simple stack-like wrapper around a deque that provides
    push, pop, and top methods for convenience.
    """

    def __init__(self):
        self._stack = deque()

    def push(self, item):
        """Pushes an item onto the top of the stack."""
        self._stack.append(item)

    def pop(self):
        """
        Pops and returns the top of the stack, or returns None
        if the stack is empty.
        """
        if self._stack:
            return self._stack.pop()
        raise IndexError(f"{self.__class__.__name__} is empty")

    def top(self):
        """
        Returns the item on the top of the stack without removing it,
        or None if the stack is empty.
        """
        if self._stack:
            return self._stack[-1]
        raise IndexError(f"{self.__class__.__name__} is empty")

    def size(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

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
_local = threading.local()


def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "kernel_launch_frame_stack"):
        _local.kernel_launch_frame_stack = FrameStack()
    return _local.kernel_launch_frame_stack


def _normalize_bindings(bindings: list[Var]) -> Var | list[Var]:
    """
    Return a bare Var when we only have a single binding so that users may write either
    `with T.Kernel(...) as pid:` or `with T.Kernel(...) as (pid,)`.
    Otherwise, keep the list semantics for multi-dimensional launches.
    """
    if len(bindings) == 1:
        return bindings[0]
    return bindings


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    """
    KernelLaunchFrame is a custom TIRFrame that manages block/thread indices
    and handles the entry and exit of the kernel launch scope.
    """

    def __enter__(self) -> Var | list[Var]:
        """
        Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
        Returns one Var if we detect exactly 5 frames (meaning there is a single
        block dimension), or a list of Vars otherwise.
        """
        super().__enter__()
        _get_current_stack().push(self)

        last_block_frame = self.frames[-1]
        assert isinstance(last_block_frame, BlockFrame), f"Last frame must be a block frame, got {last_block_frame}"

        maybe_cpu = last_block_frame.annotations.get("tilelang.is_cpu_kernel_frame", False)

        if maybe_cpu:
            # CPU kernel frame, return a list of for frame items.
            return _normalize_bindings([frame.vars[0] for frame in self.frames[0:-1]])
        else:
            # Otherwise, return a list of iter_var.var objects (excluding the last 4 frames).
            # As 4 frames for threadIdx.x, threadIdx.y, threadIdx.z and block frame with attributes
            return _normalize_bindings([frame.iter_var.var for frame in self.frames[0:-4]])

    def __exit__(self, ptype, value, trace):
        """
        Exits the KernelLaunchFrame scope and pops this frame from the stack,
        but only if it's indeed the topmost frame.
        """
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> KernelLaunchFrame | None:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def get_block_extent(self, dim: int) -> int:
        """
        Returns the block extent for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        iter_var = self.frames[dim].iter_var
        return int(iter_var.dom.extent)

    def get_block_extents(self) -> list[int]:
        """
        Returns the block extents for all three dimensions.
        """
        return [self.get_block_extent(dim) for dim in range(3)]

    def get_thread_extent(self, dim: int) -> int:
        """
        Returns the thread extent for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        iter_var = self.frames[-4 + dim].iter_var
        return int(iter_var.dom.extent)

    def get_thread_extents(self) -> list[int]:
        """
        Returns the thread extents for all three dimensions.
        """
        return [self.get_thread_extent(dim) for dim in range(3)]

    def get_thread_binding(self, dim: int = 0) -> Var:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return self.frames[-4 + dim].iter_var.var

    def get_thread_bindings(self) -> list[Var]:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return [frame.iter_var.var for frame in self.frames[-4:-1]]

    def get_num_threads(self) -> int:
        """
        Returns the thread indices from the topmost frame.
        """
        num_threads: int = 1
        for thread_dim in range(3):
            num_threads *= self.get_thread_extent(thread_dim)
        return num_threads

    def get_block_binding(self, dim: int = 0) -> Var:
        """
        Returns the block binding for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        return self.frames[dim].iter_var.var

    def get_block_bindings(self) -> list[Var]:
        """
        Returns all three block bindings.
        """
        return [frame.iter_var.var for frame in self.frames[0:-4]]

    @property
    def blocks(self) -> list[Var]:
        """
        Returns the block indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[0:-4]]

    @property
    def threads(self) -> list[Var]:
        """
        Returns the thread indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[-4:]]

    @property
    def num_threads(self) -> int:
        """
        Returns the total number of threads.
        """
        return self.get_num_threads()


def Kernel(
    *blocks: tir.PrimExpr,
    threads: int | list[int] | tuple | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
    """
    Construct a kernel launch frame for TileLang and return a launch-frame descriptor usable as a context manager.
    
    Parameters:
        blocks: One to three tir.PrimExpr values specifying grid dimensions for blockIdx.(x|y|z).
        threads: An int, list, or tuple specifying thread block dimensions. When not provided and not a CPU kernel, defaults to 128 for the first dimension and 1 for missing dimensions; values are normalized to a length-three list [x, y, z].
        is_cpu: If True, mark the frame as a CPU kernel so thread and block thread-bindings are not created.
        prelude: Optional C source to be injected before the generated kernel code via pragma_import_c.
    
    Returns:
        A KernelLaunchFrame descriptor (FFI handle) that can be used as a context manager; entering the context yields the block binding Var for a single-dimension grid or a tuple/list of Vars for multiple dimensions.
    """
    attrs: dict = {}

    if not is_cpu and threads is None:
        threads = 128  # default thread number

    if isinstance(threads, int):
        threads = [threads, 1, 1]
    elif isinstance(threads, list):
        threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        threads = list(threads) + [1] * (3 - len(threads))
    else:
        assert is_cpu, "threads must be an integer or a list of integers"

    if is_cpu:
        attrs["tilelang.is_cpu_kernel_frame"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.KernelLaunch(blocks, threads, attrs)


def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_binding(dim)


def get_thread_bindings() -> list[Var]:
    """Returns all three thread bindings."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_bindings()


def get_block_binding(dim: int = 0) -> Var:
    """Returns the block binding for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_binding(dim)


def get_block_bindings() -> list[Var]:
    """Returns all three block bindings."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_bindings()


def get_thread_extent(dim: int = 0) -> int:
    """Returns the thread extent for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extent(dim)


def get_thread_extents() -> list[int]:
    """Returns all three thread extents."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extents()


def get_block_extent(dim: int = 0) -> int:
    """Returns the block extent for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_extent(dim)


def get_block_extents() -> list[int]:
    """Returns all three block extents."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_extents()