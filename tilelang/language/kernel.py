"""Kernel launching language interface in TileLang."""

from __future__ import annotations
from collections import deque
import os
from tvm import tirx
from tvm.tirx import Var
from tvm.tirx.script.builder import evaluate as T_evaluate
from tvm.tirx.script.builder.frame import TIRFrame
from tvm.tirx.script.builder.frame import SBlockFrame
from tvm.ffi import register_object
from tilelang import _ffi_api
from tilelang.jit.exceptions import JITNoBuilderError
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


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    if hasattr(value, "value"):
        return int(value.value)
    return int(value)


def _frame_var(frame) -> Var:
    if hasattr(frame, "vars"):
        return frame.vars[0]
    return frame.iter_var.var


def _frame_extent(frame):
    if hasattr(frame, "doms"):
        return frame.doms[0].extent
    return frame.iter_var.dom.extent


def _normalize_threads(
    threads: int | list[int] | tuple | None,
) -> tuple[list[int], bool]:
    is_default = threads is None
    if threads is None:
        threads = 128  # default thread number for targets that have threads

    if isinstance(threads, int):
        return [threads, 1, 1], is_default
    if isinstance(threads, list):
        return threads + [1] * (3 - len(threads)), is_default
    if isinstance(threads, tuple):
        return list(threads) + [1] * (3 - len(threads)), is_default

    raise TypeError("threads must be an integer, a list of integers, a tuple of integers, or None")


def _normalize_cluster_dims(
    cluster_dims: int | tuple[int, int, int] | list[int] | None,
) -> list[int] | None:
    if cluster_dims is None:
        return None

    if isinstance(cluster_dims, (list, tuple)):
        cluster_dims = list(cluster_dims) + [1] * (3 - len(cluster_dims))
    elif isinstance(cluster_dims, int):
        cluster_dims = [cluster_dims, 1, 1]
    else:
        raise ValueError("cluster_dims must be a list or tuple of integers")

    return None if cluster_dims == [1, 1, 1] else cluster_dims


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    """
    KernelLaunchFrame is a custom TIRFrame that manages block/thread indices
    and handles the entry and exit of the kernel launch scope.
    """

    def __enter__(self) -> Var | list[Var]:
        """
        Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
        Returns a block index Var, or a list of block index Vars for
        multi-dimensional launches.
        """
        super().__enter__()
        _get_current_stack().push(self)

        last_block_frame = self.frames[-1]
        assert isinstance(last_block_frame, SBlockFrame), f"Last frame must be a block frame, got {last_block_frame}"

        return _normalize_bindings([_frame_var(frame) for frame in self._block_frames()])

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

    def _num_block_dims(self) -> int:
        annotations = self.frames[-1].annotations
        if "tilelang.kernel_num_blocks" in annotations:
            return _as_int(annotations.get("tilelang.kernel_num_blocks"), 0)
        if annotations.get("tilelang.is_cpu_kernel_frame", False):
            return len(self.frames) - 1
        return max(0, len(self.frames) - 4)

    def _num_thread_dims(self) -> int:
        annotations = self.frames[-1].annotations
        if "tilelang.kernel_num_threads" in annotations:
            return _as_int(annotations.get("tilelang.kernel_num_threads"), 0)
        return max(0, len(self.frames) - self._num_block_dims() - 1)

    def _block_frames(self):
        return self.frames[: self._num_block_dims()]

    def _thread_frames(self):
        start = self._num_block_dims()
        stop = start + self._num_thread_dims()
        return self.frames[start:stop]

    def get_block_extent(self, dim: int) -> int:
        """
        Returns the block extent for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        return int(_frame_extent(self._block_frames()[dim]))

    def get_block_extents(self) -> list[int]:
        """
        Returns the block extents for all three dimensions.
        """
        return [self.get_block_extent(dim) for dim in range(self._num_block_dims())]

    def get_thread_extent(self, dim: int) -> int:
        """
        Returns the thread extent for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return int(_frame_extent(self._thread_frames()[dim]))

    def get_thread_extents(self) -> list[int]:
        """
        Returns the thread extents for all three dimensions.
        """
        return [self.get_thread_extent(dim) for dim in range(self._num_thread_dims())]

    def get_thread_binding(self, dim: int = 0) -> Var:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return _frame_var(self._thread_frames()[dim])

    def get_thread_bindings(self) -> list[Var]:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return [_frame_var(frame) for frame in self._thread_frames()]

    def get_num_threads(self) -> int:
        """
        Returns the thread indices from the topmost frame.
        """
        num_threads: int = 1
        for thread_dim in range(self._num_thread_dims()):
            num_threads *= self.get_thread_extent(thread_dim)
        return num_threads

    def get_block_binding(self, dim: int = 0) -> Var:
        """
        Returns the block binding for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        return _frame_var(self._block_frames()[dim])

    def get_block_bindings(self) -> list[Var]:
        """
        Returns all three block bindings.
        """
        return [_frame_var(frame) for frame in self._block_frames()]

    @property
    def blocks(self) -> list[Var]:
        """
        Returns the block indices from the topmost frame.
        """
        return [_frame_var(frame) for frame in self._block_frames()]

    @property
    def threads(self) -> list[Var]:
        """
        Returns the thread indices from the topmost frame.
        """
        return [_frame_var(frame) for frame in self._thread_frames()]

    @property
    def num_threads(self) -> int:
        """
        Returns the total number of threads.
        """
        return self.get_num_threads()


def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple | None = None,
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
    """Tools to construct a target-neutral kernel launch frame.

    Parameters
    ----------
    blocks : int
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
        When omitted, threaded backends use the default block size while CPU
        backends lower the kernel without synthetic thread loops.
    cluster_dims : int | tuple[int, int, int] | list[int] | None
        The cluster dimensions for SM90+ cluster launch.
        For example, use 2 or (2, 1, 1) to create 2-CTA clusters.
        When specified, the kernel will be launched using cudaLaunchKernelEx
        with cudaLaunchAttributeClusterDimension.
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.

    Examples
    --------
    Create a 1-D CUDA kernel launch and unpack the single block index:

    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            # bx is the blockIdx.x binding (also iterable as (bx,))
            ...

    Launch a 2-D grid while requesting two thread dimensions:

    .. code-block:: python

        with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
            tx, ty = T.get_thread_bindings()
            ...

    The same source can be lowered for CPU or GPU targets. The target-specific
    loop form is selected during compilation:

    .. code-block:: python

        with T.Kernel(loop_extent) as i:
            ...
    """
    # In eager mode, we construct AST directly without prim_func,
    # so there must be a Builder available. If not, this function
    # is being called outside of a JIT/prim_func context.
    # lazy import to avoid circular import
    from tilelang.language.eager.builder import Builder

    if Builder.current() is None:
        raise JITNoBuilderError("T.Kernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.")

    attrs: dict = {}
    threads, threads_defaulted = _normalize_threads(threads)
    if threads_defaulted:
        attrs["tilelang.kernel_thread_default"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    cluster_dims = _normalize_cluster_dims(cluster_dims)
    if cluster_dims is not None:
        attrs["cluster_dims"] = cluster_dims

    return _ffi_api.KernelLaunch(blocks, threads, attrs)


# For CUDA source kernels, we need to load the source code from a file or string.


def _load_cuda_source(source_code_or_path: str | os.PathLike[str]) -> str:
    source = os.fspath(source_code_or_path)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source_code_or_path must be a non-empty source string or source path")

    expanded = os.path.expanduser(source)
    if os.path.isfile(expanded):
        with open(expanded, encoding="utf-8") as f:
            return f.read()

    source_markers = ("\n", "__global__", 'extern "C"', "#include")
    if any(marker in source for marker in source_markers):
        return source

    contains_path_sep = os.path.sep in source or (os.path.altsep is not None and os.path.altsep in source)
    if contains_path_sep or source.endswith((".cu", ".cuh", ".cuda", ".cpp", ".cc", ".c")):
        raise FileNotFoundError(f"CUDA source file not found: {source}")

    return source


def CUDASourceCodeKernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple | None = None,
    source_code_or_path: str | os.PathLike[str],
    entry_name: str = "main_kernel",
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
    prelude: str | None = None,
) -> None:
    """Launch a kernel from CUDA source code or a CUDA source file.

    The code must follows the following rules:
    1. The kernel source must be a valid CUDA kernel which can be correctly compiled under TileLang's context.
    2. The kernel source must either contains only one `__global__` function as an entry, or have a `__global__` entry function named `main_kernel`.

    Parameters
    ----------
    source_code_or_path : str | os.PathLike[str]
        Inline CUDA source code, or a path to a CUDA source file.
        If the argument resolves to an existing file, the file contents are
        loaded. Otherwise it is treated as inline CUDA source code.
    blocks : int
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    entry_name : str | None
        Optional name of the `__global__` CUDA entry function inside the
        provided source. When specified, TileLang launches that external CUDA
        entry directly.
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
    cluster_dims : int | tuple[int, int, int] | list[int] | None
        The cluster dimensions for SM90+ cluster launch.
        For example, use 2 or (2, 1, 1) to create 2-CTA clusters.
        When specified, the kernel will be launched using cudaLaunchKernelEx
        with cudaLaunchAttributeClusterDimension.
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.
    """
    from tilelang.language.eager.builder import Builder

    if Builder.current() is None:
        raise JITNoBuilderError(
            "T.CUDASourceCodeKernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available."
        )

    source = _load_cuda_source(source_code_or_path)
    if prelude is not None:
        source = prelude + "\n" + source

    attrs: dict = {"code_block_source": source}
    if not isinstance(entry_name, str) or not entry_name.strip():
        raise ValueError("entry_name must be a non-empty string when provided")
    attrs["code_block_entry_name"] = entry_name

    threads, threads_defaulted = _normalize_threads(threads)
    if threads_defaulted:
        attrs["tilelang.kernel_thread_default"] = True

    cluster_dims = _normalize_cluster_dims(cluster_dims)
    if cluster_dims is not None:
        attrs["cluster_dims"] = cluster_dims

    with _ffi_api.KernelLaunch(blocks, threads, attrs):
        # Keep the launch frame alive until SplitHostDevice can lift the
        # external CUDA source pragma onto the device PrimFunc.
        T_evaluate(tirx.call_extern("int32", entry_name))


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
