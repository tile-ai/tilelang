"""Kernel launching language interface in TileLang."""

from __future__ import annotations
from collections import deque
import os
from typing import Any
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


def _normalize_threads(
    threads: int | list[int] | tuple | None,
) -> list[int] | None:
    """Normalize a thread-block specification into a 3-D extent list.

    Args:
        threads: A thread count, a per-dimension extent list/tuple, or None to
            leave the choice to the backend.

    Returns:
        The extents as ``[x, y, z]``, padding missing dimensions with 1, or
        None when no thread count was requested. The frontend does not pick a
        default: the thread count is a SIMT launch hint whose default (if any)
        belongs to the backend that materializes the launch.

    Raises:
        ValueError: If ``threads`` has an unsupported type, or any concrete extent
            is not positive.
    """
    if threads is None:
        return None

    if isinstance(threads, int):
        normalized = [threads, 1, 1]
    elif isinstance(threads, list):
        normalized = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        normalized = list(threads) + [1] * (3 - len(threads))
    else:
        raise ValueError("threads must be an integer or a list of integers")

    # A block extent must be positive. A non-positive value would otherwise reach
    # codegen and launch a kernel with an empty thread block, silently writing nothing.
    if any(isinstance(extent, int) and extent <= 0 for extent in normalized):
        raise ValueError(f"threads must be positive, got {threads}")

    return normalized


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

    Grid (program index) vars are bound by the frame itself. Thread vars are
    placeholders: they have an identity so the body can reference them, but
    their extent is only known once a backend materializes the launch. Thread
    extents are therefore available at trace time only when ``threads=`` was
    passed to :func:`Kernel`.
    """

    def __enter__(self) -> Var | list[Var]:
        """
        Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
        Returns one Var for a single grid dimension, or a list of Vars otherwise.
        """
        super().__enter__()
        _get_current_stack().push(self)

        last_block_frame = self.frames[-1]
        assert isinstance(last_block_frame, SBlockFrame), f"Last frame must be a block frame, got {last_block_frame}"

        return _normalize_bindings(list(self.grid_vars))

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
        Grid axes that were not launched have extent 1.
        """
        if dim >= len(self.grid_extents):
            return 1
        return int(self.grid_extents[dim])

    def get_block_extents(self) -> list[int]:
        """
        Returns the block extents for all three dimensions.
        """
        return [self.get_block_extent(dim) for dim in range(3)]

    def get_thread_extent(self, dim: int) -> int:
        """
        Returns the thread extent for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.

        Raises:
            ValueError: If the kernel was launched without ``threads=``. The
                extent is then chosen by the backend and is not known at trace time.
        """
        if self.thread_extents is None:
            raise ValueError(
                "The thread extent is not known at trace time: T.Kernel(...) was called without "
                "threads=. Pass threads= explicitly when the kernel body needs the thread-block size."
            )
        return int(self.thread_extents[dim])

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
        return self.thread_vars[dim]

    def get_thread_bindings(self) -> list[Var]:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return list(self.thread_vars)

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
        return self.grid_vars[dim]

    def get_block_bindings(self) -> list[Var]:
        """
        Returns all three block bindings.
        """
        return list(self.grid_vars)

    def get_launch_annotation(self, key: str, default=None):
        """
        Returns the launch annotation ``key`` recorded by T.Kernel (e.g. ``cluster_dims``),
        or ``default`` when it was not given.
        """
        annotations = self.frames[-1].annotations
        if annotations is None or key not in annotations:
            return default
        return annotations[key]

    def get_cluster_dims(self) -> list[int]:
        """
        Returns the cluster dimensions as ``[x, y, z]``. A launch without
        ``cluster_dims`` has clusters of a single program, i.e. ``[1, 1, 1]``.
        """
        dims = self.get_launch_annotation("cluster_dims")
        if dims is None:
            return [1, 1, 1]
        dims = [int(d) for d in dims]
        return dims + [1] * (3 - len(dims))

    def get_cluster_size(self) -> int:
        """
        Returns the number of programs per cluster (product of the cluster dimensions).
        """
        size = 1
        for dim in self.get_cluster_dims():
            size *= dim
        return size

    def get_cluster_id(self, dim: int = 0) -> Var | tirx.PrimExpr:
        """
        Returns the index of the cluster the current program belongs to along
        ``dim``, in program-space arithmetic: ``block_id // cluster_dims[dim]``.

        A cluster is a ``cluster_dims``-shaped tile of the grid, so this is the
        same on every target (clusterIdx on CUDA, a group of consecutive
        programs elsewhere) and stays consistent with threadblock swizzling,
        which permutes the grid at cluster granularity.
        """
        if dim >= len(self.grid_vars):
            return tirx.IntImm("int32", 0)
        block = self.grid_vars[dim]
        cluster_dim = self.get_cluster_dims()[dim]
        if cluster_dim == 1:
            return block
        return tirx.floordiv(block, tirx.IntImm(block.dtype, cluster_dim))

    def get_cluster_ids(self) -> list[Var | tirx.PrimExpr]:
        """
        Returns the cluster index along every launched grid axis.
        """
        return [self.get_cluster_id(dim) for dim in range(len(self.grid_vars))]

    def get_cluster_extent(self, dim: int = 0) -> int:
        """
        Returns the number of clusters along ``dim``: ``ceil(grid_extent / cluster_dims[dim])``.
        """
        cluster_dim = self.get_cluster_dims()[dim]
        return -(-self.get_block_extent(dim) // cluster_dim)

    def get_cluster_extents(self) -> list[int]:
        """
        Returns the number of clusters along all three dimensions.
        """
        return [self.get_cluster_extent(dim) for dim in range(3)]

    @property
    def blocks(self) -> list[Var]:
        """
        Returns the block indices from the topmost frame.
        """
        return list(self.grid_vars)

    @property
    def threads(self) -> list[Var]:
        """
        Returns the thread indices from the topmost frame.
        """
        return list(self.thread_vars)

    @property
    def num_threads(self) -> int:
        """
        Returns the total number of threads.
        """
        return self.get_num_threads()


# ---------------------------------------------------------------------------
# Launch annotations
#
# T.Kernel(*grid) is the launch every target shares. Everything a backend may
# additionally need (thread count, clusters, ...) is a *launch annotation*: the
# frontend records it verbatim and the backend interprets it once the target
# is known (MaterializeKernelLaunch). Which annotations exist is declared per
# language dialect as the explicit keyword parameters of its own `Kernel`
# (see tilelang/<backend>/language/kernel.py), so `tilelang.cuda.language.Kernel`
# shows, autocompletes and accepts exactly what CUDA understands. Every
# dialect's `Kernel` funnels into `launch_kernel` below.
# ---------------------------------------------------------------------------


_KERNEL_LAUNCH_FACTORY_ATTR = "__tilelang_kernel_launch__"


def kernel_launch_factory(func):
    """Mark ``func`` as a launch factory: a callable used as ``with func(...)``
    to open a kernel launch. Every dialect's ``Kernel`` (and ``ClusterKernel``)
    carries this mark so the eager JIT rewriter can find the launch regardless
    of which dialect or alias the user went through."""
    setattr(func, _KERNEL_LAUNCH_FACTORY_ATTR, True)
    return func


def is_kernel_launch_factory(obj) -> bool:
    return getattr(obj, _KERNEL_LAUNCH_FACTORY_ATTR, False) is True


def launch_kernel(
    blocks: tuple[int | tirx.PrimExpr, ...],
    *,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
    **annotations: Any,
) -> KernelLaunchFrame:
    """Shared implementation behind every dialect's ``T.Kernel``.

    The well-known launch annotations are normalized here; any other keyword
    a dialect forwards is recorded verbatim on the launch block for that
    backend's pipeline to consume. Dialects, not this function, decide which
    keywords exist: they only forward what their own ``Kernel`` signature
    declares.
    """
    # In eager mode, we construct AST directly without prim_func,
    # so there must be a Builder available. If not, this function
    # is being called outside of a JIT/prim_func context.
    # lazy import to avoid circular import
    from tilelang.language.eager.builder import Builder

    if Builder.current() is None:
        raise JITNoBuilderError("T.Kernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.")

    attrs: dict = {}
    if prelude is not None:
        attrs["pragma_import_c"] = prelude
    cluster_dims = _normalize_cluster_dims(cluster_dims)
    if cluster_dims is not None:
        attrs["cluster_dims"] = cluster_dims
    for key, value in annotations.items():
        if value is not None:
            attrs[key] = value

    return _ffi_api.KernelLaunch(blocks, _normalize_threads(threads), attrs)


@kernel_launch_factory
def Kernel(*blocks: int | tirx.PrimExpr) -> KernelLaunchFrame:
    """Construct a kernel launch frame: a grid of tile programs.

    This is the target-neutral launch: the part every backend shares. Backend
    dialects offer their own ``T.Kernel`` with the launch annotations that
    backend understands, e.g. ``tilelang.cuda.language.Kernel(..., threads=128)``;
    ``tilelang.language`` is the CUDA dialect.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Extent of the grid along each axis (1-3 dimensions). The launch yields
        one program index per axis (``blockIdx`` on CUDA, the outer loop on
        CPU, the core index on an NPU).

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128)) as bx:
            # bx is the program index along x; also iterable as (bx,)
            ...
    """
    return launch_kernel(blocks)


@kernel_launch_factory
def ClusterKernel(
    *blocks: int | tirx.PrimExpr,
    cluster_dims: int | tuple[int, int, int] | list[int],
    threads: int | list[int] | tuple | None = None,
    prelude: str | None = None,
):
    """Construct a kernel launch frame with a CUDA thread block cluster
    (SM90+ only).

    This is the CUDA-specific variant of :func:`Kernel`: identical launch
    semantics and bindings, plus a ``cluster_dims`` annotation. The kernel
    will be launched with cudaLaunchKernelEx using
    cudaLaunchAttributeClusterDimension.

    Parameters
    ----------
    blocks : int
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    cluster_dims : int | tuple[int, int, int] | list[int]
        The cluster dimensions. For example, use 2 or (2, 1, 1) to create
        2-CTA clusters.
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.

    Examples
    --------
    .. code-block:: python

        with T.ClusterKernel(grid_x, grid_y, cluster_dims=2, threads=128) as (bx, by):
            ...
    """
    return launch_kernel(blocks, threads=threads, prelude=prelude, cluster_dims=cluster_dims)


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

    threads = _normalize_threads(threads)

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


def get_cluster_dims() -> list[int]:
    """Returns the cluster dimensions ``[x, y, z]`` of the current launch (``[1, 1, 1]`` without clusters)."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_cluster_dims()


def get_cluster_size() -> int:
    """Returns the number of programs per cluster of the current launch."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_cluster_size()


def get_cluster_id(dim: int = 0) -> Var | tirx.PrimExpr:
    """Returns the cluster index of the current program along ``dim``
    (``block_id // cluster_dims[dim]``). See :meth:`KernelLaunchFrame.get_cluster_id`."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_cluster_id(dim)


def get_cluster_ids() -> list[Var | tirx.PrimExpr]:
    """Returns the cluster index along every launched grid axis."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_cluster_ids()


def get_cluster_extent(dim: int = 0) -> int:
    """Returns the number of clusters along ``dim``."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_cluster_extent(dim)


def get_cluster_extents() -> list[int]:
    """Returns the number of clusters along all three dimensions."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_cluster_extents()
