"""Ascend NPU dialect of ``T.Kernel``.

Ascend owns its launch surface. The NPU launch is a 1-D grid of AI cores with
no SIMT thread domain at kernel scope, so this dialect's ``Kernel`` declares
``prelude`` and nothing else: passing ``threads=`` or ``cluster_dims=`` is
rejected by Python itself rather than by a runtime probe inside the shared
launch path. Thread domains are declared explicitly *inside* the kernel body by
``T.SimtVF(threads=...)`` (real threadIdx scopes) or ``T.SimdVF()`` (register
level, no threads); both emit their own thread scopes below the launch nest, so
the kernel-level ``tx/ty/tz`` placeholders are dropped by the Ascend pipeline.
"""

from __future__ import annotations

import operator
import threading

from tvm import tirx
from tvm.tirx import Var

from tilelang import _ffi_api
from tilelang.jit.exceptions import JITNoBuilderError
from tilelang.language.kernel import (
    FrameStack,
    KernelLaunchFrame,
    get_block_binding,
    get_block_bindings,
    get_block_extent,
    get_block_extents,
    get_thread_binding as _launch_thread_binding,
    get_thread_bindings as _launch_thread_bindings,
    get_thread_extent as _launch_thread_extent,
    get_thread_extents as _launch_thread_extents,
    kernel_launch_factory,
    launch_kernel,
)

__all__ = [
    "Kernel",
    "MixedKernel",
    "SimtVFContext",
    "get_block_binding",
    "get_block_bindings",
    "get_block_extent",
    "get_block_extents",
    "get_thread_binding",
    "get_thread_bindings",
    "get_thread_extent",
    "get_thread_extents",
    "pop_simtvf_context",
    "push_simtvf_context",
]

# ---------------------------------------------------------------------------
# Nested thread scopes
#
# The Ascend launch owns only the 1-D core grid; real threadIdx domains are
# opened *inside* the kernel body by T.SimtVF. This dialect's thread accessors
# therefore resolve against the innermost SimtVF scope first and only fall
# back to the launch frame. Every accessor that can run under a SimtVF scope
# is dialect-owned (T.get_thread_binding via this module, T.rng_init via
# ascend/language/random.py), so the mechanism lives here rather than in the
# shared launch module.
# ---------------------------------------------------------------------------


class SimtVFContext:
    """Thread vars and extents of the innermost nested thread scope."""

    __slots__ = ("thread_vars", "thread_extents")

    def __init__(self, thread_vars, thread_extents):
        self.thread_vars = thread_vars
        self.thread_extents = thread_extents


_thread_scope_local = threading.local()


def _get_thread_scope_stack() -> FrameStack:
    if not hasattr(_thread_scope_local, "thread_scope_stack"):
        _thread_scope_local.thread_scope_stack = FrameStack()
    return _thread_scope_local.thread_scope_stack


def _get_current_simtvf() -> SimtVFContext | None:
    """The innermost nested thread scope, or None when the launch owns threads."""
    stack = _get_thread_scope_stack()
    return stack.top() if stack else None


def push_simtvf_context(ctx: SimtVFContext):
    """Enter a nested thread scope, making its thread vars the current ones."""
    _get_thread_scope_stack().push(ctx)


def pop_simtvf_context():
    """Leave the innermost nested thread scope."""
    _get_thread_scope_stack().pop()


def _validate_thread_dim(dim: int) -> int:
    """Validate a SimtVF thread-dimension index (0/1/2)."""
    if isinstance(dim, bool):
        raise TypeError(f"thread dimension must be an int, got {type(dim).__name__}")
    try:
        dim = operator.index(dim)
    except TypeError:
        raise TypeError(f"thread dimension must be an int, got {type(dim).__name__}") from None
    if not 0 <= dim <= 2:
        raise ValueError(f"thread dimension must be within [0, 2] (SimtVF declares at most 3 dims), got {dim}")
    return dim


def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension."""
    dim = _validate_thread_dim(dim)
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return simtvf.thread_vars[dim]
    return _launch_thread_binding(dim)


def get_thread_bindings() -> list[Var]:
    """Returns all three thread bindings."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return list(simtvf.thread_vars)
    return _launch_thread_bindings()


def get_thread_extent(dim: int = 0) -> int:
    """Returns the thread extent for the given dimension."""
    dim = _validate_thread_dim(dim)
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return simtvf.thread_extents[dim]
    return _launch_thread_extent(dim)


def get_thread_extents() -> list[int]:
    """Returns all three thread extents."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return list(simtvf.thread_extents)
    return _launch_thread_extents()


# ---------------------------------------------------------------------------
# Launch frames
# ---------------------------------------------------------------------------


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    prelude: str | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for Ascend: a 1-D grid of AI cores.

    The grid becomes the NPU core index (``blockIdx.x``). There is no SIMT
    thread domain at this scope, so this dialect has no ``threads`` parameter:
    ``with T.Kernel(N) as bx`` yields one program index, and ``bx`` is iterable
    as ``(bx,)``. Use ``T.SimtVF(threads=...)`` inside the body to run
    thread-parallel code, or ``T.MixedKernel`` for an AIC+AIV mixed kernel.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Extent of the 1-D core grid. Exactly one dimension is allowed; a
        multi-dimensional launch is rejected here rather than silently
        flattened downstream.
    prelude : str, optional
        AscendC source injected before the generated kernel, e.g. ``#include``
        lines or helper functions.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(NUM_CORES) as bx:
            with T.SimtVF(threads=128):
                for i in T.Parallel(128):
                    out[bx * 128 + i] = x[bx * 128 + i] * 2.0
    """
    if len(blocks) != 1:
        raise ValueError(f"Ascend targets a 1-D core grid: T.Kernel(N) takes exactly one grid extent. Got {len(blocks)}-D: {blocks}.")
    return launch_kernel(blocks, prelude=prelude)


@kernel_launch_factory
def MixedKernel(
    *blocks: int | tirx.PrimExpr,
    sids: int = 2,
    prelude: str | None = None,
):
    """Construct an Ascend NPU Mixed-kernel launch frame with sub-kernel ID binding.

    Parameters
    ----------
    blocks : int
        Number of blocks in the 1-D grid (blockIdx.x extent).
    sids : int
        Number of active AIV sub-cores (1 or 2). Default is 2. On dav-3510,
        mixed kernels always launch the physical ``__mix__(1, 2)`` group;
        ``sids=1`` restricts the vector body to sub-core 0. Binds as ``sid``
        via ``asc_get_sub_block_id()`` in generated code.
    prelude : str, optional
        Import C code injected before the generated kernel.

    Returns
    -------
    res : KernelLaunchFrame
        The resulting frame providing ``(bx, sid)`` variable bindings.

    Examples
    --------
    .. code-block:: python

        with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
            with T.Cube():
                # AIC code using bx
                ...
            with T.Vector():
                # AIV code using both bx and sid
                ...
    """
    from tilelang.language.eager.builder import Builder

    if Builder.current() is None:
        raise JITNoBuilderError("T.MixedKernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.")

    if len(blocks) != 1:
        raise ValueError(f"T.MixedKernel() only supports 1-D block grid. Got {len(blocks)}-D: {blocks}")

    # bool and float both satisfy `in (1, 2)` (True == 1, 1.0 == 1).
    if isinstance(sids, bool) or not isinstance(sids, int):
        raise TypeError(f"T.MixedKernel() sids must be a plain int, got {sids!r}")
    if sids not in (1, 2):
        raise ValueError(f"T.MixedKernel() sids must be 1 or 2. Got {sids}")

    attrs: dict = {}

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.MixedKernelLaunch(blocks, sids, attrs)
