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

import threading

from tvm import tirx

from tilelang import _ffi_api
from tilelang.jit.exceptions import JITNoBuilderError
from tilelang.language.kernel import (
    FrameStack,
    KernelLaunchFrame,
    get_block_binding,
    get_block_bindings,
    get_block_extent,
    get_block_extents,
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
# SIMT thread scopes
#
# ``T.SimtVF`` owns a thread domain that is nested *inside* the kernel launch,
# so ``T.get_thread_binding()`` and friends have to resolve against the active
# SimtVF scope when there is one and against the launch frame otherwise. Both
# the scope state and the accessors are Ascend's: CUDA/ROCm/Metal declare their
# thread domain on the launch itself and never need the indirection. Keeping
# them here is what lets the shared ``tilelang.language.kernel`` stay free of
# backend branches.
# ---------------------------------------------------------------------------


class SimtVFContext:
    """Stores thread binding info for an active SimtVF scope."""

    __slots__ = ("thread_vars", "thread_extents")

    def __init__(self, thread_vars, thread_extents):
        self.thread_vars = thread_vars
        self.thread_extents = thread_extents


_simtvf_local = threading.local()


def _get_simtvf_stack() -> FrameStack:
    if not hasattr(_simtvf_local, "simtvf_stack"):
        _simtvf_local.simtvf_stack = FrameStack()
    return _simtvf_local.simtvf_stack


def _get_current_simtvf() -> SimtVFContext | None:
    stack = _get_simtvf_stack()
    return stack.top() if stack else None


def push_simtvf_context(ctx: SimtVFContext):
    """Enter a SimtVF thread scope, making its thread vars the current ones."""
    _get_simtvf_stack().push(ctx)


def pop_simtvf_context():
    """Leave the innermost SimtVF thread scope."""
    _get_simtvf_stack().pop()


def get_thread_binding(dim: int = 0):
    """Returns the thread binding for the given dimension.

    Inside a ``T.SimtVF`` block this is the SimtVF thread var; otherwise it is
    the kernel launch's placeholder, which only has a meaning if the pipeline
    materialized SIMT threads (Ascend never does at kernel scope).
    """
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return simtvf.thread_vars[dim]
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_binding(dim)


def get_thread_bindings() -> list:
    """Returns all three thread bindings."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return list(simtvf.thread_vars)
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_bindings()


def get_thread_extent(dim: int = 0) -> int:
    """Returns the thread extent for the given dimension."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return simtvf.thread_extents[dim]
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extent(dim)


def get_thread_extents() -> list:
    """Returns all three thread extents."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return list(simtvf.thread_extents)
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extents()


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
        via ``get_subblockid()`` in generated code.
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

    if sids not in (1, 2):
        raise ValueError(f"T.MixedKernel() sids must be 1 or 2. Got {sids}")

    attrs: dict = {}

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.MixedKernelLaunch(blocks, sids, attrs)
