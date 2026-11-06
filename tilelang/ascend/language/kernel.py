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
    "PersistentKernel",
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


def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension."""
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

    if sids not in (1, 2):
        raise ValueError(f"T.MixedKernel() sids must be 1 or 2. Got {sids}")

    attrs: dict = {}

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.MixedKernelLaunch(blocks, sids, attrs)


@kernel_launch_factory
def PersistentKernel(
    *blocks: int | tirx.PrimExpr,
    num_cores: int,
    num_stages: int = 0,
    annotations: dict[str, object] | None = None,
    prelude: str | None = None,
):
    """Construct an Ascend kernel whose logical grid is folded onto NPU cores.

    Unlike :func:`Kernel`, this launch explicitly opts into the Ascend
    ``AutoPersistent`` transform. When the logical grid exceeds ``num_cores``,
    the transform launches at most ``num_cores`` physical kernels and executes
    the remaining logical tasks in a serial wave loop.

    Parameters
    ----------
    blocks : int or tirx.PrimExpr
        Positive number of logical tasks in the 1-D launch grid. The extent
        must have a signed integer dtype.
    num_cores : int
        Number of physical Ascend cores to fold the logical grid onto. Required
        and authoritative.
    num_stages : int
        Number of stages attached to the generated wave loop. It has no effect
        when no wave loop is required.
    annotations : dict[str, object], optional
        Additional annotations for the generated wave loop. This follows
        :func:`Pipelined`; for example, opt into offset scheduling with
        ``annotations={"enable_offset": True}``.
    prelude : str, optional
        Import C code injected before the generated kernel.

    Notes
    -----
    ``T.Persistent`` inside ``T.PersistentKernel`` is accepted as an ordinary
    nested serial loop. AutoPersistent does not coordinate or deduplicate the
    two scheduling schemes, so using them together is not recommended.
    """
    from tvm import DataType, DataTypeCode
    from tvm.target import Target

    from tilelang.ascend.target import target_is_ascend
    from tilelang.language.eager.builder import Builder

    if Builder.current() is None:
        raise JITNoBuilderError(
            "T.PersistentKernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available."
        )

    current_target = Target.current(allow_none=True)
    if current_target is not None and not target_is_ascend(current_target):
        raise ValueError("T.PersistentKernel() is only supported by the Ascend backend")

    if len(blocks) != 1:
        raise ValueError(f"T.PersistentKernel() only supports a 1-D grid. Got {len(blocks)}-D grid")
    logical_extent = blocks[0]
    if isinstance(logical_extent, bool):
        raise ValueError(f"T.PersistentKernel() launch extent must be a signed integer expression, got {logical_extent!r}")
    if isinstance(logical_extent, int):
        if logical_extent <= 0:
            raise ValueError(f"T.PersistentKernel() launch extent must be positive, got {logical_extent}")
    elif isinstance(logical_extent, tirx.PrimExpr):
        if DataType(logical_extent.dtype).type_code != DataTypeCode.INT:
            raise ValueError(f"T.PersistentKernel() launch extent must have a signed integer dtype, got {logical_extent.dtype}")
        if isinstance(logical_extent, tirx.IntImm) and int(logical_extent.value) <= 0:
            raise ValueError(f"T.PersistentKernel() launch extent must be positive, got {logical_extent.value}")
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError(f"T.PersistentKernel() num_cores must be a positive integer, got {num_cores!r}")
    if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < 0:
        raise ValueError(f"T.PersistentKernel() num_stages must be a non-negative integer, got {num_stages!r}")
    if annotations is not None and not isinstance(annotations, dict):
        raise ValueError("T.PersistentKernel() annotations must be a dict or None")

    # num_stages is merged into the wave-loop annotations map so the frontend
    # and the pass share a single annotation field (matching T.Pipelined, where
    # an explicit num_stages overrides any same-named key in `annotations`).
    wave_annotations = dict(annotations or {})
    if num_stages > 0:
        wave_annotations["num_stages"] = num_stages

    # These keys are a private contract between this frontend and AutoPersistent:
    # recorded verbatim on the launch block (like ClusterKernel forwards
    # cluster_dims), then consumed and stripped by AutoPersistent before later
    # lowering, so they never reach codegen.
    launch_annotations = {"tilelang.persistent_kernel_num_cores": num_cores}
    if wave_annotations:
        launch_annotations["tilelang.persistent_kernel_annotations"] = wave_annotations
    return launch_kernel(blocks, prelude=prelude, **launch_annotations)
