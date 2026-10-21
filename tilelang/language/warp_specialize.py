"""Typed warp-specialization schedule objects.

A :class:`WSSchedule` is the complete description of how to transform a
straight-line kernel into a warp-specialized one. It is built inside the
kernel (after buffer allocations, so pipelines can reference the buffers
directly), attached with ``T.annotate_ws_schedule``, and consumed by the
``MaterializeWSSchedule`` pass.
"""

from __future__ import annotations

from typing import ClassVar

import tvm_ffi
from tvm.ir import Node
from tvm import tirx
from tvm_ffi.dataclasses import Enum

from tilelang import _ffi_api

__all__ = [
    "WSRole",
    "WSPipeline",
    "WSInstr",
    "WSOpRef",
    "WSSync",
    "WSSyncKind",
    "WSScope",
    "WSSchedule",
]


class WSSyncKind(Enum, type_key="tl.WSSyncKind"):
    """Pipeline synchronization kind.

    The producer waits for the empty barrier (ACQUIRE) and signals the full
    barrier (COMMIT); the consumer waits for the full barrier (WAIT) and
    signals the empty barrier (RELEASE).
    """

    # Bare ClassVar binders attach to the C++-registered variants.
    PRODUCER_ACQUIRE: ClassVar[WSSyncKind]
    PRODUCER_COMMIT: ClassVar[WSSyncKind]
    CONSUMER_WAIT: ClassVar[WSSyncKind]
    CONSUMER_RELEASE: ClassVar[WSSyncKind]


@tvm_ffi.register_object("tl.WSRole")
class WSRole(Node):
    """A contiguous warp range with a single duty.

    Parameters
    ----------
    name : str
        Role name; keys the per-role bodies of every scope.
    warps_lo : int
        First warp of the role's range.
    warps_hi : int
        One past the last warp: the role covers warps lo..hi-1.
    max_nreg : int
        setmaxnreg budget for the role's warps; 0 leaves registers untouched.
    """

    def __init__(self, name: str, *, warps_lo: int, warps_hi: int, max_nreg: int = 0):
        self.__init_handle_by_constructor__(_ffi_api.WSRole, name, int(warps_lo), int(warps_hi), max_nreg)


@tvm_ffi.register_object("tl.WSPipeline")
class WSPipeline(Node):
    """A full/empty mbarrier pair protecting multi-versioned buffers.

    The producer waits for the empty barrier and signals the full barrier;
    the consumer waits for the full barrier and signals the empty barrier.
    A pipeline synchronizes exactly one scope, between exactly two roles.
    Multiple buffers can share one pipeline; conversely, a buffer may be
    bound to several pipelines whose scopes are strictly nested — its
    version count is then the product of their depths (outer-major slots).

    Parameters
    ----------
    name : str
        Pipeline name; referenced by :class:`WSSync` instructions.
    buffers : list[tirx.Buffer]
        The on-chip buffers this pipeline protects (and multi-versions).
    depth : int
        The number of versions this pipeline contributes to each buffer.
    """

    def __init__(self, name: str, buffers: list[tirx.Buffer], depth: int):
        self.__init_handle_by_constructor__(_ffi_api.WSPipeline, name, list(buffers), depth)


@tvm_ffi.register_object("tl.WSInstr")
class WSInstr(Node):
    """Base class of one step in a role's program."""


@tvm_ffi.register_object("tl.WSOpRef")
class WSOpRef(WSInstr):
    """Reference to a tile op or child scope by its stable ``tl.ws_op_id``."""

    def __init__(self, id: str):  # noqa: A002
        self.__init_handle_by_constructor__(_ffi_api.WSOpRef, id)


@tvm_ffi.register_object("tl.WSSync")
class WSSync(WSInstr):
    """A pipeline synchronization point.

    Prefer the classmethod constructors::

        WSSync.producer_acquire("smem", stage=0)
        WSSync.producer_commit("smem", stage=0)
        WSSync.consumer_wait("smem", stage=num_stages - 1)
        WSSync.consumer_release("smem", stage=num_stages - 1)

    Within one role's scope body, acquire/commit (and wait/release) of a
    pipeline must pair up at the same stage; entries between them execute at
    that stage's iteration offset.
    """

    def __init__(self, kind: WSSyncKind, pipeline: str, stage: int = 0):
        self.__init_handle_by_constructor__(_ffi_api.WSSync, kind, pipeline, stage)

    @classmethod
    def producer_acquire(cls, pipeline: str, stage: int = 0) -> WSSync:
        """Wait for the empty barrier; binds the stage's buffer versions."""
        return cls(WSSyncKind.PRODUCER_ACQUIRE, pipeline, stage)

    @classmethod
    def producer_commit(cls, pipeline: str, stage: int = 0) -> WSSync:
        """Signal the full barrier; ends the producer's span."""
        return cls(WSSyncKind.PRODUCER_COMMIT, pipeline, stage)

    @classmethod
    def consumer_wait(cls, pipeline: str, stage: int = 0) -> WSSync:
        """Wait for the full barrier; binds the stage's buffer versions."""
        return cls(WSSyncKind.CONSUMER_WAIT, pipeline, stage)

    @classmethod
    def consumer_release(cls, pipeline: str, stage: int = 0) -> WSSync:
        """Signal the empty barrier; ends the consumer's span."""
        return cls(WSSyncKind.CONSUMER_RELEASE, pipeline, stage)


@tvm_ffi.register_object("tl.WSScope")
class WSScope(Node):
    """A loop (or the root scope) with per-role instruction lists.

    Parameters
    ----------
    id : str
        The ``tl.ws_op_id`` of the loop this scope schedules, or
        :data:`WSScope.ROOT` for the kernel's implicit root scope.
    bodies : dict[str, list[WSInstr | str]]
        Role name -> instruction sequence. Plain strings are shorthand for
        :class:`WSOpRef`.
    """

    ROOT = "tl.ws_scope_root"
    """The id of the kernel's implicit root scope."""

    def __init__(self, id: str, bodies: dict[str, list]):  # noqa: A002
        typed_bodies = {}
        for role, instrs in bodies.items():
            typed_bodies[role] = [WSOpRef(i) if isinstance(i, str) else i for i in instrs]
        self.__init_handle_by_constructor__(_ffi_api.WSScope, id, typed_bodies)


@tvm_ffi.register_object("tl.WSSchedule")
class WSSchedule(Node):
    """The complete warp-specialization schedule of one kernel.

    Parameters
    ----------
    num_warps : int
        Total warp count; overrides the kernel's thread extent.
    roles : list[WSRole]
    pipelines : list[WSPipeline]
    scopes : list[WSScope]
    """

    def __init__(
        self,
        num_warps: int,
        roles: list[WSRole],
        pipelines: list[WSPipeline],
        scopes: list[WSScope],
    ):
        assert num_warps > 0, f"num_warps must be positive, got {num_warps}"
        assert num_warps % 4 == 0, f"num_warps must be a multiple of 4 (setmaxnreg acts on whole warpgroups), got {num_warps}"
        self.__init_handle_by_constructor__(_ffi_api.WSSchedule, num_warps, list(roles), list(pipelines), list(scopes))
