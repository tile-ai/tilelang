"""Backend-neutral TileLang language surface.

This module owns the common (backend-agnostic) language definitions. Each
per-backend dialect (``tilelang.cuda.language`` etc.) re-exports this surface
via ``from tilelang.language.common import *`` and layers its own extensions on
top. ``tilelang.language`` itself is a thin facade that re-exports the default
(CUDA) dialect.
"""

from __future__ import annotations

# Import the upstream TIR script surface through a target-aware adapter so
# backend-only parser exports do not leak into the common language manifest.
from .tir.common import *  # noqa: F401,F403
from . import overrides as _overrides  # noqa: F401

from .eager import *  # noqa: F401,F403
from .tir.ir import *  # noqa: F401,F403
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import ptr, make_tensor, make_tensor_from_addr, Buffer, Tensor, StridedTensor, FragmentBuffer, SharedBuffer, LocalBuffer  # noqa: F401
from .loop import (
    Parallel,  # noqa: F401
    Persistent,  # noqa: F401
    Pipelined,  # noqa: F401
    serial,  # noqa: F401
    unroll,  # noqa: F401
    vectorized,  # noqa: F401
    Serial,  # noqa: F401
    Unroll,  # noqa: F401
    Vectorized,  # noqa: F401
)
from .frame import has_let_value, get_let_value  # noqa: F401
from .math_intrinsics import *  # noqa: F401,F403
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .allocate import (
    alloc_var,  # noqa: F401
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_global,  # noqa: F401
    alloc_barrier,  # noqa: F401
    alloc_reducer,  # noqa: F401
    empty,  # noqa: F401
)
from tvm.tirx.script.builder.ir import alloc_buffer as allocate  # noqa: F401
from .copy_op import (  # noqa: F401
    copy,
    async_copy,
    transpose,
    im2col,
    c2d_im2col,
)
from tilelang.tileop.base import GemmWarpPolicy  # noqa: F401
from .gemm_op import (  # noqa: F401
    gemm,
)
from .experimental.gemm_sp_op import (  # noqa: F401
    gemm_sp,
)
from .fill_op import fill, clear  # noqa: F401
from .reduce_op import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    reduce_bitand,  # noqa: F401
    reduce_bitor,  # noqa: F401
    reduce_bitxor,  # noqa: F401
    finalize_reducer,  # noqa: F401
    warp_reduce_sum,  # noqa: F401
    warp_reduce_max,  # noqa: F401
    warp_reduce_min,  # noqa: F401
    warp_reduce_bitand,  # noqa: F401
    warp_reduce_bitor,  # noqa: F401
)
from .scan_op import cumsum, cummax  # noqa: F401
from .customize import (
    atomic_max,  # noqa: F401
    atomic_min,  # noqa: F401
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    atomic_addx4,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
    atomic_load,  # noqa: F401
    atomic_or,  # noqa: F401
    atomic_store,  # noqa: F401
    loop_break,  # noqa: F401
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import (  # noqa: F401
    access_ptr,
    activemask,
    all_sync,
    any_sync,
    ballot,
    ballot_sync,
    barrier_arrive,
    barrier_wait,
    get_lane_idx,
    get_warp_idx,
    get_warp_idx_sync,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_expect_tx,
    mbarrier_wait_parity,
    no_set_max_nreg,
    shfl_down,
    shfl_sync,
    shfl_up,
    shfl_xor,
    sync_global,
    sync_grid,
    sync_threads,
    sync_warp,
    syncthreads_and,
    syncthreads_count,
    syncthreads_or,
)

from .utils import index_to_coordinates  # noqa: F401

from .symbolics import dynamic, symbolic  # noqa: F401
from .annotations import (  # noqa: F401
    use_swizzle,
    annotate_layout,
    annotate_safe_value,
    annotate_restrict_buffers,
)

from .meta import (
    inline,  # noqa: F401
    meta_class,  # noqa: F401
)

from .tile_schedule import (
    BaseTileScheduler,  # noqa: F401
    PersistentTileScheduler,  # noqa: F401
)


def import_source(source: str | None = None):
    # source is the source code to be imported
    from tvm.tirx.script.builder.ir import sblock_attr

    return sblock_attr({"pragma_import_c": source}) if source is not None else None


# ``__all__`` is the union of the star-imported submodule surfaces (each of
# which owns its own ``__all__``) plus the names imported explicitly above.
from .tir.common import __all__ as _TIR_COMMON_ALL  # noqa: E402
from .eager import __all__ as _EAGER_ALL  # noqa: E402
from .tir.ir import __all__ as _TIR_IR_ALL  # noqa: E402
from .math_intrinsics import __all__ as _MATH_ALL  # noqa: E402

_LOCAL_EXPORTS = (
    "BaseTileScheduler",
    "Fragment",
    "FragmentBuffer",
    "GemmWarpPolicy",
    "Kernel",
    "KernelLaunchFrame",
    "Layout",
    "LocalBuffer",
    "Parallel",
    "Persistent",
    "PersistentTileScheduler",
    "Pipelined",
    "Serial",
    "SharedBuffer",
    "StridedTensor",
    "Tensor",
    "Unroll",
    "Vectorized",
    "access_ptr",
    "activemask",
    "all_of",
    "all_sync",
    "alloc_barrier",
    "alloc_fragment",
    "alloc_global",
    "alloc_local",
    "alloc_reducer",
    "alloc_shared",
    "alloc_var",
    "allocate",
    "annotate_layout",
    "annotate_restrict_buffers",
    "annotate_safe_value",
    "any_of",
    "any_sync",
    "async_copy",
    "atomic_add",
    "atomic_addx2",
    "atomic_addx4",
    "atomic_load",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_store",
    "ballot",
    "ballot_sync",
    "barrier_arrive",
    "barrier_wait",
    "c2d_im2col",
    "clamp",
    "clear",
    "copy",
    "cummax",
    "cumsum",
    "dynamic",
    "empty",
    "fill",
    "finalize_reducer",
    "gemm",
    "gemm_sp",
    "get_block_binding",
    "get_block_bindings",
    "get_lane_idx",
    "get_let_value",
    "get_thread_binding",
    "get_thread_bindings",
    "get_warp_idx",
    "get_warp_idx_sync",
    "has_let_value",
    "im2col",
    "import_source",
    "index_to_coordinates",
    "inline",
    "loop_break",
    "make_tensor",
    "make_tensor_from_addr",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_expect_tx",
    "mbarrier_wait_parity",
    "meta_class",
    "no_set_max_nreg",
    "reduce",
    "reduce_absmax",
    "reduce_abssum",
    "reduce_bitand",
    "reduce_bitor",
    "reduce_bitxor",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    "reshape",
    "shfl_down",
    "shfl_sync",
    "shfl_up",
    "shfl_xor",
    "symbolic",
    "sync_global",
    "sync_grid",
    "sync_threads",
    "sync_warp",
    "syncthreads_and",
    "syncthreads_count",
    "syncthreads_or",
    "transpose",
    "use_swizzle",
    "view",
    "warp_reduce_bitand",
    "warp_reduce_bitor",
    "warp_reduce_max",
    "warp_reduce_min",
    "warp_reduce_sum",
)

__all__ = tuple(
    dict.fromkeys(
        (
            *_TIR_COMMON_ALL,
            *_EAGER_ALL,
            *_TIR_IR_ALL,
            *_MATH_ALL,
            *_LOCAL_EXPORTS,
        )
    )
)

__tilelang_dialect__ = "common"

del _TIR_COMMON_ALL, _EAGER_ALL, _TIR_IR_ALL, _MATH_ALL, _LOCAL_EXPORTS
