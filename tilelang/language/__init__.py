"""CUDA-compatible TileLang language facade built on the common surface."""

from __future__ import annotations

# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.tirx.script.parser import *
from . import overrides as _overrides  # noqa: F401

# from .tir import prim_func, macro,  # noqa: F401
from .eager import *  # noqa: F401
from .tir.ir import *  # noqa: F401
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
from .math_intrinsics import *  # noqa: F401
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


def _is_language_export(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and not name.endswith("__"))


__tilelang_common_all__ = tuple(name for name in globals() if _is_language_export(name))
__tilelang_dialect__ = "common"
__all__ = __tilelang_common_all__


def _activate_cuda_facade() -> None:
    """Attach the CUDA extension after top-level TileLang initialization."""

    if __tilelang_dialect__ == "cuda":
        return

    import sys

    from tilelang.cuda import language as cuda_language

    module = sys.modules[__name__]
    for name in cuda_language.__all__:
        setattr(module, name, getattr(cuda_language, name))
    module.__all__ = tuple(dict.fromkeys((*__tilelang_common_all__, *cuda_language.__all__)))
    module.__tilelang_dialect__ = "cuda"


del _is_language_export
