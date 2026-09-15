"""Ascend language dialect: common TileLang plus Ascend extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

# alloc_shared deliberately shadows the common surface: Ascend UB takes
# bool buffers in the requested scope, without CUDA's static-scope hack.
from .allocate import alloc_shared, alloc_l1, alloc_l0a, alloc_l0b, alloc_l0c  # noqa: F401
from .annotations import (  # noqa: F401
    annotate_buffer_versions,
    annotate_manual_multi_buffer,
    annotate_unlimit_memory,
)
from .copy_op import copy, dual_copy  # noqa: F401
from .gemm_op import blockscaled_gemm, gemm  # noqa: F401

# Ascend owns the unroll-factor knob: the Ascend codegen lowers the
# "pragma_unroll_factor" annotation to `#pragma unroll N`.
from .loop import unroll  # noqa: F401

# Ascend owns its launch and its thread-scope accessors. These deliberately
# shadow the common surface imported above: `T.Kernel` here is the 1-D NPU core
# grid (no threads=), and `T.get_thread_binding()` resolves inside T.SimtVF.
from .kernel import (  # noqa: F401
    Kernel,
    MixedKernel,
    get_thread_binding,
    get_thread_bindings,
    get_thread_extent,
    get_thread_extents,
)
from .schedule_hint import PerCoreTask, Stage, Task, assume_conflict, assume_no_conflict  # noqa: F401
from .tile_schedule import (  # noqa: F401
    AscendBaseTileScheduler,
    AscendBatchedTileScheduler,
    AscendKGroupedTileScheduler,
    AscendMGroupedTileScheduler,
    AscendTileScheduler,
)
from .frame import Cube, CubeFrame, SimdVF, SimdVFFrame, SimtVF, SimtVFFrame, Vector, VectorFrame  # noqa: F401

from . import simd as simd  # noqa: F401 (exposed as T.simd.*)

# Ascend owns its debug surface: device_assert lowers through the toolkit's
# assert() macro, print gates by the NPU execution model (no CUDA-style
# single-thread condition), and rng_init derives its default sequence id from
# the SimtVF thread scope.
from ..debug import device_assert  # noqa: F401
from .print import print  # noqa: F401,A001
from .random import rng_init, rng_rand, rng_rand_float  # noqa: F401

from .dma import *  # noqa: F401,F403
from .dma import __all__ as _DMA_ALL
from .mode import *  # noqa: F401,F403
from .mode import __all__ as _MODE_ALL
from .sync import *  # noqa: F401,F403
from .sync import __all__ as _SYNC_ALL

_ASCEND_API_ALL = (
    "AscendBaseTileScheduler",
    "AscendBatchedTileScheduler",
    "AscendKGroupedTileScheduler",
    "AscendMGroupedTileScheduler",
    "AscendTileScheduler",
    "Cube",
    "CubeFrame",
    "MixedKernel",
    "PerCoreTask",
    "Stage",
    "Task",
    "SimdVF",
    "SimdVFFrame",
    "SimtVF",
    "SimtVFFrame",
    "Vector",
    "VectorFrame",
    "alloc_l0a",
    "alloc_l0b",
    "alloc_l0c",
    "alloc_l1",
    "annotate_buffer_versions",
    "annotate_manual_multi_buffer",
    "annotate_unlimit_memory",
    "assume_conflict",
    "assume_no_conflict",
    "blockscaled_gemm",
    "copy",
    "gemm",
    "device_assert",
    "dual_copy",
    "print",
    "rng_init",
    "rng_rand",
    "rng_rand_float",
    "simd",
    "unroll",
)

__tilelang_dialect__ = "ascend"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_ASCEND_API_ALL, *_DMA_ALL, *_MODE_ALL, *_SYNC_ALL)))

del _ASCEND_API_ALL, _COMMON_ALL, _DMA_ALL, _MODE_ALL, _SYNC_ALL
