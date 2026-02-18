import cutlass  # noqa: F401
import cutlass.cute as cute  # noqa: F401

# re-export cutlass.cute.arch functions first
from cutlass.cute.arch import sync_threads  # noqa: F401
from cutlass.cute.arch import alloc_smem, get_dyn_smem  # noqa: F401
from cutlass.cute.arch import warpgroup_reg_alloc, warpgroup_reg_dealloc  # noqa: F401
from cutlass.cute.nvgpu.warpgroup.helpers import wait_group as wgmma_wait_group  # noqa: F401

from cutlass.cute import make_tensor, make_rmem_tensor, recast_ptr, where  # noqa: F401
from cutlass.cute.typing import Numeric  # noqa: F401

from cutlass.base_dsl.typing import as_numeric, Int8, Int16, Int32, Uint8, Uint16, Uint32, Float16, Float32, BFloat16  # noqa: F401
from cutlass._mlir.dialects import llvm, arith, nvvm  # noqa: F401
from cutlass._mlir import ir as mlir_ir  # noqa: F401
from cutlass.cutlass_dsl import dsl_user_op  # noqa: F401

# Import our custom implementations (will override if names conflict)
from .utils import *  # noqa: F401,F403
from .cpasync import *  # noqa: F401,F403
from .gemm_v1 import *  # noqa: F401,F403
from .reduce import *  # noqa: F401,F403
from .ldsm import *  # noqa: F401,F403
from .ptx_mma import *  # noqa: F401,F403
from .math import *  # noqa: F401,F403
from .threadblock_swizzle import *  # noqa: F401,F403
from .atomic import *  # noqa: F401,F403
from .quantize import *  # noqa: F401,F403
from .warp import *  # noqa: F401,F403
from .gemm_v2 import *  # noqa: F401,F403
from .gemm_tcgen05 import *  # noqa: F401,F403
from .ieee_math import *  # noqa: F401,F403
from .grid_sync import *  # noqa: F401,F403


def thread_idx():
    """Return linear thread index (threadIdx.x)."""
    tidx, _, _ = cute.arch.thread_idx()
    return tidx
