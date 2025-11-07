from enum import IntEnum
from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target
from tvm.ir.base import Node
from tvm.runtime import Scriptable
import tvm_ffi
from tilelang.ir import GemmWarpPolicy
from .gemm_mma import GemmMMA
from .gemm_mma_sm70 import GemmMMASm70
from .gemm_wgmma import GemmWGMMA
from .gemm_tcgen05 import GemmTCGEN5
from .gemm_mfma import GemmMFMA
from tilelang import _ffi_api
from tilelang.utils.target import target_is_volta


@tvm_ffi.register_global_func("tl.gemm_py.infer_layout")
def gemm_py_infer_layout(gemm_py, target, thread_bounds):
    thread_nums = thread_bounds.extent
    return gemm_py.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm_py.lower")
def gemm_py_lower(gemm_py, layout_map, target, thread_bounds, thread_var):
    thread_nums = thread_bounds.extent
    stmt = gemm_py.lower(layout_map, target, thread_nums, thread_var)
    return stmt


# TODO(lei): support Volta and WMMA?
# same definition with src/op/gemm_py.h
class GemmInst(IntEnum):
    MMA = 0
    WGMMA = 1
    TCGEN5MMA = 2
    MFMA = 3

    def is_mma(self) -> bool:
        return self == GemmInst.MMA

    def is_wgmma(self) -> bool:
        return self == GemmInst.WGMMA

    def is_tcgen5mma(self) -> bool:
        return self == GemmInst.TCGEN5MMA

    def is_mfma(self) -> bool:
        return self == GemmInst.MFMA

    def __repr__(self) -> str:
        return self.name


@tvm_ffi.register_object("tl.GemmPy")
class GemmPy(Node, Scriptable):
    A: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer

    APtr: tir.PrimExpr
    BPtr: tir.PrimExpr
    CPtr: tir.PrimExpr

    M: int
    N: int
    K: int

    trans_A: bool
    trans_B: bool

    stride_A: int
    stride_B: int
    offset_A: int
    offset_B: int
    clear_accum: bool
    k_pack: int
    wg_wait: int
    policy: GemmWarpPolicy

    def infer_layout(self, target: Target, thread_nums: int):
        """Infer the layout for the GEMM operation based on target architecture."""
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        """Lower the GEMM operation to TIR statements based on target architecture."""
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).lower(layout_map, target, thread_nums, thread_var)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> GemmInst:
        """Select the appropriate GEMM instruction based on target and thread configuration.

        The selection logic follows this priority:
        1. WGMMA for Hopper architecture with sufficient matrix size and warp count
        2. MFMA for CDNA (AMD) architecture
        3. MMA for CUDA architecture
        4. Fallback to MMA for other cases

        Args:
            thread_nums: Number of threads in the block
            target: Target architecture

        Returns:
            GemmInst: The selected GEMM instruction type
        """
        return GemmInst(_ffi_api.GemmPyGemmInst(self, int(thread_nums), target))

    def _get_implementation_class(self, gemm_inst: GemmInst, target: Target):
        """Get the appropriate implementation class for the given GEMM instruction.

        Args:
            gemm_inst: The selected GEMM instruction type

        Returns:
            The implementation class for the instruction type

        Raises:
            NotImplementedError: If the instruction type is not supported
            ValueError: If the instruction type is unknown
        """
        if gemm_inst.is_mma():
            if target_is_volta(target):
                return GemmMMASm70
            return GemmMMA
        elif gemm_inst.is_wgmma():
            return GemmWGMMA
        elif gemm_inst.is_tcgen5mma():
            return GemmTCGEN5
        elif gemm_inst.is_mfma():
            return GemmMFMA
        elif gemm_inst.is_tcgen5mma():
            raise NotImplementedError("TCGEN5MMA is not implemented")
        else:
            raise ValueError(f"Unsupported GEMM instruction: {gemm_inst}")
