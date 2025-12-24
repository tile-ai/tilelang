from enum import IntEnum
from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from .gemm_mma import GemmMMA
from .gemm_mma_sm70 import GemmMMASm70
from .gemm_wgmma import GemmWGMMA
from .gemm_tcgen05 import GemmTCGEN5
from .gemm_mfma import GemmMFMA
from tilelang import _ffi_api
from tilelang.utils.target import target_is_volta


@tvm_ffi.register_global_func("tl.gemm_py.infer_layout")
def gemm_py_infer_layout(gemm_py: GemmMMA, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm_py.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm_py.lower")
def gemm_py_lower(gemm_py: GemmMMA, layout_map, target: Target, thread_bounds: Range, thread_var: tir.Var):
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
    # FFI fields (LLVM/MLIR-style lowerCamel via reflection):
    # a, b, c, aPtr, bPtr, cPtr, m, n, k, transA, transB,
    # strideA, strideB, offsetA, offsetB, clearAccum, kPack, wgWait, policy
    #
    # Backward-compat alias properties are provided below to support old names.

    # Backward-compat alias properties (old API → new FFI fields)
    @property
    def A(self):
        return self.a

    @property
    def B(self):
        return self.b

    @property
    def C(self):
        return self.c

    @property
    def APtr(self):
        return self.aPtr

    @property
    def BPtr(self):
        return self.bPtr

    @property
    def CPtr(self):
        return self.cPtr

    @property
    def M(self):
        return self.m

    @property
    def N(self):
        return self.n

    @property
    def K(self):
        return self.k

    @property
    def trans_A(self):
        return self.transA

    @property
    def trans_B(self):
        return self.transB

    @property
    def stride_A(self):
        return self.strideA

    @property
    def stride_B(self):
        return self.strideB

    @property
    def offset_A(self):
        return self.offsetA

    @property
    def offset_B(self):
        return self.offsetB

    @property
    def clear_accum(self):
        return self.clearAccum

    @property
    def k_pack(self):
        return self.kPack

    @property
    def wg_wait(self):
        return self.wgWait

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
