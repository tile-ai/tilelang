from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from .inst import GemmInst
from .gemm_mma import GemmMMA
from .gemm_mma_sm70 import GemmMMASm70
from .gemm_wgmma import GemmWGMMA
from .gemm_tcgen05 import GemmTCGEN5
from .gemm_mfma import GemmMFMA
from .gemm_wmma import GemmWMMA
from .gemm_scalar import GemmScalar
from .gemm_metal import GemmMetal
from tilelang import _ffi_api
from tilelang.utils.target import target_is_volta, target_is_metal


@tvm_ffi.register_global_func("tl.gemm.infer_layout")
def gemm_infer_layout(gemm: GemmMMA, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm.lower")
def gemm_lower(
    gemm: GemmMMA,
    layout_map,
    target: Target,
    thread_bounds: Range,
    thread_var: tir.Var,
    mbar_phase_expr: tir.PrimExpr,
):
    # We pass thread_bounds rather than thread_extents because tcgen5mma need to check this
    stmt = gemm.lower(layout_map, target, thread_bounds, thread_var, mbar_phase_expr)
    return stmt


@tvm_ffi.register_object("tl.Gemm")
class Gemm(Node, Scriptable):
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

    @property
    def is_tcgen05(self):
        return getattr(self, "isTcgen05", False)

    @property
    def sf_a_id(self):
        return self.sfAId

    @property
    def sf_b_id(self):
        return self.sfBId

    def infer_layout(self, target: Target, thread_nums: int):
        """Infer the layout for the GEMM operation based on target architecture."""
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr,
    ):
        """Lower the GEMM operation to TIR statements based on target architecture."""
        thread_nums = thread_bounds.extent
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).lower(layout_map, target, thread_bounds, thread_var, mbar_phase_expr)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> GemmInst:
        """Select the appropriate GEMM instruction based on target and thread configuration.

        The selection logic follows this priority:
        1. TCGEN5MMA for Blackwell architecture
        2. WGMMA for Hopper architecture with sufficient matrix size and warp count
        3. MFMA for CDNA (AMD) architecture
        4. WMMA for RDNA (AMD) architecture
        5. MMA for CUDA architecture
        6. METAL_SIMDGROUP for Metal target (simdgroup_matrix)
        7. Scalar for CPU target (scalar fallback)

        Special-case on Metal:

        - FP8 inputs: Apple Silicon has no native FP8 hardware (M1-M5
          inclusive -- see Apple WWDC 2025 cooperative tensors session).
          The TileLang Metal codegen rejects allocating metal.simdgroup
          buffers with FP8 dtype (see codegen_metal.cc:454 --
          "Only float16, float32, and bfloat16 are supported"). We route
          FP8-input GEMMs to the scalar fallback (GemmInst.Scalar),
          which on Metal targets emits per-element T.cast(value, accum_dtype)
          reads for both operands. The T.cast from FP8 to a wider dtype is
          handled by the storage-only FP8 emulation patch in
          codegen_metal.cc::VisitExpr_(CastNode) which expands to
          __tvm_fp8_e4m3_to_half / __tvm_fp8_e5m2_to_half
          helper calls (see
          docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch).

          This mirrors the audiohacking fp8_scaled_matmul_kernel
          reference (https://github.com/audiohacking/fp8-mps-metal): a
          scalar dequant-multiply-accumulate loop in half /
          float rather than any FP8 simdgroup intrinsic.

        Args:
            thread_nums: Number of threads in the block
            target: Target architecture

        Returns:
            GemmInst: The selected GEMM instruction type
        """
        if target_is_metal(target):
            # FP8 (e4m3 / e5m2 / e8m0fnu) inputs: Apple has no native FP8
            # ALU through M5; route to the scalar fallback so the per-element
            # T.cast(..., accum_dtype) invokes the storage-only FP8 decode
            # helpers. The runtime mapping of GemmInst.Scalar to
            # GemmMetalScalar on Metal targets is provided by the
            # tilelang_gemm_mixed_dtype companion patch (PR #2118 stack);
            # without it the resulting kernel will not lower correctly,
            # but the routing decision is the load-bearing change here.
            if self._has_fp8_input_dtype():
                return GemmInst.Scalar
            return GemmInst.METAL_SIMDGROUP
        return GemmInst(_ffi_api.GemmGetGemmInst(self, int(thread_nums), target))

    def _has_fp8_input_dtype(self) -> bool:
        """Return True if either A or B carries an FP8 dtype.

        Used by the Metal dispatcher to force routing through the scalar
        fallback (GemmMetalScalar via the companion mixed-dtype patch)
        because Metal has no native FP8 ALU (Apple Silicon M1-M5 inclusive)
        and the TileLang codegen rejects allocating metal.simdgroup buffers
        with FP8 dtype. The scalar fallback's T.cast(..., accum_dtype) reads
        invoke the storage-only FP8 decode helpers emitted by
        codegen_metal.cc::VisitExpr_(CastNode).
        """
        a = getattr(self, "a", None)
        b = getattr(self, "b", None)
        for buf in (a, b):
            if buf is None:
                continue
            try:
                if str(buf.dtype).startswith("float8"):
                    return True
            except AttributeError:  # pragma: no cover - defensive
                continue
        return False

    def _get_implementation_class(self, gemm_inst: GemmInst, target: Target):
        """Get the appropriate implementation class for the given GEMM instruction.

        Args:
            gemm_inst: The selected GEMM instruction type
            target: Target architecture

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
        elif gemm_inst.is_wmma():
            return GemmWMMA
        elif gemm_inst.is_scalar():
            return GemmScalar
        elif gemm_inst.is_metal_simdgroup():
            return GemmMetal
        else:
            raise ValueError(f"Unsupported GEMM instruction: {gemm_inst}")
