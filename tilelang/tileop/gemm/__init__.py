from tilelang import tvm as tvm
from tvm import tirx
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from .registry import resolve_gemm_impl
from tilelang import _ffi_api


def _is_mma_sync_dtype_supported(node, target) -> bool:
    if not _ffi_api.TargetIsAmpere(target):
        return True
    a = str(node.a.dtype)
    b = str(node.b.dtype)
    c = str(node.c.dtype)
    arch = str(target.attrs.get("arch", ""))
    try:
        sm = int(arch.split("_")[-1])
    except (ValueError, IndexError):
        sm = 0
    if a.startswith("float8") and b.startswith("float8"):
        return sm >= 89 and (c in ("float16", "float32"))
    if a != b:
        return False
    if a == "float16":
        return c in ("float16", "float32")
    if a in ("int4", "int8", "uint4", "uint8"):
        return c == "int32"
    if a == "bfloat16":
        return c == "float32"
    if a == "custom[tfloat32]":
        return c == "float32"
    if a == "float32":
        return c == "float32"
    if a == "float64":
        return c == "float64"
    return False


def _raise_mma_sync_dtype_unsupported(self, target):
    raise tvm.error.InternalError(
        f"T.gemm requires native mma.sync lowering on Ampere/Ada, "
        f"but the operand dtype configuration is not supported. "
        f"Got target={target}, "
        f"A(scope={self.a.scope()}, dtype={self.a.dtype}), "
        f"B(scope={self.b.scope()}, dtype={self.b.dtype}), "
        f"C(scope={self.c.scope()}, dtype={self.c.dtype}), "
        f"M={self.m}, N={self.n}, K={self.k}."
    )


@tvm_ffi.register_global_func("tl.gemm.infer_layout")
def gemm_infer_layout(gemm, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm.lower")
def gemm_lower(
    gemm,
    layout_map,
    target: Target,
    thread_bounds: Range,
    thread_index: tirx.PrimExpr,
    mbar_phase_expr: tirx.PrimExpr,
):
    # We pass thread_bounds rather than thread_extents because tcgen5mma need to check this
    stmt = gemm.lower(layout_map, target, thread_bounds, thread_index, mbar_phase_expr)
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
    def sf_k_start(self):
        return self.sfKStart

    def infer_layout(self, target: Target, thread_nums: int):
        """Infer the layout for the GEMM operation based on target architecture."""
        if not _is_mma_sync_dtype_supported(self, target):
            _raise_mma_sync_dtype_unsupported(self, target)
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr,
    ):
        """Lower the GEMM operation to TIR statements based on target architecture."""
        if not _is_mma_sync_dtype_supported(self, target):
            _raise_mma_sync_dtype_unsupported(self, target)
        thread_nums = thread_bounds.extent
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).lower(layout_map, target, thread_bounds, thread_index, mbar_phase_expr)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> str:
        """Select the appropriate GEMM instruction key based on target and thread configuration.

        The selection logic chooses:
        1. TCGEN5MMA for Blackwell architecture
        2. WGMMA for Hopper architecture with sufficient matrix size and warp count
        3. MFMA for CDNA (AMD) architecture
        4. MMA for CUDA architecture
        5. Scalar for CPU target (scalar fallback)

        Args:
            thread_nums: Number of threads in the block
            target: Target architecture

        Returns:
            The selected backend-specific GEMM instruction key.
        """
        return str(_ffi_api.GemmGetGemmInstructionKey(self, int(thread_nums), target))

    def _get_implementation_class(self, gemm_inst: str, target: Target):
        """Get the appropriate implementation class for the given GEMM instruction key.

        Args:
            gemm_inst: The selected backend-specific GEMM instruction key
            target: Target architecture

        Returns:
            The implementation class for the instruction key

        Raises:
            NotImplementedError: If the instruction key is not supported
            ValueError: If the instruction key is unknown
        """
        return resolve_gemm_impl(gemm_inst, target)
