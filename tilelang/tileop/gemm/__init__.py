from tilelang import tvm as tvm
from tvm import tir
from tilelang.utils.target import (
    target_is_cuda,)
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,)
from tilelang.layout import make_swizzled_layout
from tilelang import language as T
from tvm.target import Target
from tvm.ir.base import Node
from tvm.runtime import Scriptable
import tvm.ffi
from tilelang.ir import GemmWarpPolicy
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_shared, is_fragment

@tvm.ffi.register_func("tl.gemm_py.infer_layout")
def gemm_py_infer_layout(gemm_py, target, thread_bounds):
    thread_nums = thread_bounds.extent
    return gemm_py.infer_layout(target, thread_nums)


@tvm.ffi.register_func("tl.gemm_py.lower")
def gemm_py_lower(gemm_py, target, thread_bounds, thread_var):
    thread_nums = thread_bounds.extent
    stmt = gemm_py.lower(target, thread_nums, thread_var)
    return stmt


@tvm.ffi.register_object("tl.GemmPy")
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
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            # Now only implement ssr layout
            m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target,
                                                                False)
            warp_row_tiles = int(self.M // m_warp)
            warp_col_tiles = int(self.N // n_warp)
            mma_emitter = TensorCoreIntrinEmitter(
                a_dtype=self.in_dtype,
                b_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=self.chunk,
            )
            if self.is_gemm_ss():
                return {
                    self.A: make_swizzled_layout(self.A),
                    self.B: make_swizzled_layout(self.B),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                }
            elif self.is_gemm_sr():
                return {
                    self.A: make_swizzled_layout(self.A),
                    self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                }
            elif self.is_gemm_rs():
                return {
                    self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                    self.B: make_swizzled_layout(self.B),
                    self.C: mma_emitter.make_mma_store_layout(self.C), 
                }
            elif self.is_gemm_rr():
                return {
                    self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                    self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                }
            else:
                raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")
        else:
            raise ValueError(f"Unsupported target: {target}")

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            # Now only implement ssr layout
            m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target,
                                                                False)
            warp_row_tiles = int(self.M // m_warp)
            warp_col_tiles = int(self.N // n_warp)
            mma_emitter = TensorCoreIntrinEmitter(
                a_dtype=self.in_dtype,
                b_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=self.chunk,
                thread_var=thread_var,
            )

            in_dtype = self.in_dtype
            warp_rows = mma_emitter.warp_rows
            warp_cols = mma_emitter.warp_cols
            local_size_a = mma_emitter.local_size_a
            local_size_b = mma_emitter.local_size_b
            block_K = mma_emitter.chunk
            micro_size_k = mma_emitter.micro_size_k
            A_shared = self.A
            B_shared = self.B
            C_local = self.C

            if self.is_gemm_ss():
                @T.prim_func
                def _gemm_ssr() -> None:
                    """
                    The inner macro that loads data from shared buffers A_shared and
                    B_shared into local fragments, then issues Tensor Core mma ops,
                    accumulating into C_local.
                    """
                    A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
                    B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)

                    for ki in T.serial(0, (block_K // micro_size_k)):
                        # Load A into fragment
                        mma_emitter.ldmatrix_a(
                            A_local,
                            A_shared,
                            ki,
                        )

                        # Load B into fragment
                        mma_emitter.ldmatrix_b(
                            B_local,
                            B_shared,
                            ki,
                        )

                        # Perform Matrix Multiplication
                        mma_emitter.mma(A_local, B_local, C_local, ki)

                # Simplify to optimize the index computing
                # Must inline let statements to simplify the analysis
                return _Simplify(_gemm_ssr, inline_let=True)
            elif self.is_gemm_sr():
                B_local = self.B
                @T.prim_func
                def _gemm_srr() -> None:
                    """
                    The inner macro that loads data from shared buffers A_shared and
                    B_shared into local fragments, then issues Tensor Core mma ops,
                    accumulating into C_local.
                    """
                    A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)

                    for ki in T.serial(0, (block_K // micro_size_k)):
       
                        # Load A into fragment
                        mma_emitter.ldmatrix_a(
                            A_local,
                            A_shared,
                            ki,
                        )

                        # Perform Matrix Multiplication
                        mma_emitter.mma(A_local, B_local, C_local, ki)

                # Simplify to optimize the index computing
                # Must inline let statements to simplify the analysis
                # alloc_buffers body
                # insert into parrent block
                return _Simplify(_gemm_srr, inline_let=True)
            elif self.is_gemm_rs():
                A_local = self.A
                @T.prim_func
                def _gemm_rsr() -> None:
                    """
                    The inner macro that loads data from shared buffers A_shared and
                    B_shared into local fragments, then issues Tensor Core mma ops,
                    accumulating into C_local.
                    """
                    B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)

                    for ki in T.serial(0, (block_K // micro_size_k)):
       
                        # Load B into fragment
                        mma_emitter.ldmatrix_b(
                            B_local,
                            B_shared,
                            ki,
                        )

                        # Perform Matrix Multiplication
                        mma_emitter.mma(A_local, B_local, C_local, ki)

                # Simplify to optimize the index computing
                # Must inline let statements to simplify the analysis
                return _Simplify(_gemm_rsr, inline_let=True)
            elif self.is_gemm_rr():
                A_local = self.A
                B_local = self.B
                @T.prim_func
                def _gemm_rsr() -> None:
                    """
                    The inner macro that loads data from shared buffers A_shared and
                    B_shared into local fragments, then issues Tensor Core mma ops,
                    accumulating into C_local.
                    """

                    for ki in T.serial(0, (block_K // micro_size_k)):
                        # Perform Matrix Multiplication
                        mma_emitter.mma(A_local, B_local, C_local, ki)

                # Simplify to optimize the index computing
                # Must inline let statements to simplify the analysis
                return _Simplify(_gemm_rsr, inline_let=True)
            else:
                raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")
        else:
            raise ValueError(f"Unsupported target: {target}")

    @property
    def in_dtype(self) -> str:
        assert self.A.dtype == self.B.dtype, "A and B must have the same dtype"
        return self.A.dtype

    @property
    def accum_dtype(self) -> str:
        return self.C.dtype

    @property
    def chunk(self) -> int:
        return self.A.shape[-2] if self.trans_A else self.A.shape[-1]

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)
    
    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)
    
    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
