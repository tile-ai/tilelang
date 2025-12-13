import cutlass
import cutlass.cute as cute
import cutlass.utils as utils  # noqa: F401
import math
import cutlass.utils.hopper_helpers as hopper_utils
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource, make_smem_layout_atom


def make_aligned_tensor(ptr: cute.Pointer, layout: cute.Layout, align_bytes: int, swizzle=False):
    ptr = ptr.align(align_bytes)
    if swizzle and isinstance(layout, cute.ComposedLayout):
        ptr = cute.recast_ptr(ptr=ptr, swizzle_=layout.inner, dtype=ptr.dtype)
        return cute.make_tensor(ptr, layout.outer)
    return cute.make_tensor(ptr, layout)


def gemm_ss(
    M,
    N,
    K,
    warp_m,
    warp_n,
    trans_A,
    trans_B,
    clear_accum,
    stride_A,
    stride_B,
    offset_A,
    offset_B,
    use_wgmma=None,
    wg_wait=0,
    A_ptr: cute.Pointer = None,
    B_ptr: cute.Pointer = None,
    C_ptr: cute.Pointer = None,
):
    """GEMM with both A and B from shared memory"""
    if use_wgmma:
        gemm = Gemm_SM90(
            M,
            N,
            K,
            warp_m,
            warp_n,
            trans_A,
            trans_B,
            clear_accum,
            stride_A,
            stride_B,
            offset_A,
            offset_B,
            A_ptr.dtype,
            B_ptr.dtype,
            C_ptr.dtype,
        )
        gemm(A_ptr, B_ptr, C_ptr, wg_wait=wg_wait, clear_accum=clear_accum)
    else:
        gemm = Gemm_SM80(
            M,
            N,
            K,
            warp_m,
            warp_n,
            trans_A,
            trans_B,
            clear_accum,
            stride_A,
            stride_B,
            offset_A,
            offset_B,
            A_ptr.dtype,
            B_ptr.dtype,
            C_ptr.dtype,
        )
        gemm(A_ptr, B_ptr, C_ptr)


def gemm_rs(
    M,
    N,
    K,
    warp_m,
    warp_n,
    trans_A,
    trans_B,
    clear_accum,
    stride_A,
    stride_B,
    offset_A,
    offset_B,
    use_wgmma=None,
    wg_wait=0,
    A_ptr: cute.Pointer = None,
    B_ptr: cute.Pointer = None,
    C_ptr: cute.Pointer = None,
):
    """GEMM with A from register/fragment and B from shared memory"""
    if use_wgmma:
        gemm = Gemm_SM90(
            M,
            N,
            K,
            warp_m,
            warp_n,
            trans_A,
            trans_B,
            clear_accum,
            stride_A,
            stride_B,
            offset_A,
            offset_B,
            A_ptr.dtype,
            B_ptr.dtype,
            C_ptr.dtype,
        )
        gemm.body_rs(A_ptr, B_ptr, C_ptr, wg_wait=wg_wait, clear_accum=clear_accum)
    else:
        gemm = Gemm_SM80(
            M,
            N,
            K,
            warp_m,
            warp_n,
            trans_A,
            trans_B,
            clear_accum,
            stride_A,
            stride_B,
            offset_A,
            offset_B,
            A_ptr.dtype,
            B_ptr.dtype,
            C_ptr.dtype,
        )
        gemm.body_rs(A_ptr, B_ptr, C_ptr)


def gemm_sr(
    M,
    N,
    K,
    warp_m,
    warp_n,
    trans_A,
    trans_B,
    clear_accum,
    stride_A,
    stride_B,
    offset_A,
    offset_B,
    use_wgmma=None,
    wg_wait=0,
    A_ptr: cute.Pointer = None,
    B_ptr: cute.Pointer = None,
    C_ptr: cute.Pointer = None,
):
    """GEMM with A from shared memory and B from register/fragment"""
    # wgmma doesn't support gemm_sr, only use SM80
    gemm = Gemm_SM80(
        M,
        N,
        K,
        warp_m,
        warp_n,
        trans_A,
        trans_B,
        clear_accum,
        stride_A,
        stride_B,
        offset_A,
        offset_B,
        A_ptr.dtype,
        B_ptr.dtype,
        C_ptr.dtype,
    )
    gemm.body_sr(A_ptr, B_ptr, C_ptr)


def gemm_rr(
    M,
    N,
    K,
    warp_m,
    warp_n,
    trans_A,
    trans_B,
    clear_accum,
    stride_A,
    stride_B,
    offset_A,
    offset_B,
    use_wgmma=None,
    wg_wait=0,
    A_ptr: cute.Pointer = None,
    B_ptr: cute.Pointer = None,
    C_ptr: cute.Pointer = None,
):
    """GEMM with both A and B from register/fragment"""
    # Both operands in register, no copy needed
    gemm = Gemm_SM80(
        M,
        N,
        K,
        warp_m,
        warp_n,
        trans_A,
        trans_B,
        clear_accum,
        stride_A,
        stride_B,
        offset_A,
        offset_B,
        A_ptr.dtype,
        B_ptr.dtype,
        C_ptr.dtype,
    )
    # For gemm_rr, directly call _body_impl with copy_A=False, copy_B=False
    gemm._body_impl(A_ptr, B_ptr, C_ptr, copy_A=False, copy_B=False)


class Gemm_SM80:
    _instances = {}  # cache instances for the same arguments

    def __new__(cls, *args):
        key = args
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    # in Tilelang, trans_A == 0 or trans_B == 1 means K major
    # in Cute, trans == 0 means K major
    def __init__(
        self, M, N, K, warp_m, warp_n, trans_A, trans_B, clear_accum, stride_A, stride_B, offset_A, offset_B, A_type, B_type, C_type
    ):
        if not hasattr(self, "initialized"):
            self.cta_tiler = (M, N, K)
            self.mma_inst_shape = (16, 8, 16)
            self.trans_A = trans_A != 0  # same with Tilelang
            self.trans_B = trans_B == 0  # inverse with Tilelang
            A_major_mode = LayoutEnum.COL_MAJOR if self.trans_A else LayoutEnum.ROW_MAJOR
            B_major_mode = LayoutEnum.COL_MAJOR if self.trans_B else LayoutEnum.ROW_MAJOR
            self.A_layout = self._make_smem_layout_AB(A_type, A_major_mode, 128, (M, K))
            self.B_layout = self._make_smem_layout_AB(B_type, B_major_mode, 128, (N, K))
            self.ab_dtype = A_type
            self.acc_dtype = C_type
            self.tiled_mma = self._make_tiled_mma(warp_m, warp_n)
            self.clear_accum = clear_accum

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        is_row_major = major_mode == LayoutEnum.ROW_MAJOR
        major_mode_size = smem_tiler[1] if is_row_major else smem_tiler[0]
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if is_row_major
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1) if is_row_major else (1, 0))
        return layout

    def _make_tiled_mma(self, warp_m, warp_n):
        atom_layout_mnk = (warp_m, warp_n, 1)
        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            atom_layout_mnk[0] * self.mma_inst_shape[0],
            atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(op, atom_layout_mnk, permutation_mnk)
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        sA_ptr: cute.Pointer,
        sB_ptr: cute.Pointer,
        rC_ptr: cute.Pointer,
    ):
        """GEMM body: both A and B from shared memory"""
        self._body_impl(sA_ptr, sB_ptr, rC_ptr, copy_A=True, copy_B=True)

    @cute.jit
    def body_rs(
        self,
        rA_ptr: cute.Pointer,  # A already in register
        sB_ptr: cute.Pointer,  # B from shared memory
        rC_ptr: cute.Pointer,
    ):
        """GEMM body_rs: A from register, B from shared memory"""
        self._body_impl(rA_ptr, sB_ptr, rC_ptr, copy_A=False, copy_B=True)

    @cute.jit
    def body_sr(
        self,
        sA_ptr: cute.Pointer,  # A from shared memory
        rB_ptr: cute.Pointer,  # B already in register
        rC_ptr: cute.Pointer,
    ):
        """GEMM body_sr: A from shared memory, B from register"""
        self._body_impl(sA_ptr, rB_ptr, rC_ptr, copy_A=True, copy_B=False)

    @cute.jit
    def _body_impl(
        self,
        A_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        rC_ptr: cute.Pointer,
        copy_A: cutlass.Constexpr = True,
        copy_B: cutlass.Constexpr = True,
    ):
        """Internal implementation with configurable copy operations"""
        tidx, _, _ = cute.arch.thread_idx()
        thr_mma = self.tiled_mma.get_slice(tidx)

        tCrA = None
        tCrB = None
        tCrC = cute.make_tensor(rC_ptr, self.tiled_mma.partition_shape_C((self.cta_tiler[0], self.cta_tiler[1])))

        # Create copy operations only for operands that need copying
        if cutlass.const_expr(copy_A):
            sA = make_aligned_tensor(A_ptr, self.A_layout, 16)
            tCsA = thr_mma.partition_A(sA)
            tCrA = self.tiled_mma.make_fragment_A(tCsA)
            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.trans_A, 4),
                sA.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy(
                atom_copy_s2r_A,
                layout_tv=self.tiled_mma.tv_layout_A_tiled,
                tiler_mn=(self.tiled_mma.get_tile_size(0), self.tiled_mma.get_tile_size(2)),
            )
            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        else:
            # A already in register
            tCrA = cute.make_tensor(A_ptr, self.tiled_mma.partition_shape_A((self.cta_tiler[0], self.cta_tiler[2])))

        if cutlass.const_expr(copy_B):
            sB = make_aligned_tensor(B_ptr, self.B_layout, 16)
            tCsB = thr_mma.partition_B(sB)
            tCrB = self.tiled_mma.make_fragment_B(tCsB)
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.trans_B, 4),
                sB.element_type,
            )
            tiled_copy_s2r_B = cute.make_tiled_copy(
                atom_copy_s2r_B,
                layout_tv=self.tiled_mma.tv_layout_B_tiled,
                tiler_mn=(self.tiled_mma.get_tile_size(1), self.tiled_mma.get_tile_size(2)),
            )
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
        else:
            # B already in register
            tCrB = cute.make_tensor(B_ptr, self.tiled_mma.partition_shape_B((self.cta_tiler[1], self.cta_tiler[2])))

        if self.clear_accum:
            tCrC.fill(0)

        for k in cutlass.range(cute.size(tCrA, mode=[2])):
            if cutlass.const_expr(copy_A):
                cute.copy(tiled_copy_s2r_A, tCsA_copy_view[None, None, k], tCrA_copy_view[None, None, k])
            if cutlass.const_expr(copy_B):
                cute.copy(tiled_copy_s2r_B, tCsB_copy_view[None, None, k], tCrB_copy_view[None, None, k])
            cute.gemm(self.tiled_mma, tCrC, tCrA[None, None, k], tCrB[None, None, k], tCrC)


class Gemm_SM90:
    _instances = {}  # cache instances for the same arguments

    def __new__(cls, *args):
        key = args
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    # in Tilelang, trans_A == 0 or trans_B == 1 means K major
    # in Cute, trans == 0 means K major
    def __init__(
        self, M, N, K, warp_m, warp_n, trans_A, trans_B, clear_accum, stride_A, stride_B, offset_A, offset_B, A_type, B_type, C_type
    ):
        if not hasattr(self, "initialized"):
            self.cta_tiler = (M, N, K)
            self.tiler_mn = (M, N)
            self.atom_layout_mnk = (warp_m // 4, warp_n, 1)
            self.trans_A = trans_A != 0  # same with Tilelang
            self.trans_B = trans_B == 0  # inverse with Tilelang
            self.a_leading_mode = OperandMajorMode.MN if self.trans_A else OperandMajorMode.K
            self.b_leading_mode = OperandMajorMode.MN if self.trans_B else OperandMajorMode.K
            A_major_mode = LayoutEnum.COL_MAJOR if self.trans_A else LayoutEnum.ROW_MAJOR
            B_major_mode = LayoutEnum.COL_MAJOR if self.trans_B else LayoutEnum.ROW_MAJOR
            self.A_layout = self.make_smem_layout_AB(A_type, A_major_mode, (M, K))
            self.B_layout = self.make_smem_layout_AB(B_type, B_major_mode, (N, K))
            self.a_dtype = A_type
            self.b_dtype = B_type
            self.acc_dtype = C_type
            self.tiled_mma = None
            self.A_source = None
            self.clear_accum = clear_accum

    @staticmethod
    def make_tma_atom(
        tensor,
        smem_layout_staged,
        smem_tile,
        mcast_dim,
    ):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() if mcast_dim == 1 else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))

        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )

        return tma_atom

    @staticmethod
    def get_tma_atom(tensor, tiler_mk, stages=1):
        smem_layout_staged = Gemm_SM90.make_smem_layout_AB(tensor.element_type, LayoutEnum.from_tensor(tensor), tiler_mk, stages)
        tma_atom = Gemm_SM90.make_tma_atom(tensor, smem_layout_staged, tiler_mk, 1)
        return tma_atom

    @staticmethod
    def make_smem_layout_AB(dtype, major_mode: LayoutEnum, tiler_mk, stages=1):
        smem_shape = tiler_mk
        # Determine if K is the major mode and get the major mode size
        is_k_major = major_mode.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        major_mode_size = tiler_mk[1] if is_k_major else tiler_mk[0]

        # Create SMEM layout atom for A tensor based on major mode and data type
        smem_layout_atom = make_smem_layout_atom(
            hopper_utils.get_smem_layout_atom(major_mode, dtype, major_mode_size),
            dtype,
        )
        # Tile the SMEM layout atom to the A tensor shape and add staging dimension
        smem_layout = cute.tile_to_shape(smem_layout_atom, cute.append(smem_shape, stages), order=(0, 1, 2) if is_k_major else (1, 0, 2))
        return smem_layout

    def _make_tiled_mma(self, is_rsMode=False):
        tiled_mma = hopper_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_leading_mode,
            self.b_leading_mode,
            self.acc_dtype,
            self.atom_layout_mnk,
            (64, self.tiler_mn[1] // self.atom_layout_mnk[1]),
            OperandSource.SMEM if not is_rsMode else OperandSource.RMEM,
        )
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        sA_ptr: cute.Pointer,
        sB_ptr: cute.Pointer,
        rC_ptr: cute.Pointer,
        wg_wait: cutlass.Constexpr = 0,
        clear_accum: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        self.tiled_mma = self._make_tiled_mma()
        thr_mma = self.tiled_mma.get_slice(tidx)

        sA_ptr = cute.recast_ptr(sA_ptr, self.A_layout.inner, dtype=sA_ptr.dtype)
        sB_ptr = cute.recast_ptr(sB_ptr, self.B_layout.inner, dtype=sB_ptr.dtype)
        sA = cute.make_tensor(sA_ptr, self.A_layout.outer)
        sB = cute.make_tensor(sB_ptr, self.B_layout.outer)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = self.tiled_mma.make_fragment_A(tCsA)
        tCrB = self.tiled_mma.make_fragment_B(tCsB)
        tCrC = cute.make_tensor(rC_ptr, self.tiled_mma.partition_shape_C((self.cta_tiler[0], self.cta_tiler[1])))

        cute.nvgpu.warpgroup.fence()
        if cutlass.const_expr(clear_accum):
            self.tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        else:
            self.tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k in cutlass.range(num_k_blocks):
            tCrA_1phase = tCrA[None, None, k, 0]
            tCrB_1phase = tCrB[None, None, k, 0]
            cute.gemm(self.tiled_mma, tCrC, tCrA_1phase, tCrB_1phase, tCrC)
            self.tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        cute.nvgpu.warpgroup.commit_group()
        if cutlass.const_expr(wg_wait >= 0):
            cute.nvgpu.warpgroup.wait_group(wg_wait)

    @cute.jit
    def body_rs(
        self,
        rA_ptr: cute.Pointer,  # A already in register (Fragment)
        sB_ptr: cute.Pointer,  # B from shared memory
        rC_ptr: cute.Pointer,
        wg_wait: cutlass.Constexpr = 0,
        clear_accum: cutlass.Constexpr = False,
    ):
        """
        GEMM body_rs for SM90/Hopper: A from register, B from shared memory.
        Based on cute::tl_wgmma::GemmTensorOp::body_rs from gemm_sm90.h
        """
        tidx, _, _ = cute.arch.thread_idx()
        self.tiled_mma = self._make_tiled_mma(is_rsMode=True)
        # if self.A_source != OperandSource.RMEM or self.tiled_mma is None:
        #     self.tiled_mma = self._make_tiled_mma(is_rsMode = True)
        #     self.A_source = OperandSource.RMEM
        # B from shared memory (with swizzle)
        sB_ptr = cute.recast_ptr(sB_ptr, self.B_layout.inner, dtype=sB_ptr.dtype)
        sB = cute.make_tensor(sB_ptr, self.B_layout.outer)

        # Use the existing tiled_mma
        thr_mma = self.tiled_mma.get_slice(tidx)

        # Partition B from shared memory - standard path
        tCsB = thr_mma.partition_B(sB)
        tCrB = self.tiled_mma.make_fragment_B(tCsB)

        # A already in register
        # For body_rs, A is NOT partitioned through thr_mma (it's already partitioned)
        # We create the tensor directly with the full shape
        # This matches C++: make_tensor(make_rmem_ptr(pA), partition_shape_A(...))
        tCrA = cute.make_tensor(rA_ptr, self.tiled_mma.partition_shape_A((self.cta_tiler[0], self.cta_tiler[2])))

        # C accumulator
        tCrC = cute.make_tensor(rC_ptr, self.tiled_mma.partition_shape_C((self.cta_tiler[0], self.cta_tiler[1])))

        # Fence operands (prepare for wgmma)
        cute.nvgpu.warpgroup.fence()
        # Note: warpgroup_arrive() is called internally by wgmma
        # Set accumulation mode
        if cutlass.const_expr(clear_accum):
            self.tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        else:
            self.tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        # GEMM loop
        num_k_blocks = cute.size(tCrB, mode=[2])
        for k_block in cutlass.range(num_k_blocks):
            # Match the indexing pattern from __call__
            # If tCrB has 4 dimensions (with pipeline), use [None, None, k, 0]
            # Otherwise use [None, None, k]
            tCrB_k = tCrB[None, None, k_block, 0] if cute.rank(tCrB) >= 4 else tCrB[None, None, k_block]
            tCrA_k = tCrA[None, None, k_block, 0] if cute.rank(tCrA) >= 4 else tCrA[None, None, k_block]
            cute.gemm(self.tiled_mma, tCrC, tCrA_k, tCrB_k, tCrC)
            self.tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        cute.nvgpu.warpgroup.commit_group()
        if cutlass.const_expr(wg_wait >= 0):
            cute.nvgpu.warpgroup.wait_group(wg_wait)
