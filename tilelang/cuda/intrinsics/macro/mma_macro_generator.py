from __future__ import annotations
import tilelang.language as T
from dataclasses import dataclass
from typing import Literal
from collections.abc import Callable
from tilelang.common import TransformKind
from tvm import DataType
from tvm import tirx
from tvm.ir import Range
from tvm.tirx import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tilelang import tvm as tvm
from tvm.runtime import convert
from ..layout.utils import (
    mma_store_index_map,
    mma_store_index_map_fp64,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment, get_buffer_region_from_load
from tilelang.cuda.intrinsics.layout.mma_layout import (
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x8_to_mma_32x4_layout_sr_b,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_b,
    shared_16x32_to_mma_32x16_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_b,
    mma_load_a_32x4_to_shared_16x8_layout,
    mma_load_b_32x4_to_shared_16x8_layout,
    mma_load_b_32x8_to_shared_16x16_layout,
    mma_load_a_32x16_to_shared_16x32_layout,
    mma_load_b_32x16_to_shared_16x32_layout,
    mma_load_a_32x8_to_shared_16x16_layout,
    shared_16x64_to_mma_32x32_layout_sr_a,
    shared_8x64_to_mma_32x16_layout_sr_b,
    ldmatrix_32x32_to_shared_16x64_layout_a,
    ldmatrix_32x32_to_shared_16x64_layout_b,
    ldmatrix_32x16_to_shared_8x64_layout_b,
)

lift = convert


@dataclass(frozen=True)
class BlockScaleMmaConfig:
    """Static SM120 warp-level block-scale MMA configuration."""

    kind: str
    mma_prefix: str
    atom_k: int
    scale_vec_size: int
    sf_vec_size: int
    scale_type: str
    a_dtype_abbrv: str
    b_dtype_abbrv: str
    accum_dtype: str = T.float32
    active_sfa_threads: int = 16
    active_sfb_threads: int = 8


_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS = {
    ("mxf4nvf4", 4, "ue4m3"): BlockScaleMmaConfig(
        kind="mxf4nvf4",
        mma_prefix="m16n8k64",
        atom_k=64,
        scale_vec_size=4,
        sf_vec_size=16,
        scale_type="ue4m3",
        a_dtype_abbrv="e2m1",
        b_dtype_abbrv="e2m1",
    ),
}


def _get_block_scale_mma_config(kind: str, scale_vec_size: int, scale_type: str) -> BlockScaleMmaConfig:
    key = (kind, scale_vec_size, scale_type)
    if key not in _SUPPORTED_BLOCK_SCALE_MMA_CONFIGS:
        supported = ", ".join(str(k) for k in sorted(_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS))
        raise ValueError(f"Unsupported SM120 block-scale MMA config {key}; supported: {supported}")
    return _SUPPORTED_BLOCK_SCALE_MMA_CONFIGS[key]


class TensorCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    # use lowercase as n_dim can be dynamic
    # the smallest instructions can be m16n8k16, so the n_dim can also be 8
    n_dim = 16
    WARP_SIZE = 32
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int4": "int4",
        "int8": "int8",
        "uint8": "uint8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e4m3fn": "e4m3",
        "float8_e4m3fnuz": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e5m2fnuz": "e5m2",
        "float6_e2m3fn": "e2m3",
        "float6_e3m2fn": "e3m2",
        "float4_e2m1fn": "e2m1",
        "custom[float4_e2m1_unpacked]8": "e2m1",
        "custom[tfloat32]": "tf32",
    }

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first: bool = False
    warp_rows: int = 1
    warp_cols: int = 1

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self._initialize_k_dim(self.a_dtype)
        self._initialize_m_dim(self.a_dtype)
        self._initialize_micro_size(self.M_DIM, self.k_dim)
        self._initialize_local_size(self.M_DIM, self.n_dim, self.k_dim, self.WARP_SIZE)
        self._initialize_abbrev(self.a_dtype, self.b_dtype, accum_dtype)
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_is_m_first(is_m_first)

        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self.thread_var = thread_var

        if self.warp_rows == 0 or self.warp_cols == 0:
            raise ValueError(
                f"Invalid threads configuration for this tile shape, {self.warp_rows} x {self.warp_cols} with threads {self.threads}"
            )

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = min(256 // a_dtype.bits, self.chunk)

    def _initialize_m_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits == 64:
            # FP64 MMA uses m8n8k4; n_dim is set by _initialize_micro_size.
            self.M_DIM = 8

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self._get_dtype_abbrv(a_dtype)
        self.b_dtype_abbrv = self._get_dtype_abbrv(b_dtype)
        self.accum_dtype_abbrv = self._get_dtype_abbrv(accum_dtype)
        if self._should_use_tf32_mma_operand(a_dtype, accum_dtype):
            self.a_dtype_abbrv = "tf32"
        if self._should_use_tf32_mma_operand(b_dtype, accum_dtype):
            self.b_dtype_abbrv = "tf32"

    def _get_dtype_abbrv(self, dtype: str) -> str:
        if "float4_e2m1_unpacked" in dtype:
            return "e2m1"
        if dtype not in self.dtype_abbrv:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return self.dtype_abbrv[dtype]

    @staticmethod
    def _should_use_tf32_mma_operand(dtype: str, accum_dtype: str) -> bool:
        operand_dtype = DataType(dtype)
        accumulator_dtype = DataType(accum_dtype)
        return str(operand_dtype) == "float32" and str(accumulator_dtype) == "float32"

    def _initialize_mma_prefix(self, k_dim: int = 16):
        if k_dim == 4:
            # fp64
            self.mma_prefix = "m8n8k4"
        elif k_dim == 8:
            # typically used for tfloat32
            self.mma_prefix = "m16n8k8"
        elif k_dim == 16:
            # typically used for float16/bfloat16
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            # typically used for int8/fp8
            # sometimes int4/uint4 is also supported
            self.mma_prefix = "m16n8k32"
        elif k_dim == 64:
            # typically used for int4/uint4
            self.mma_prefix = "m16n8k64"
        elif k_dim == 128:
            # typically used for int2/uint2
            self.mma_prefix = "m16n8k128"
        elif k_dim == 256:
            # typically used for uint1
            self.mma_prefix = "m16n8k256"
        else:
            raise ValueError(f"Unsupported k_dim {k_dim}")

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        if k_dim == 4:
            assert m_dim == 8, f"For fp64 MMA, m_dim must be 8, got {m_dim}"
            self.n_dim = 8
            self.micro_size_y = 8
            self.warp_rows = warp_row_tiles // m_dim
            self.warp_cols = warp_col_tiles // 8
        else:
            assert warp_row_tiles >= 16, f"warp_row_tiles must be greater than 16, got {warp_row_tiles}"
            assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
            assert warp_col_tiles >= 8, f"warp_col_tiles must be greater than 8, got {warp_col_tiles}"
            assert warp_col_tiles % 8 == 0, f"warp_col_tiles must be divisible by 8, got {warp_col_tiles}"

            self.warp_rows = warp_row_tiles // m_dim

            if warp_col_tiles % 16 == 0:
                self.n_dim = 16
                self.micro_size_y = 16
                self.warp_cols = warp_col_tiles // 16
            else:
                # must be divisible by 8
                self.n_dim = 8
                self.micro_size_y = 8
                self.warp_cols = warp_col_tiles // 8

        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def _use_fp64_store_index_map(self) -> bool:
        # m8n8 MMA atoms produce two C registers and share the FP64 lane map.
        return DataType(self.accum_dtype).bits == 64 or self.local_size_out == 2

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        if self._use_fp64_store_index_map():
            index_map = IndexMap.from_func(mma_store_index_map_fp64, index_dtype=T.int32)
        else:
            index_map = IndexMap.from_func(mma_store_index_map, index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def extract_thread_binding(self, thread_id: PrimExpr, is_m_first: bool | None = None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        """
        is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
        which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
        Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        # Fast path for fp64: no ldmatrix support, do direct per-lane loads
        a_dtype = self.a_dtype
        if DataType(a_dtype).bits == 64:
            warp_row_tiles = self.warp_row_tiles
            warp_rows = self.warp_rows
            chunk = self.chunk
            micro_size_x = self.micro_size_x  # 8
            micro_size_k = self.micro_size_k  # 4
            local_size_a = self.local_size_a  # 1
            a_transposed = self.a_transposed

            thread_binding = self.get_thread_binding()
            # legalize shared buffer to region
            A_region = self._legalize_to_buffer_region(A_shared_buf)
            A_buf = A_region.buffer
            A_base0 = A_region.region[-2].min
            A_base1 = A_region.region[-1].min
            A_other = [r.min for r in A_region.region[:-2]]

            @T.macro
            def _warp_ld_a_fp64(
                A_local_buf,
                A_shared_buf,
                ki,
                thread_binding,
                rk=0,
            ):
                tx, _, warp_m = self.extract_thread_binding(thread_binding)
                for i in T.serial(warp_rows):
                    wi = warp_m * warp_row_tiles + i * micro_size_x
                    wk = rk * chunk + ki * micro_size_k
                    mi = tx // micro_size_k
                    mk = tx % micro_size_k
                    if a_transposed:
                        A_local_buf[i * local_size_a] = A_buf[tuple(A_other) + (A_base0 + wk + mk, A_base1 + wi + mi)]
                    else:
                        A_local_buf[i * local_size_a] = A_buf[tuple(A_other) + (A_base0 + wi + mi, A_base1 + wk + mk)]

            return _warp_ld_a_fp64(A_local_buf, A_region, ki, thread_binding, rk)

        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(a_dtype).bits != 16 and a_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(a_dtype).bits == 8:
                mma_load_layout = mma_load_a_32x16_to_shared_16x32_layout
            elif DataType(a_dtype).bits == 16:
                mma_load_layout = mma_load_a_32x8_to_shared_16x16_layout
            elif DataType(a_dtype).bits == 32:
                mma_load_layout = mma_load_a_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {a_dtype}")

        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]
        A_stride_last = A_buf.shape[-1]

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_stride_last
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            trans = self.a_transposed

            for i in T.unroll(warp_rows):
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * micro_size_k

                if ldmatrix_available:
                    row_off, col_off = get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed)
                    src_indices = (
                        tuple(A_other) + (A_base0 + wk + row_off, A_base1 + wi + col_off)
                        if a_transposed
                        else tuple(A_other) + (A_base0 + wi + row_off, A_base1 + wk + col_off)
                    )
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        4,
                        T.access_ptr(A_buf[src_indices], "r", extent=8),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=8),
                    )
                else:
                    for j in T.serial(local_size_a):
                        mi, mk = mma_load_layout(tx, j)
                        if a_transposed:
                            A_local_buf[i * local_size_a + j] = A_buf[tuple(A_other) + (A_base0 + wk + mk, A_base1 + wi + mi)]
                        else:
                            A_local_buf[i * local_size_a + j] = A_buf[tuple(A_other) + (A_base0 + wi + mi, A_base1 + wk + mk)]

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        # Fast path for fp64: no ldmatrix support, do direct per-lane loads
        b_dtype = self.b_dtype
        if DataType(b_dtype).bits == 64:
            warp_col_tiles = self.warp_col_tiles
            warp_cols = self.warp_cols
            chunk = self.chunk
            micro_size_y = self.micro_size_y  # 8
            micro_size_k = self.micro_size_k  # 4
            local_size_b = self.local_size_b  # 1
            b_transposed = self.b_transposed
            thread_binding = self.get_thread_binding()

            # legalize shared buffer to region
            B_region = self._legalize_to_buffer_region(B_shared_buf)
            B_buf = B_region.buffer
            B_base0 = B_region.region[-2].min
            B_base1 = B_region.region[-1].min
            B_other = [r.min for r in B_region.region[:-2]]

            @T.macro
            def _warp_ld_b_fp64(
                B_local_buf,
                B_shared_buf,
                ki,
                thread_binding,
                rk=0,
            ):
                tx, warp_n, _ = self.extract_thread_binding(thread_binding)
                for j in T.serial(warp_cols):
                    wi = warp_n * warp_col_tiles + j * micro_size_y
                    wk = rk * chunk + ki * micro_size_k
                    mi = tx // micro_size_k
                    mk = tx % micro_size_k
                    if b_transposed:
                        B_local_buf[j * local_size_b] = B_buf[tuple(B_other) + (B_base0 + wi + mi, B_base1 + wk + mk)]
                    else:
                        B_local_buf[j * local_size_b] = B_buf[tuple(B_other) + (B_base0 + wk + mk, B_base1 + wi + mi)]

            return _warp_ld_b_fp64(B_local_buf, B_region, ki, thread_binding, rk)

        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]
        B_stride_last = B_buf.shape[-1]
        replicate_b = self.n_dim == 16
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(b_dtype).bits != 16 and not b_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(b_dtype).bits == 8:
                mma_load_layout = mma_load_b_32x16_to_shared_16x32_layout
            elif DataType(b_dtype).bits == 16:
                mma_load_layout = mma_load_b_32x8_to_shared_16x16_layout
            elif DataType(b_dtype).bits == 32:
                mma_load_layout = mma_load_b_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {b_dtype}")

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_stride_last
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            trans = not b_transposed

            for i in T.unroll(warp_cols):
                # Assign B_shared_elem
                wi, wk = (
                    warp_n * warp_col_tiles + i * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )

                if ldmatrix_available:
                    num = 4 if replicate_b else 2
                    row_off, col_off = get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed)
                    src_indices = (
                        tuple(B_other) + (B_base0 + wi + row_off, B_base1 + wk + col_off)
                        if b_transposed
                        else tuple(B_other) + (B_base0 + wk + row_off, B_base1 + wi + col_off)
                    )
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        num,
                        T.access_ptr(B_buf[src_indices], "r", extent=2 * num),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=2 * num),
                    )

                else:
                    # load 16x32 data from shared buffer to local buffer
                    # must be transposed.
                    for j in T.serial(local_size_b):
                        mi, mk = mma_load_layout(tx, j)
                        if b_transposed:
                            B_local_buf[i * local_size_b + j] = B_buf[tuple(B_other) + (B_base0 + wi + mi, B_base1 + wk + mk)]
                        else:
                            B_local_buf[i * local_size_b + j] = B_buf[tuple(B_other) + (B_base0 + wk + mk, B_base1 + wi + mi)]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                self.mma_atom(A_local_buf, B_local_buf, C_local_buf, i, j, k_inner)

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    # ---- Atom-level interface ----

    @property
    def mma_num_inst_m(self) -> int:
        """Number of MMA instruction atoms along the M dimension."""
        return self.warp_rows

    @property
    def mma_num_inst_n(self) -> int:
        """Number of MMA instruction atoms along the N dimension."""
        return self.warp_cols

    def mma_atom(
        self,
        A_local_buf: Buffer,
        B_local_buf: Buffer,
        C_local_buf: Buffer,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        k_inner: PrimExpr | int = 0,
    ):
        """Emit a single MMA atom for tile (inst_m_idx, inst_n_idx).

        This is the atomic building block of ``mma()``.  Calling this method
        for every ``(i, j)`` in ``T.grid(mma_num_inst_m, mma_num_inst_n)``
        produces identical TIR to a single ``mma()`` call.

        Parameters
        ----------
        A_local_buf : Buffer
            Fragment buffer for operand A.
        B_local_buf : Buffer
            Fragment buffer for operand B.
        C_local_buf : Buffer
            Accumulator fragment buffer.
        inst_m_idx : int or PrimExpr
            M-dimension atom index (0 .. mma_num_inst_m - 1).
        inst_n_idx : int or PrimExpr
            N-dimension atom index (0 .. mma_num_inst_n - 1).
        k_inner : int or PrimExpr
            K-inner step index used to offset A/B fragments.
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        A_offset = a_local_stride + inst_m_idx * local_size_a
        B_offset = b_local_stride + inst_n_idx * local_size_b
        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out

        @T.macro
        def _atom_mma(A_local_buf, B_local_buf, C_local_buf):
            T.ptx_mma(
                accum_dtype,
                mma_prefix,
                "row",
                "col",
                a_dtype_abbrv,
                b_dtype_abbrv,
                accum_dtype_abbrv,
                A_local_buf.data,
                A_offset,
                B_local_buf.data,
                B_offset,
                C_local_buf.data,
                C_offset,
                T.bool(False),
            )
            if replicate_b:
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    A_offset,
                    B_local_buf.data,
                    B_offset + lift(local_size_b) // 2,
                    C_local_buf.data,
                    C_offset + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _atom_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out

        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, n_dim = self.M_DIM, self.n_dim
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        thread_binding = self.get_thread_binding()
        store_index_map = mma_store_index_map_fp64 if self._use_fp64_store_index_map() else mma_store_index_map

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(store_index_map(tx, local_id))
                        if C_buf_dims == 2:
                            C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * n_dim + col] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]
                        else:
                            C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(store_index_map(tx, local_id))
                        C_buf[
                            (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * n_dim + col,
                        ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )

    def make_mma_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tirx.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        assert matrix in ["A", "B"], "matrix should be either A or B"
        matrix_is_a: bool = matrix == "A"
        matrix_is_b: bool = matrix == "B"
        dtype = self.a_dtype if matrix_is_a else self.b_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed if matrix_is_a else self.b_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        transform_func_sr_b: Callable = None
        if dtype_bits == 32:
            transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
            transform_func_sr_b = shared_16x8_to_mma_32x4_layout_sr_b
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
            transform_func_sr_b = shared_16x16_to_mma_32x8_layout_sr_b
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a
            transform_func_sr_b = shared_16x32_to_mma_32x16_layout_sr_b
        elif dtype_bits == 4:
            transform_func_sr_a = shared_16x64_to_mma_32x32_layout_sr_a
            transform_func_sr_b = shared_8x64_to_mma_32x16_layout_sr_b
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        is_sr_conditions = [False]
        is_sr_conditions.append(matrix_is_a and not transposed)
        is_sr_conditions.append(matrix_is_b and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        # the layout of mma.sync is row.col.
        # so the b matrix expected a transposed basic layout
        transform_func: Callable = None
        if matrix_is_a:
            transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)
        elif matrix_is_b:
            transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_sr_b(j, i)
        else:
            raise ValueError(f"Unsupported matrix {matrix}")

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        if matrix_is_a:
            micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k
        else:
            micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.chunk

        warp_s = warp_rows if matrix_is_a else warp_cols
        warp_r = chunk // micro_size_r
        block_s = block_row_warps if matrix_is_a else block_col_warps
        replicate = block_col_warps if matrix_is_a else block_row_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")
        else:
            warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")

        return block_fragment

    def make_mma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tirx.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        shape = local_buf.shape
        assert is_fragment(local_buf), f"local_buf {local_buf} must be a fragment, but got {local_buf.scope()}"
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)

        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of block_row_warps and block_col_warps are warp_rows and warp_cols
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of warp_i and warp_j are warp_rows and warp_cols
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        """
        Convert Buffer/BufferRegion/BufferLoad to a BufferRegion.

        - Buffer -> full-region BufferRegion covering entire shape
        - BufferRegion -> returned as-is
        - BufferLoad -> best-effort convert via get_buffer_region_from_load;
        if scalar, fall back to 1-sized ranges at given indices
        """
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tirx.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            # Fallback: scalar load -> 1-sized ranges at indices
            mins = [idx for idx in obj.indices]
            ones = [tirx.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")


class TensorCoreIntrinEmitterWithLadderTransform(TensorCoreIntrinEmitter):
    """
    To eliminate Python syntax within TIR Macro.
    With Ladder Transform Plugin.
    """

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        transform_kind_a: int | TransformKind = 0,
        transform_kind_b: int | TransformKind = 0,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
        )
        self._initialize_transform_kind(transform_kind_a, transform_kind_b)

    def _initialize_k_dim(self, a_dtype=T.float16):
        self.k_dim = 256 // DataType(a_dtype).bits

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self._get_dtype_abbrv(a_dtype)
        self.b_dtype_abbrv = self._get_dtype_abbrv(b_dtype)
        self.accum_dtype_abbrv = self._get_dtype_abbrv(accum_dtype)

    def _initialize_mma_prefix(self, k_dim=16):
        if k_dim == 16:
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            self.mma_prefix = "m16n8k32"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = self.n_dim
        self.micro_size_k = k_dim

    def _initialize_transform_kind(self, transform_kind_a, transform_kind_b):
        if isinstance(transform_kind_a, int):
            self.transform_kind_a = TransformKind(transform_kind_a)
        elif isinstance(transform_kind_a, TransformKind):
            self.transform_kind_a = transform_kind_a
        else:
            raise ValueError("Unsupported transform_kind_a")

        if isinstance(transform_kind_b, int):
            self.transform_kind_b = TransformKind(transform_kind_b)
        elif isinstance(transform_kind_b, TransformKind):
            self.transform_kind_b = transform_kind_b
        else:
            raise ValueError("Unsupported transform_kind_b")

        assert transform_kind_a in [0, 1, 2, 3], "Input transform stage should be 0, 1, 2, or 3"
        assert transform_kind_b in [0, 1, 2, 3], "Weight transform stage should be 0, 1, 2, or 3"

    def ldmatrix_a(self, A_local_buf, A_shared_buf, ki, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed
        transform_kind_a = self.transform_kind_a

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_shared_buf.shape[-1]
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if transform_kind_a == TransformKind.NonTransform:
                for i in T.serial(warp_rows):
                    row_off, col_off = get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed)
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(
                            A_shared_buf[
                                warp_m * warp_row_tiles + i * micro_size_x + row_off,
                                rk * chunk + ki * micro_size_k + col_off,
                            ],
                            "r",
                            extent=8,
                        ),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=8),
                    )
            elif transform_kind_a == TransformKind.InterWarpTransform:
                for i in T.serial(warp_rows):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_x,
                        (rj) // micro_size_k,
                        (ri) % micro_size_x,
                        (rj) % micro_size_k,
                    )
                    row_off, col_off = get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed)
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(A_shared_buf[ni, nj, nii + row_off, njj + col_off], "r", extent=8),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=8),
                    )
            elif transform_kind_a == TransformKind.IntraWarpTransform:
                for i in T.serial(warp_rows):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_x,
                        (rj) // micro_size_k,
                        (ri) % micro_size_x,
                        (rj) % micro_size_k,
                    )
                    row_off = (tx * local_size_a) // stride
                    col_off = (tx * local_size_a) % stride
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(A_shared_buf[ni, nj, nii + row_off, njj + col_off], "r", extent=8),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=8),
                    )
            elif transform_kind_a == TransformKind.LDMatrixTransform:
                for j in T.serial(warp_rows):
                    for local_id in T.vectorized(local_size_a):
                        # Assign A_shared_elem
                        ri, rj = (
                            warp_m * warp_rows + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (tx * local_size_a + local_id) // micro_size_k, (tx * local_size_a + local_id) % (micro_size_k)
                        A_local_buf[j * local_size_a + local_id] = A_shared_buf[ri, rj, rii, rjj]
            else:
                raise ValueError("Unsupported TransformKind for Input A")

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf, B_shared_buf, ki, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        transform_kind_b = self.transform_kind_b
        b_transposed = self.b_transposed
        num_elems_per_byte = self.num_elems_per_byte

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            if transform_kind_b == TransformKind.NonTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    row_off, col_off = get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed)
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(B_shared_buf[ri + row_off, rj + col_off], "r", extent=8),
                        T.access_ptr(B_local_buf[j * local_size_b], "w", extent=8),
                    )
            elif transform_kind_b == TransformKind.InterWarpTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    row_off, col_off = get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed)
                    T.ptx_ldmatrix(
                        T.bool(False),  # TODO(lei): should be optimized
                        4,
                        T.access_ptr(B_shared_buf[ni, nj, nii + row_off, njj + col_off], "r", extent=8),
                        T.access_ptr(B_local_buf[j * local_size_b], "w", extent=8),
                    )
            elif transform_kind_b == TransformKind.IntraWarpTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    row_off = (tx * local_size_b) // stride
                    col_off = (tx * local_size_b) % stride
                    T.ptx_ldmatrix(
                        T.bool(False),  # TODO(lei): should be optimized
                        4,
                        T.access_ptr(B_shared_buf[ni, nj, nii + row_off, njj + col_off], "r", extent=8),
                        T.access_ptr(B_local_buf[j * local_size_b], "w", extent=8),
                    )
            elif transform_kind_b == TransformKind.LDMatrixTransform:
                local_size_dequantize = local_size_b // num_elems_per_byte
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_dequantize):
                        # Assign B_shared_elem
                        ri, rj = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (
                            (tx * local_size_dequantize + local_id) // (micro_size_k // num_elems_per_byte),
                            (tx * local_size_dequantize + local_id) % (micro_size_k // num_elems_per_byte),
                        )
                        B_local_buf[j * local_size_dequantize + local_id] = B_shared_buf[ri, rj, rii, rjj]
            else:
                raise ValueError("Unsupported TransformKind for Input B")

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self, A_local_buf, B_local_buf, C_local_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)


class TensorCoreIntrinEmitterWithBlockScale(TensorCoreIntrinEmitter):
    """SM120 warp-level block-scale MMA emitter.

    The emitter keeps scale-factor storage explicit, matching TileLang's
    TCGEN05 block-scaled style while targeting warp-level ``mma.sync``.
    """

    def __init__(
        self,
        a_dtype: str = T.float4_e2m1fn,
        b_dtype: str = T.float4_e2m1fn,
        accum_dtype: str = T.float32,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 32,
        warp_col_tiles: int = 32,
        chunk: int = 64,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
        kind: str = "mxf4nvf4",
        scale_vec_size: int = 4,
        stype: str = "ue4m3",
    ):
        self.block_scale_config = _get_block_scale_mma_config(kind, scale_vec_size, stype)
        a_dtype_abbrv = self._get_dtype_abbrv(str(a_dtype))
        b_dtype_abbrv = self._get_dtype_abbrv(str(b_dtype))
        if (
            a_dtype_abbrv != self.block_scale_config.a_dtype_abbrv
            or b_dtype_abbrv != self.block_scale_config.b_dtype_abbrv
            or str(accum_dtype) != self.block_scale_config.accum_dtype
        ):
            raise ValueError(
                f"{self.block_scale_config.kind} expects a_dtype={self.block_scale_config.a_dtype_abbrv}, "
                f"b_dtype={self.block_scale_config.b_dtype_abbrv}, "
                f"accum_dtype={self.block_scale_config.accum_dtype}; "
                f"got a_dtype={a_dtype}, b_dtype={b_dtype}, accum_dtype={accum_dtype}"
            )
        self.kind = self.block_scale_config.kind
        self.scale_vec_size = self.block_scale_config.scale_vec_size
        self.stype = self.block_scale_config.scale_type
        self.sf_vec_size = self.block_scale_config.sf_vec_size
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )

    def _initialize_k_dim(self, a_dtype=T.float16):
        self.k_dim = self.block_scale_config.atom_k

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.block_scale_config.a_dtype_abbrv
        self.b_dtype_abbrv = self.block_scale_config.b_dtype_abbrv
        self.accum_dtype_abbrv = self._get_dtype_abbrv(accum_dtype)

    def _initialize_mma_prefix(self, k_dim: int = 16):
        self.mma_prefix = self.block_scale_config.mma_prefix

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed

        thread_binding = self.get_thread_binding()
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        @T.macro
        def _warp_ld_a_e2m1(A_local_buf, A_shared_buf, ki, thread_binding, rk=0):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            for i in T.unroll(warp_rows):
                wi = warp_m * warp_row_tiles + i * micro_size_x
                wk = rk * chunk + ki * micro_size_k
                row_off, col_off = ldmatrix_32x32_to_shared_16x64_layout_a(tx)
                if a_transposed:
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(
                            A_buf[tuple(A_other) + (A_base0 + wk + row_off, A_base1 + wi + col_off)],
                            "r",
                            extent=local_size_a,
                        ),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=local_size_a),
                    )
                else:
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(
                            A_buf[tuple(A_other) + (A_base0 + wi + row_off, A_base1 + wk + col_off)],
                            "r",
                            extent=local_size_a,
                        ),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=local_size_a),
                    )

        return _warp_ld_a_e2m1(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        replicate_b = self.n_dim == 16

        thread_binding = self.get_thread_binding()
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        @T.macro
        def _warp_ld_b_e2m1(B_local_buf, B_shared_buf, ki, thread_binding, rk=0):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            for i in T.unroll(warp_cols):
                wi = warp_n * warp_col_tiles + i * micro_size_y
                wk = rk * chunk + ki * micro_size_k
                if replicate_b:
                    row_off, col_off = ldmatrix_32x32_to_shared_16x64_layout_b(tx)
                else:
                    row_off, col_off = ldmatrix_32x16_to_shared_8x64_layout_b(tx)
                if b_transposed:
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4 if replicate_b else 2,
                        T.access_ptr(
                            B_buf[tuple(B_other) + (B_base0 + wi + row_off, B_base1 + wk + col_off)],
                            "r",
                            extent=local_size_b,
                        ),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=local_size_b),
                    )
                else:
                    T.ptx_ldmatrix(
                        T.bool(True),
                        4 if replicate_b else 2,
                        T.access_ptr(
                            B_buf[tuple(B_other) + (B_base0 + wk + row_off, B_base1 + wi + col_off)],
                            "r",
                            extent=local_size_b,
                        ),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=local_size_b),
                    )

        return _warp_ld_b_e2m1(B_local_buf, B_region, ki, thread_binding, rk)

    def ldmatrix_b_atom(
        self,
        B_local_buf: Buffer,
        B_shared_buf: Buffer | BufferRegion,
        ki: PrimExpr,
        inst_n_idx: PrimExpr | int,
        rk: PrimExpr | None = 0,
    ):
        warp_col_tiles = self.warp_col_tiles
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        replicate_b = self.n_dim == 16

        thread_binding = self.get_thread_binding()
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        @T.macro
        def _warp_ld_b_atom_e2m1(B_local_buf, B_shared_buf, ki, thread_binding, rk=0):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            wi = warp_n * warp_col_tiles + inst_n_idx * micro_size_y
            wk = rk * chunk + ki * micro_size_k
            if replicate_b:
                row_off, col_off = ldmatrix_32x32_to_shared_16x64_layout_b(tx)
            else:
                row_off, col_off = ldmatrix_32x16_to_shared_8x64_layout_b(tx)
            if b_transposed:
                T.ptx_ldmatrix(
                    T.bool(False),
                    4 if replicate_b else 2,
                    T.access_ptr(
                        B_buf[tuple(B_other) + (B_base0 + wi + row_off, B_base1 + wk + col_off)],
                        "r",
                        extent=local_size_b,
                    ),
                    T.access_ptr(B_local_buf[0], "w", extent=local_size_b),
                )
            else:
                T.ptx_ldmatrix(
                    T.bool(True),
                    4 if replicate_b else 2,
                    T.access_ptr(
                        B_buf[tuple(B_other) + (B_base0 + wk + row_off, B_base1 + wi + col_off)],
                        "r",
                        extent=local_size_b,
                    ),
                    T.access_ptr(B_local_buf[0], "w", extent=local_size_b),
                )

        return _warp_ld_b_atom_e2m1(B_local_buf, B_region, ki, thread_binding, rk)

    def _scale_region_parts(self, scale_buf: Buffer | BufferRegion):
        if isinstance(scale_buf, BufferRegion):
            scale_region = scale_buf
        elif isinstance(scale_buf, Buffer):
            scale_region = self._legalize_to_buffer_region(scale_buf)
        else:
            raise ValueError(f"Unsupported scale buffer type: {type(scale_buf)}")
        return (
            scale_region.buffer,
            [r.min for r in scale_region.region[:-2]],
            scale_region.region[-2].min,
            scale_region.region[-1].min,
        )

    @staticmethod
    def _sfa_row_in_atom(tx: PrimExpr):
        # CUTLASS SFALayout for k64 uses ((2,2,8),64), stride ((8,0,1),16).
        # With K-major flattening, the M coordinate is 8 * (lane % 2) + lane // 4.
        return 8 * (tx % 2) + (tx // 4)

    @staticmethod
    def _sfb_col_in_atom(tx: PrimExpr):
        # CUTLASS SFBLayout for k64 uses ((4,8),64), stride ((0,1),8), so the
        # logical N coordinate is lane // 4 with broadcast across four groups.
        return tx // 4

    def _scale_word_k(self, k_start: PrimExpr, ki: PrimExpr, sf_granularity_k: int):
        packed_word_k = int(sf_granularity_k) * 4
        if packed_word_k != self.sf_vec_size * 4:
            raise ValueError(
                f"{self.kind} expects packed scale words covering {self.sf_vec_size * 4} K elements, "
                f"got sf_granularity_k={sf_granularity_k}"
            )
        _k_start = tvm.tirx.const(k_start, "int32") if isinstance(k_start, int) else k_start
        return (_k_start + self.micro_size_k * ki) // packed_word_k

    def mma(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        replicate_b = self.n_dim == 16
        if sf_layout not in ("rowmajor", "cutlass_128x4", "blockscaled_chunk_kmajor"):
            raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")

        def _cutlass_sf_word(idx, word_k):
            return T.call_pure_extern("int32", "tl::detail::sm120_blockscaled_chunk_kmajor_sf_word", idx, word_k)

        @T.macro
        def _warp_mma_block_scale(A_local_buf, B_local_buf, C_local_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            for i, j in T.grid(warp_rows, warp_cols):
                scale_m = warp_m * warp_row_tiles + i * micro_size_x + sfa_row
                scale_n = warp_n * warp_col_tiles + j * micro_size_y + sfb_col
                if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                    scale_a_word = _cutlass_sf_word(scale_m, scale_a_word_k)
                    scale_b_word = _cutlass_sf_word(scale_n, scale_b_word_k)
                    scale_a_ptr = T.access_ptr(
                        SFA_data[tuple(SFA_other) + (SFA_base_m + scale_a_word // 4, SFA_base_k + scale_a_word % 4)],
                        "r",
                    )
                    scale_b_ptr = T.access_ptr(
                        SFB_data[tuple(SFB_other) + (SFB_base_n + scale_b_word // 4, SFB_base_k + scale_b_word % 4)],
                        "r",
                    )
                else:
                    scale_a_ptr = T.access_ptr(
                        SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)],
                        "r",
                    )
                    scale_b_ptr = T.access_ptr(
                        SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_k)],
                        "r",
                    )
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    scale_a_ptr,
                    scale_b_ptr,
                )
                if replicate_b:
                    if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                        scale_b_rep_n = scale_n + 8
                        scale_b_rep_word = _cutlass_sf_word(scale_b_rep_n, scale_b_word_k)
                        scale_b_rep_ptr = T.access_ptr(
                            SFB_data[tuple(SFB_other) + (SFB_base_n + scale_b_rep_word // 4, SFB_base_k + scale_b_rep_word % 4)],
                            "r",
                        )
                    else:
                        scale_b_rep_ptr = T.access_ptr(
                            SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n + 8, SFB_base_k + scale_b_word_k)],
                            "r",
                        )
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_local_buf.data,
                        j * local_size_b + lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                        scale_a_ptr,
                        scale_b_rep_ptr,
                    )

        return _warp_mma_block_scale(A_local_buf, B_local_buf, C_local_buf, SFA_data, SFB_data, thread_binding)

    def ldscale(
        self,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        replicate_b = self.n_dim == 16
        if sf_layout not in ("rowmajor", "cutlass_128x4", "blockscaled_chunk_kmajor"):
            raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")

        def _cutlass_sf_word(idx, word_k):
            return T.call_pure_extern("int32", "tl::detail::sm120_blockscaled_chunk_kmajor_sf_word", idx, word_k)

        @T.macro
        def _warp_ldscale_block_scale(SFA_local_buf, SFB_local_buf, SFB_rep_local_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            for i in T.unroll(warp_rows):
                scale_m = warp_m * warp_row_tiles + i * micro_size_x + sfa_row
                if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                    scale_a_word = _cutlass_sf_word(scale_m, scale_a_word_k)
                    SFA_local_buf[i] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_a_word // 4, SFA_base_k + scale_a_word % 4)]
                else:
                    SFA_local_buf[i] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)]
            for j in T.unroll(warp_cols):
                scale_n = warp_n * warp_col_tiles + j * micro_size_y + sfb_col
                if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                    scale_b_word = _cutlass_sf_word(scale_n, scale_b_word_k)
                    SFB_local_buf[j] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_b_word // 4, SFB_base_k + scale_b_word % 4)]
                else:
                    SFB_local_buf[j] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_k)]
                if replicate_b:
                    if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                        scale_b_rep_n = scale_n + 8
                        scale_b_rep_word = _cutlass_sf_word(scale_b_rep_n, scale_b_word_k)
                        SFB_rep_local_buf[j] = SFB_data[
                            tuple(SFB_other) + (SFB_base_n + scale_b_rep_word // 4, SFB_base_k + scale_b_rep_word % 4)
                        ]
                    else:
                        SFB_rep_local_buf[j] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n + 8, SFB_base_k + scale_b_word_k)]

        return _warp_ldscale_block_scale(
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def ldscale_fragment(
        self,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        """Load SM120 block-scale fragments into local registers.

        This is currently a thin wrapper over the existing scale-word load.
        The separate name gives the SM120 MMA lowering a stable hook for a
        CUTLASS-like scale-fragment copy path.
        """
        return self.ldscale(
            SFA_fragment_buf,
            SFB_fragment_buf,
            SFB_rep_fragment_buf,
            SFA_buf,
            SFB_buf,
            ki=ki,
            k_start=k_start,
            sf_a_granularity_k=sf_a_granularity_k,
            sf_b_granularity_k=sf_b_granularity_k,
            sf_layout=sf_layout,
        )

    def ldscale_fragment_b_owner(
        self,
        SFA_fragment_buf,
        SFB_owner_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        """Load SFA normally and pack four B scale owners across a lane quad."""

        warp_rows = self.warp_rows
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        if sf_layout != "rowmajor":
            raise ValueError("ldscale_fragment_b_owner currently supports rowmajor scale layout only")
        if self.n_dim != 16 or int(self.warp_cols) != 2:
            raise ValueError("ldscale_fragment_b_owner currently requires replicated B with warp_cols=2")

        @T.macro
        def _warp_ldscale_b_owner(SFA_fragment_buf, SFB_owner_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            qlane = tx & 3
            for i in T.unroll(warp_rows):
                scale_m = warp_m * warp_row_tiles + i * micro_size_x + sfa_row
                SFA_fragment_buf[i] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)]
            SFB_owner_buf[0] = SFB_data[
                tuple(SFB_other) + (SFB_base_n + warp_n * warp_col_tiles + sfb_col + qlane * 8, SFB_base_k + scale_b_word_k)
            ]

        return _warp_ldscale_b_owner(
            SFA_fragment_buf,
            SFB_owner_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def ldscale_fragment_ab_owner(
        self,
        SFA_owner_buf,
        SFB_owner_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        """Load two A scale owners and one B scale owner for full-tile SM120."""

        warp_rows = self.warp_rows
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        if sf_layout != "rowmajor":
            raise ValueError("ldscale_fragment_ab_owner currently supports rowmajor scale layout only")
        if int(warp_rows) != 4 or self.n_dim != 16 or int(self.warp_cols) != 2:
            raise ValueError("ldscale_fragment_ab_owner currently requires fulltile warp_rows=4, warp_cols=2")

        @T.macro
        def _warp_ldscale_ab_owner(SFA_owner_buf, SFB_owner_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            qlane = tx & 3
            a_owner_in_pair = qlane >> 1
            for g in T.unroll(2):
                scale_m = warp_m * warp_row_tiles + g * (2 * micro_size_x) + a_owner_in_pair * micro_size_x + sfa_row
                SFA_owner_buf[g] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)]
            SFB_owner_buf[0] = SFB_data[
                tuple(SFB_other) + (SFB_base_n + warp_n * warp_col_tiles + sfb_col + qlane * 8, SFB_base_k + scale_b_word_k)
            ]

        return _warp_ldscale_ab_owner(
            SFA_owner_buf,
            SFB_owner_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def ldscale_fragment_ab_owner_wide(
        self,
        SFA_owner_buf,
        SFB_owner_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        """Load owner scale words for full-tile SM120 with warp_cols=4.

        Each owner word is later selected by the block-scale MMA
        `{byte_id, thread_id}` operands.  For a 4x4 warp-atom tile this reduces
        per-kblock scale loads from 4 A + 4 B + 4 B-rep words to 2 A-owner +
        2 B-owner words.
        """

        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        if int(warp_rows) != 4 or self.n_dim != 16 or int(warp_cols) != 4:
            raise ValueError("ldscale_fragment_ab_owner_wide requires warp_rows=4, warp_cols=4")
        if sf_layout not in ("rowmajor", "cutlass_128x4", "blockscaled_chunk_kmajor"):
            raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")

        def _cutlass_sf_word(idx, word_k):
            return T.call_pure_extern("int32", "tl::detail::sm120_blockscaled_chunk_kmajor_sf_word", idx, word_k)

        @T.macro
        def _warp_ldscale_ab_owner_wide(SFA_owner_buf, SFB_owner_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            qlane = tx & 3
            a_owner_in_pair = qlane >> 1
            for g in T.unroll(2):
                scale_m = warp_m * warp_row_tiles + g * (2 * micro_size_x) + a_owner_in_pair * micro_size_x + sfa_row
                if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                    scale_a_word = _cutlass_sf_word(scale_m, scale_a_word_k)
                    SFA_owner_buf[g] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_a_word // 4, SFA_base_k + scale_a_word % 4)]
                else:
                    SFA_owner_buf[g] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)]
            for g in T.unroll(2):
                scale_n = warp_n * warp_col_tiles + g * (2 * micro_size_y) + sfb_col + qlane * 8
                if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                    scale_b_word = _cutlass_sf_word(scale_n, scale_b_word_k)
                    SFB_owner_buf[g] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_b_word // 4, SFB_base_k + scale_b_word % 4)]
                else:
                    SFB_owner_buf[g] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_k)]

        return _warp_ldscale_ab_owner_wide(
            SFA_owner_buf,
            SFB_owner_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def ldscale_fragment_kpack(
        self,
        SFA_pack_buf,
        SFB_pack_buf,
        SFB_rep_pack_buf,
        SFA_buf,
        SFB_buf,
        num_k_blocks: int,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
    ):
        """Load a small K-block pack of SM120 scale fragments.

        The retained shared scale layout has contiguous uint32 scale words along
        the K-word dimension.  This private loader keeps that layout and tries to
        expose the four K words as a vectorized copy into registers.
        """

        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_start = self._scale_word_k(k_start, 0, sf_a_granularity_k)
        scale_b_word_start = self._scale_word_k(k_start, 0, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_ldscale_kpack(SFA_pack_buf, SFB_pack_buf, SFB_rep_pack_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            for i in T.unroll(warp_rows):
                scale_m = warp_m * warp_row_tiles + i * micro_size_x + sfa_row
                for kk in T.vectorized(num_k_blocks):
                    SFA_pack_buf[i * num_k_blocks + kk] = SFA_data[
                        tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_start + kk)
                    ]
            for j in T.unroll(warp_cols):
                scale_n = warp_n * warp_col_tiles + j * micro_size_y + sfb_col
                for kk in T.vectorized(num_k_blocks):
                    SFB_pack_buf[j * num_k_blocks + kk] = SFB_data[
                        tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_start + kk)
                    ]
                    if replicate_b:
                        SFB_rep_pack_buf[j * num_k_blocks + kk] = SFB_data[
                            tuple(SFB_other) + (SFB_base_n + scale_n + 8, SFB_base_k + scale_b_word_start + kk)
                        ]

        return _warp_ldscale_kpack(
            SFA_pack_buf,
            SFB_pack_buf,
            SFB_rep_pack_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def mma_backend_kblock_fulltile(
        self,
        A_shared_buf: Buffer | BufferRegion,
        B_shared_buf: Buffer | BufferRegion,
        C_local_buf: Buffer,
        SFA_buf: Buffer | BufferRegion,
        SFB_buf: Buffer | BufferRegion,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        """Emit one SM120 NVF4 K=64 full-tile helper call.

        This path deliberately keeps A/B/SFA/SFB fragments out of TIR local
        buffers.  The CUDA helper owns the ldmatrix loads, scale loads, and the
        full set of m16n8k64 block-scaled MMA atoms for one K block.
        """
        if int(self.warp_rows) != 4 or int(self.warp_cols) != 4:
            raise ValueError("sm120 backend fulltile helper requires warp_rows=4 and warp_cols=4")
        if self.n_dim != 16:
            raise ValueError("sm120 backend fulltile helper requires replicated B n_dim=16")
        if not self.b_transposed:
            raise ValueError("sm120 backend fulltile helper currently requires transpose_B=True")

        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        sf_vec_size = self.sf_vec_size
        a_transposed = self.a_transposed

        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        if sf_layout not in ("rowmajor", "cutlass_128x4", "blockscaled_chunk_kmajor"):
            raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        @T.macro
        def _warp_backend_kblock(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        ):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            a_row_off, a_col_off = ldmatrix_32x32_to_shared_16x64_layout_a(tx)
            b_row_off, b_col_off = ldmatrix_32x32_to_shared_16x64_layout_b(tx)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            wk = ki * micro_size_k

            if a_transposed:
                a0 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a1 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + micro_size_x + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a2 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + 2 * micro_size_x + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a3 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + 3 * micro_size_x + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
            else:
                a0 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a1 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + micro_size_x + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a2 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + 2 * micro_size_x + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a3 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + 3 * micro_size_x + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )

            b0 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            b1 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + micro_size_y + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            b2 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + 2 * micro_size_y + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            b3 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + 3 * micro_size_y + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )

            scale_m0 = warp_m * warp_row_tiles + sfa_row
            scale_m1 = warp_m * warp_row_tiles + micro_size_x + sfa_row
            scale_m2 = warp_m * warp_row_tiles + 2 * micro_size_x + sfa_row
            scale_m3 = warp_m * warp_row_tiles + 3 * micro_size_x + sfa_row
            scale_n0 = warp_n * warp_col_tiles + sfb_col
            scale_n1 = warp_n * warp_col_tiles + micro_size_y + sfb_col
            scale_n2 = warp_n * warp_col_tiles + 2 * micro_size_y + sfb_col
            scale_n3 = warp_n * warp_col_tiles + 3 * micro_size_y + sfb_col

            if sf_layout in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
                sfa0 = T.access_ptr(SFA_data[tuple(SFA_other) + (SFA_base_m, SFA_base_k)], "r")
                sfa1 = sfa0
                sfa2 = sfa0
                sfa3 = sfa0
                sfb0 = T.access_ptr(SFB_data[tuple(SFB_other) + (SFB_base_n, SFB_base_k)], "r")
                sfb1 = sfb0
                sfb2 = sfb0
                sfb3 = sfb0
                sfb_rep0 = sfb0
                sfb_rep1 = sfb0
                sfb_rep2 = sfb0
                sfb_rep3 = sfb0
            else:
                sfa0 = T.access_ptr(
                    SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m0, SFA_base_k + scale_a_word_k)],
                    "r",
                )
                sfa1 = T.access_ptr(
                    SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m1, SFA_base_k + scale_a_word_k)],
                    "r",
                )
                sfa2 = T.access_ptr(
                    SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m2, SFA_base_k + scale_a_word_k)],
                    "r",
                )
                sfa3 = T.access_ptr(
                    SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m3, SFA_base_k + scale_a_word_k)],
                    "r",
                )
                sfb0 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n0, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb1 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n1, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb2 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n2, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb3 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n3, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb_rep0 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n0 + 8, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb_rep1 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n1 + 8, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb_rep2 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n2 + 8, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                sfb_rep3 = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n3 + 8, SFB_base_k + scale_b_word_k)],
                    "r",
                )

            T.sm120_mma_blockscaled_kblock_fulltile(
                C_local_buf.data,
                0,
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                b2,
                b3,
                sfa0,
                sfa1,
                sfa2,
                sfa3,
                sfb0,
                sfb1,
                sfb2,
                sfb3,
                sfb_rep0,
                sfb_rep1,
                sfb_rep2,
                sfb_rep3,
                ki,
            )

        return _warp_backend_kblock(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def mma_backend_kblock_fulltile_ab_owner_wide(
        self,
        A_shared_buf: Buffer | BufferRegion,
        B_shared_buf: Buffer | BufferRegion,
        C_local_buf: Buffer,
        SFA_buf: Buffer | BufferRegion,
        SFB_buf: Buffer | BufferRegion,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
        backend_op: str = "ab_owner_wide",
    ):
        """Emit one SM120 K=64 helper call with backend-owned scale owners.

        This is the backend equivalent of the earlier owner-wide TIR probe.  It
        passes only packed A/B source pointers and SFA/SFB base pointers to C++,
        so the owner words are scalar helper locals rather than TIR local arrays.
        """
        if backend_op not in ("ab_owner_wide", "afull_bpanel_owner_wide"):
            raise ValueError(f"Unsupported SM120 backend owner-wide op: {backend_op}")
        if int(self.warp_rows) != 4 or int(self.warp_cols) != 4:
            raise ValueError("sm120 backend owner-wide helper requires warp_rows=4 and warp_cols=4")
        if self.n_dim != 16:
            raise ValueError("sm120 backend owner-wide helper requires replicated B n_dim=16")
        if not self.b_transposed:
            raise ValueError("sm120 backend owner-wide helper currently requires transpose_B=True")
        if sf_layout not in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
            raise ValueError(
                "sm120 backend owner-wide helper currently requires sf_layout='blockscaled_chunk_kmajor' or legacy 'cutlass_128x4'"
            )

        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        a_transposed = self.a_transposed
        thread_binding = self.get_thread_binding()

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        @T.macro
        def _warp_backend_kblock_owner_wide(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        ):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            a_row_off, a_col_off = ldmatrix_32x32_to_shared_16x64_layout_a(tx)
            b_row_off, b_col_off = ldmatrix_32x32_to_shared_16x64_layout_b(tx)
            wk = ki * micro_size_k

            if a_transposed:
                a0 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a1 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + micro_size_x + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a2 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + 2 * micro_size_x + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a3 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + wk + a_row_off, A_base1 + warp_m * warp_row_tiles + 3 * micro_size_x + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
            else:
                a0 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a1 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + micro_size_x + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a2 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + 2 * micro_size_x + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )
                a3 = T.access_ptr(
                    A_buf[tuple(A_other) + (A_base0 + warp_m * warp_row_tiles + 3 * micro_size_x + a_row_off, A_base1 + wk + a_col_off)],
                    "r",
                    extent=local_size_a,
                )

            b0 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            b1 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + micro_size_y + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            b2 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + 2 * micro_size_y + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            b3 = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0 + warp_n * warp_col_tiles + 3 * micro_size_y + b_row_off, B_base1 + wk + b_col_off)],
                "r",
                extent=local_size_b,
            )
            sfa_base = T.access_ptr(SFA_data[tuple(SFA_other) + (SFA_base_m, SFA_base_k)], "r", extent=1)
            sfb_base = T.access_ptr(SFB_data[tuple(SFB_other) + (SFB_base_n, SFB_base_k)], "r", extent=1)

            if backend_op == "afull_bpanel_owner_wide":
                T.sm120_mma_blockscaled_kblock_fulltile_afull_bpanel_owner_wide(
                    C_local_buf.data,
                    0,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    b2,
                    b3,
                    sfa_base,
                    sfb_base,
                    ki,
                )
            else:
                T.sm120_mma_blockscaled_kblock_fulltile_ab_owner_wide(
                    C_local_buf.data,
                    0,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    b2,
                    b3,
                    sfa_base,
                    sfb_base,
                    ki,
                )

        return _warp_backend_kblock_owner_wide(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def mma_backend_cute_consumer_bridge(
        self,
        A_shared_buf: Buffer | BufferRegion,
        B_shared_buf: Buffer | BufferRegion,
        C_local_buf: Buffer,
        SFA_buf: Buffer | BufferRegion,
        SFB_buf: Buffer | BufferRegion,
        ki: PrimExpr = 0,
    ):
        """Emit a private SM120 full shared-tile consumer bridge call.

        Unlike mma_backend_kblock_fulltile, this hook passes whole A/B/SFA/SFB
        shared tile bases to CUDA.  It is the lowering contract needed for a
        backend-owned blockscaled operand package.
        """
        if int(self.warp_rows) != 4 or int(self.warp_cols) != 4:
            raise ValueError("sm120 CuTe consumer bridge requires warp_rows=4 and warp_cols=4")
        if self.n_dim != 16:
            raise ValueError("sm120 CuTe consumer bridge requires replicated B n_dim=16")
        if not self.b_transposed:
            raise ValueError("sm120 CuTe consumer bridge currently requires transpose_B=True")

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        @T.macro
        def _warp_backend_cute_consumer_bridge(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
        ):
            a_base = T.access_ptr(
                A_buf[tuple(A_other) + (A_base0, A_base1)],
                "r",
                extent=1,
            )
            b_base = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0, B_base1)],
                "r",
                extent=1,
            )
            sfa_base = T.access_ptr(
                SFA_data[tuple(SFA_other) + (SFA_base_m, SFA_base_k)],
                "r",
                extent=1,
            )
            sfb_base = T.access_ptr(
                SFB_data[tuple(SFB_other) + (SFB_base_n, SFB_base_k)],
                "r",
                extent=1,
            )
            T.sm120_mma_blockscaled_cute_consumer_bridge(
                C_local_buf.data,
                0,
                a_base,
                b_base,
                sfa_base,
                sfb_base,
                ki,
            )

        return _warp_backend_cute_consumer_bridge(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
        )

    def mma_backend_kblock_fulltile_package_pingpong(
        self,
        A_shared_buf: Buffer | BufferRegion,
        B_shared_buf: Buffer | BufferRegion,
        C_local_buf: Buffer,
        SFA_buf: Buffer | BufferRegion,
        SFB_buf: Buffer | BufferRegion,
        sf_layout: str = "rowmajor",
    ):
        """Emit one backend-owned SM120 package lifecycle helper call.

        The CUDA helper receives full A/B/SFA/SFB shared K-stage bases and owns
        the copy_kblock_package(next) -> gemm_kblock_package(current) schedule.
        """
        if int(self.warp_rows) != 4 or int(self.warp_cols) != 4:
            raise ValueError("sm120 package pingpong helper requires warp_rows=4 and warp_cols=4")
        if self.n_dim != 16:
            raise ValueError("sm120 package pingpong helper requires replicated B n_dim=16")
        if not self.b_transposed:
            raise ValueError("sm120 package pingpong helper currently requires transpose_B=True")
        if sf_layout not in ("cutlass_128x4", "blockscaled_chunk_kmajor"):
            raise ValueError(
                "sm120 package pingpong helper currently requires sf_layout='blockscaled_chunk_kmajor' or legacy 'cutlass_128x4'"
            )

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        @T.macro
        def _warp_backend_package_pingpong(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
        ):
            a_base = T.access_ptr(
                A_buf[tuple(A_other) + (A_base0, A_base1)],
                "r",
                extent=1,
            )
            b_base = T.access_ptr(
                B_buf[tuple(B_other) + (B_base0, B_base1)],
                "r",
                extent=1,
            )
            sfa_base = T.access_ptr(
                SFA_data[tuple(SFA_other) + (SFA_base_m, SFA_base_k)],
                "r",
                extent=1,
            )
            sfb_base = T.access_ptr(
                SFB_data[tuple(SFB_other) + (SFB_base_n, SFB_base_k)],
                "r",
                extent=1,
            )
            T.sm120_mma_blockscaled_kblock_fulltile_package_pingpong(
                C_local_buf.data,
                0,
                a_base,
                b_base,
                sfa_base,
                sfb_base,
            )

        return _warp_backend_package_pingpong(
            A_buf,
            B_buf,
            C_local_buf,
            SFA_data,
            SFB_data,
        )

    def mma_with_scale_fragments(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
    ):
        """Issue SM120 block-scaled MMA using prefetched scale fragments."""
        return self.mma_with_prefetched_scales(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_fragment_buf,
            SFB_fragment_buf,
            SFB_rep_fragment_buf,
        )

    def mma_atom_with_scale_fragments(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
    ):
        """Issue one SM120 block-scaled MMA atom with scale fragments."""
        return self.mma_atom_with_prefetched_scales(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_fragment_buf,
            SFB_fragment_buf,
            SFB_rep_fragment_buf,
            inst_m_idx,
            inst_n_idx,
        )

    def mma_full_b_atom_with_scale_fragments(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
    ):
        """Issue one SM120 block-scaled MMA atom from a full B fragment tile."""
        return self.mma_full_b_atom_with_prefetched_scales(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_fragment_buf,
            SFB_fragment_buf,
            SFB_rep_fragment_buf,
            inst_m_idx,
            inst_n_idx,
        )

    def mma_with_prefetched_scales(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            for i, j in T.grid(warp_rows, warp_cols):
                scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                scale_b_ptr = T.access_ptr(SFB_local_buf[j], "r")
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    scale_a_ptr,
                    scale_b_ptr,
                )
                if replicate_b:
                    scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[j], "r")
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_local_buf.data,
                        j * local_size_b + lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                        scale_a_ptr,
                        scale_b_rep_ptr,
                    )

        return _warp_mma_block_scale_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )

    def mma_with_prefetched_scales_cutlass_order(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_prefetched_cutlass_order(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            if replicate_b:
                for j in T.unroll(warp_cols):
                    scale_b_ptr = T.access_ptr(SFB_local_buf[j], "r")
                    for i in T.unroll(warp_rows):
                        scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            i * local_size_a,
                            B_local_buf.data,
                            j * local_size_b,
                            C_local_buf.data,
                            i * warp_cols * local_size_out + j * local_size_out,
                            scale_a_ptr,
                            scale_b_ptr,
                        )
                    scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[j], "r")
                    for rev_i in T.unroll(warp_rows):
                        i = warp_rows - 1 - rev_i
                        scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            i * local_size_a,
                            B_local_buf.data,
                            j * local_size_b + lift(local_size_b) // 2,
                            C_local_buf.data,
                            i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                            scale_a_ptr,
                            scale_b_rep_ptr,
                        )
            else:
                for j in T.unroll(warp_cols):
                    scale_b_ptr = T.access_ptr(SFB_local_buf[j], "r")
                    for i in T.unroll(warp_rows):
                        scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            i * local_size_a,
                            B_local_buf.data,
                            j * local_size_b,
                            C_local_buf.data,
                            i * warp_cols * local_size_out + j * local_size_out,
                            scale_a_ptr,
                            scale_b_ptr,
                        )

        return _warp_mma_block_scale_prefetched_cutlass_order(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )

    def mma_with_prefetched_scales_selector_probe(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_prefetched_selector_probe(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            for i in T.unroll(warp_rows):
                scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                scale_a_thread_id = i & 1
                for j in T.unroll(warp_cols):
                    scale_b_ptr = T.access_ptr(SFB_local_buf[j], "r")
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_local_buf.data,
                        j * local_size_b,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out,
                        scale_a_ptr,
                        scale_b_ptr,
                        0,
                        scale_a_thread_id,
                        0,
                        j & 3,
                    )
                    if replicate_b:
                        scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[j], "r")
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            i * local_size_a,
                            B_local_buf.data,
                            j * local_size_b + lift(local_size_b) // 2,
                            C_local_buf.data,
                            i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                            scale_a_ptr,
                            scale_b_rep_ptr,
                            0,
                            scale_a_thread_id,
                            0,
                            (j + 2) & 3,
                        )

        return _warp_mma_block_scale_prefetched_selector_probe(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )

    def mma_with_prefetched_scales_b_owner(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_owner_buf,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        if self.n_dim != 16 or int(warp_cols) != 2:
            raise ValueError("mma_with_prefetched_scales_b_owner currently requires replicated B with warp_cols=2")

        @T.macro
        def _warp_mma_block_scale_b_owner(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_owner_buf,
        ):
            scale_b_owner_ptr = T.access_ptr(SFB_owner_buf[0], "r")
            for i in T.unroll(warp_rows):
                scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                for j in T.unroll(warp_cols):
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_local_buf.data,
                        j * local_size_b,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out,
                        scale_a_ptr,
                        scale_b_owner_ptr,
                        0,
                        0,
                        0,
                        j * 2,
                    )
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_local_buf.data,
                        j * local_size_b + lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                        scale_a_ptr,
                        scale_b_owner_ptr,
                        0,
                        0,
                        0,
                        j * 2 + 1,
                    )

        return _warp_mma_block_scale_b_owner(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_owner_buf,
        )

    def mma_with_prefetched_scales_ab_owner(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_owner_buf,
        SFB_owner_buf,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        if int(warp_rows) != 4 or self.n_dim != 16 or int(warp_cols) != 2:
            raise ValueError("mma_with_prefetched_scales_ab_owner currently requires fulltile warp_rows=4, warp_cols=2")

        @T.macro
        def _warp_mma_block_scale_ab_owner(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_owner_buf,
            SFB_owner_buf,
        ):
            scale_b_owner_ptr = T.access_ptr(SFB_owner_buf[0], "r")
            for g in T.unroll(2):
                scale_a_owner_ptr = T.access_ptr(SFA_owner_buf[g], "r")
                for ai in T.unroll(2):
                    for j in T.unroll(warp_cols):
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            (g * 2 + ai) * local_size_a,
                            B_local_buf.data,
                            j * local_size_b,
                            C_local_buf.data,
                            (g * 2 + ai) * warp_cols * local_size_out + j * local_size_out,
                            scale_a_owner_ptr,
                            scale_b_owner_ptr,
                            0,
                            ai,
                            0,
                            j * 2,
                        )
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            (g * 2 + ai) * local_size_a,
                            B_local_buf.data,
                            j * local_size_b + lift(local_size_b) // 2,
                            C_local_buf.data,
                            (g * 2 + ai) * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                            scale_a_owner_ptr,
                            scale_b_owner_ptr,
                            0,
                            ai,
                            0,
                            j * 2 + 1,
                        )

        return _warp_mma_block_scale_ab_owner(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_owner_buf,
            SFB_owner_buf,
        )

    def mma_with_prefetched_scales_ab_owner_wide(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_owner_buf,
        SFB_owner_buf,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        if int(warp_rows) != 4 or self.n_dim != 16 or int(warp_cols) != 4:
            raise ValueError("mma_with_prefetched_scales_ab_owner_wide requires warp_rows=4, warp_cols=4")

        @T.macro
        def _warp_mma_block_scale_ab_owner_wide(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_owner_buf,
            SFB_owner_buf,
        ):
            for gm in T.unroll(2):
                scale_a_owner_ptr = T.access_ptr(SFA_owner_buf[gm], "r")
                for ai in T.unroll(2):
                    i = gm * 2 + ai
                    for gn in T.unroll(2):
                        scale_b_owner_ptr = T.access_ptr(SFB_owner_buf[gn], "r")
                        for ji in T.unroll(2):
                            j = gn * 2 + ji
                            T.ptx_mma_block_scale(
                                accum_dtype,
                                mma_prefix,
                                "row",
                                "col",
                                kind,
                                scale_vec_size,
                                a_dtype_abbrv,
                                b_dtype_abbrv,
                                stype,
                                A_local_buf.data,
                                i * local_size_a,
                                B_local_buf.data,
                                j * local_size_b,
                                C_local_buf.data,
                                i * warp_cols * local_size_out + j * local_size_out,
                                scale_a_owner_ptr,
                                scale_b_owner_ptr,
                                0,
                                ai,
                                0,
                                ji * 2,
                            )
                            T.ptx_mma_block_scale(
                                accum_dtype,
                                mma_prefix,
                                "row",
                                "col",
                                kind,
                                scale_vec_size,
                                a_dtype_abbrv,
                                b_dtype_abbrv,
                                stype,
                                A_local_buf.data,
                                i * local_size_a,
                                B_local_buf.data,
                                j * local_size_b + lift(local_size_b) // 2,
                                C_local_buf.data,
                                i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                                scale_a_owner_ptr,
                                scale_b_owner_ptr,
                                0,
                                ai,
                                0,
                                ji * 2 + 1,
                            )

        return _warp_mma_block_scale_ab_owner_wide(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_owner_buf,
            SFB_owner_buf,
        )

    def mma_atom_with_prefetched_scales(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
    ):
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        warp_cols = self.warp_cols
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_atom_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            scale_a_ptr = T.access_ptr(SFA_local_buf[inst_m_idx], "r")
            scale_b_ptr = T.access_ptr(SFB_local_buf[inst_n_idx], "r")
            T.ptx_mma_block_scale(
                accum_dtype,
                mma_prefix,
                "row",
                "col",
                kind,
                scale_vec_size,
                a_dtype_abbrv,
                b_dtype_abbrv,
                stype,
                A_local_buf.data,
                inst_m_idx * local_size_a,
                B_local_buf.data,
                0,
                C_local_buf.data,
                inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out,
                scale_a_ptr,
                scale_b_ptr,
            )
            if replicate_b:
                scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[inst_n_idx], "r")
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    inst_m_idx * local_size_a,
                    B_local_buf.data,
                    lift(local_size_b) // 2,
                    C_local_buf.data,
                    inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out + lift(local_size_out) // 2,
                    scale_a_ptr,
                    scale_b_rep_ptr,
                )

        return _warp_mma_block_scale_atom_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )

    def mma_b_atom_n8_serpentine_with_prefetched_scales(
        self,
        A_local_buf,
        B_atom_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
        inst_n_idx: PrimExpr | int,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_b_atom_n8_serpentine_prefetched(
            A_local_buf,
            B_atom_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            scale_b_ptr = T.access_ptr(SFB_local_buf[inst_n_idx], "r")
            for i in T.unroll(warp_rows):
                scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    i * local_size_a,
                    B_atom_buf.data,
                    0,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + inst_n_idx * local_size_out,
                    scale_a_ptr,
                    scale_b_ptr,
                )
            if replicate_b:
                scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[inst_n_idx], "r")
                for rev_i in T.unroll(warp_rows):
                    i = warp_rows - 1 - rev_i
                    scale_a_ptr = T.access_ptr(SFA_local_buf[i], "r")
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_atom_buf.data,
                        lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + inst_n_idx * local_size_out + lift(local_size_out) // 2,
                        scale_a_ptr,
                        scale_b_rep_ptr,
                    )

        return _warp_mma_b_atom_n8_serpentine_prefetched(
            A_local_buf,
            B_atom_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )

    def mma_b_atom_n8_serpentine_with_scale_pack(
        self,
        A_local_buf,
        B_atom_buf,
        C_local_buf,
        SFA_pack_buf,
        SFB_pack_buf,
        SFB_rep_pack_buf,
        k_block: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        num_k_blocks: int,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_b_atom_n8_serpentine_scale_pack(
            A_local_buf,
            B_atom_buf,
            C_local_buf,
            SFA_pack_buf,
            SFB_pack_buf,
            SFB_rep_pack_buf,
        ):
            scale_b_ptr = T.access_ptr(SFB_pack_buf[inst_n_idx * num_k_blocks + k_block], "r")
            for i in T.unroll(warp_rows):
                scale_a_ptr = T.access_ptr(SFA_pack_buf[i * num_k_blocks + k_block], "r")
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    i * local_size_a,
                    B_atom_buf.data,
                    0,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + inst_n_idx * local_size_out,
                    scale_a_ptr,
                    scale_b_ptr,
                )
            if replicate_b:
                scale_b_rep_ptr = T.access_ptr(SFB_rep_pack_buf[inst_n_idx * num_k_blocks + k_block], "r")
                for rev_i in T.unroll(warp_rows):
                    i = warp_rows - 1 - rev_i
                    scale_a_ptr = T.access_ptr(SFA_pack_buf[i * num_k_blocks + k_block], "r")
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_atom_buf.data,
                        lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + inst_n_idx * local_size_out + lift(local_size_out) // 2,
                        scale_a_ptr,
                        scale_b_rep_ptr,
                    )

        return _warp_mma_b_atom_n8_serpentine_scale_pack(
            A_local_buf,
            B_atom_buf,
            C_local_buf,
            SFA_pack_buf,
            SFB_pack_buf,
            SFB_rep_pack_buf,
        )

    def mma_full_b_atom_with_prefetched_scales(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
    ):
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        warp_cols = self.warp_cols
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_full_b_atom_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            scale_a_ptr = T.access_ptr(SFA_local_buf[inst_m_idx], "r")
            scale_b_ptr = T.access_ptr(SFB_local_buf[inst_n_idx], "r")
            T.ptx_mma_block_scale(
                accum_dtype,
                mma_prefix,
                "row",
                "col",
                kind,
                scale_vec_size,
                a_dtype_abbrv,
                b_dtype_abbrv,
                stype,
                A_local_buf.data,
                inst_m_idx * local_size_a,
                B_local_buf.data,
                inst_n_idx * local_size_b,
                C_local_buf.data,
                inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out,
                scale_a_ptr,
                scale_b_ptr,
            )
            if replicate_b:
                scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[inst_n_idx], "r")
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    inst_m_idx * local_size_a,
                    B_local_buf.data,
                    inst_n_idx * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out + lift(local_size_out) // 2,
                    scale_a_ptr,
                    scale_b_rep_ptr,
                )

        return _warp_mma_block_scale_full_b_atom_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )

    def mma_atom(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_buf,
        SFB_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
    ):
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_atom(A_local_buf, B_local_buf, C_local_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            scale_m = warp_m * warp_row_tiles + inst_m_idx * micro_size_x + sfa_row
            scale_n = warp_n * warp_col_tiles + inst_n_idx * micro_size_y + sfb_col
            scale_a_ptr = T.access_ptr(
                SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)],
                "r",
            )
            scale_b_ptr = T.access_ptr(
                SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_k)],
                "r",
            )
            T.ptx_mma_block_scale(
                accum_dtype,
                mma_prefix,
                "row",
                "col",
                kind,
                scale_vec_size,
                a_dtype_abbrv,
                b_dtype_abbrv,
                stype,
                A_local_buf.data,
                inst_m_idx * local_size_a,
                B_local_buf.data,
                0,
                C_local_buf.data,
                inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out,
                scale_a_ptr,
                scale_b_ptr,
            )
            if replicate_b:
                scale_b_rep_ptr = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n + 8, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    inst_m_idx * local_size_a,
                    B_local_buf.data,
                    lift(local_size_b) // 2,
                    C_local_buf.data,
                    inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out + lift(local_size_out) // 2,
                    scale_a_ptr,
                    scale_b_rep_ptr,
                )

        return _warp_mma_block_scale_atom(A_local_buf, B_local_buf, C_local_buf, SFA_data, SFB_data, thread_binding)


class SM120BlockScaledOperandPackage:
    """Internal K-block package for SM120 blockscaled MMA lowering.

    This object intentionally owns no scheduling policy.  It only groups the
    A/B fragments and matching scale fragments so the GEMM lowering can express
    CUTLASS-like copy_kblock/gemm_kblock structure without scattering the
    operand lifetime rules across the lowering body.
    """

    def __init__(
        self,
        mma_emitter: TensorCoreIntrinEmitterWithBlockScale,
        A_fragment_buf,
        B_fragment_buf,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
        A_region,
        B_region,
        C_buf,
        SFA_region,
        SFB_region,
        sf_k_start,
        sf_a_granularity_k: int,
        sf_b_granularity_k: int,
        sf_layout: str = "rowmajor",
    ):
        self.mma_emitter = mma_emitter
        self.A_fragment_buf = A_fragment_buf
        self.B_fragment_buf = B_fragment_buf
        self.SFA_fragment_buf = SFA_fragment_buf
        self.SFB_fragment_buf = SFB_fragment_buf
        self.SFB_rep_fragment_buf = SFB_rep_fragment_buf
        self.A_region = A_region
        self.B_region = B_region
        self.C_buf = C_buf
        self.SFA_region = SFA_region
        self.SFB_region = SFB_region
        self.sf_k_start = sf_k_start
        self.sf_a_granularity_k = sf_a_granularity_k
        self.sf_b_granularity_k = sf_b_granularity_k
        self.sf_layout = sf_layout

    def copy_kblock(self, k_block: PrimExpr | int):
        self.mma_emitter.ldmatrix_a(self.A_fragment_buf, self.A_region, k_block)
        self.mma_emitter.ldmatrix_b(self.B_fragment_buf, self.B_region, k_block)
        self.mma_emitter.ldscale_fragment(
            self.SFA_fragment_buf,
            self.SFB_fragment_buf,
            self.SFB_rep_fragment_buf,
            self.SFA_region,
            self.SFB_region,
            ki=k_block,
            k_start=self.sf_k_start,
            sf_a_granularity_k=self.sf_a_granularity_k,
            sf_b_granularity_k=self.sf_b_granularity_k,
            sf_layout=self.sf_layout,
        )

    def gemm_kblock(self):
        self.mma_emitter.mma_with_scale_fragments(
            self.A_fragment_buf,
            self.B_fragment_buf,
            self.C_buf,
            self.SFA_fragment_buf,
            self.SFB_fragment_buf,
            self.SFB_rep_fragment_buf,
        )

    def gemm_atom(self, inst_m_idx: PrimExpr | int, inst_n_idx: PrimExpr | int):
        self.mma_emitter.mma_full_b_atom_with_scale_fragments(
            self.A_fragment_buf,
            self.B_fragment_buf,
            self.C_buf,
            self.SFA_fragment_buf,
            self.SFB_fragment_buf,
            self.SFB_rep_fragment_buf,
            inst_m_idx,
            inst_n_idx,
        )
