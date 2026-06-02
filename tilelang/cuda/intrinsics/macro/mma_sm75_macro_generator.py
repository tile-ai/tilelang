from __future__ import annotations

import tilelang.language as T
from tvm import DataType

from .mma_macro_generator import TensorCoreIntrinEmitter


class TensorCoreIntrinEmitterSM75(TensorCoreIntrinEmitter):
    """SM75/Turing-specific MMA shape policy."""

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits == 4:
            self.k_dim = min(32, self.chunk)
        elif a_dtype.bits == 8:
            self.k_dim = min(16, self.chunk)
        elif a_dtype.bits == 16:
            self.k_dim = min(8, self.chunk)
        else:
            self.k_dim = min(256 // a_dtype.bits, self.chunk)

    def _initialize_m_dim(self, a_dtype=T.float16):
        super()._initialize_m_dim(a_dtype)
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits in (4, 8):
            self.M_DIM = 8

    def _initialize_mma_prefix(self, k_dim: int = 16):
        if k_dim == 8:
            self.mma_prefix = "m16n8k8"
        elif k_dim == 16:
            self.mma_prefix = "m8n8k16"
        elif k_dim == 32:
            self.mma_prefix = "m8n8k32"
        else:
            super()._initialize_mma_prefix(k_dim)

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        if k_dim == 4:
            super()._initialize_micro_size(m_dim, k_dim)
            return

        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        use_m8n8 = k_dim in (16, 32)
        use_n8 = use_m8n8 or k_dim == 8
        if use_n8:
            if use_m8n8:
                assert m_dim == 8, f"For SM75 integer MMA, m_dim must be 8, got {m_dim}"
            self.n_dim = 8
            self.micro_size_y = 8
            self.warp_rows = warp_row_tiles // m_dim
            self.warp_cols = warp_col_tiles // 8
            self.micro_size_x = m_dim
            self.micro_size_k = k_dim
            return

        super()._initialize_micro_size(m_dim, k_dim)
