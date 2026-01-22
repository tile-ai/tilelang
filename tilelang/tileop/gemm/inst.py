from enum import IntEnum


# TODO(lei): support Volta and WMMA?
# same definition with src/op/gemm.h
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
