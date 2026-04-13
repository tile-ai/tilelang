from enum import IntEnum


# same definition with src/op/gemm.h
class GemmInst(IntEnum):
    MMA = 0
    WGMMA = 1
    TCGEN5MMA = 2
    MFMA = 3
    Scalar = 4
    WMMA = 5  # AMD RDNA WMMA (gfx11/gfx12)

    def is_mma(self) -> bool:
        return self == GemmInst.MMA

    def is_wgmma(self) -> bool:
        return self == GemmInst.WGMMA

    def is_tcgen5mma(self) -> bool:
        return self == GemmInst.TCGEN5MMA

    def is_mfma(self) -> bool:
        return self == GemmInst.MFMA

    def is_scalar(self) -> bool:
        return self == GemmInst.Scalar

    def is_wmma(self) -> bool:
        return self == GemmInst.WMMA

    def __repr__(self) -> str:
        return self.name
