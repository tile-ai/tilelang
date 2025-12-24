# Copied from bitblas
from enum import IntEnum


class TransformKind(IntEnum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2
    LDMatrixTransform = 3

    def is_non_transform(self):
        return self == TransformKind.NonTransform

    def is_inter_warp_transform(self):
        return self == TransformKind.InterWarpTransform

    def is_intra_warp_transform(self):
        return self == TransformKind.IntraWarpTransform

    def is_ld_matrix_transform(self):
        return self == TransformKind.LDMatrixTransform
