from tilelang.cuda.intrinsics.macro.mma_sm75_macro_generator import TensorCoreIntrinEmitterSM75

from .gemm_mma import GemmMMA


class GemmMMASm75(GemmMMA):
    intrin_emitter_cls = TensorCoreIntrinEmitterSM75
