from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = SoftmaxStrategy("ara_full_load", "[A1, R, A0]", 1, ("max[A0]", "sum[A0]", "exp tile"), "R x tile_A0 and fp32 workspace fit UB")
