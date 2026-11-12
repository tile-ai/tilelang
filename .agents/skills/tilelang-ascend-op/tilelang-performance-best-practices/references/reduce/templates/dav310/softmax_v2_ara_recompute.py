from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = SoftmaxStrategy("ara_recompute", "[A1, R, A0]", 3, ("max[A0]", "sum[A0]"), "stream strided R chunks in max, sum, output passes")
