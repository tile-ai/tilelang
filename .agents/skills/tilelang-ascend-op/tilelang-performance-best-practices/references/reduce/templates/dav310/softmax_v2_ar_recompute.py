from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = SoftmaxStrategy("ar_recompute", "[A, R]", 3, ("max", "sum"), "R exceeds full-load UB; stream max, stream sum, stream output")
