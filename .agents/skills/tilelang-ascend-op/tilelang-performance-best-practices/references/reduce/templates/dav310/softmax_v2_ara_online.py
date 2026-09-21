from ..softmax_v2_tiling_data import SoftmaxStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = SoftmaxStrategy("ara_online", "[A1, R, A0]", 2, ("running_max[A0]", "running_sum[A0]"), "online max/sum pass followed by output pass")


def merge_online(m_old, l_old, chunk_max, chunk_exp_sum, exp):
    """Scalar form of the fp32 online merge used per A0 lane."""
    m_new = max(m_old, chunk_max)
    l_new = l_old * exp(m_old - m_new) + chunk_exp_sum * exp(chunk_max - m_new)
    return m_new, l_new
