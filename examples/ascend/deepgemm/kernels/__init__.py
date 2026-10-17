"""Select the MNK or KMN kernel builders. MNK is the default."""


def load_kernels(loop_order="mnk"):
    if loop_order == "mnk":
        from . import mnk

        return mnk
    if loop_order == "kmn":
        from . import kmn

        return kmn
    raise ValueError(f"TILELANG_DEEPGEMM_LOOP_ORDER must be 'mnk' or 'kmn', got {loop_order!r}")
