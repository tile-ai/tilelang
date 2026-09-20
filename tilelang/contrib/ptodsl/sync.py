"""PTO synchronization compatibility helpers for generated TileLang kernels."""

from ptodsl import pto, scalar
from ptoas.mlir.dialects import pto as mlir_pto

from .common import unwrap_surface_value


_CROSS_CORE_MODES = (0, 1, 2, 4)


def _event_id_operand(flag_id, *, context):
    if isinstance(flag_id, int):
        return flag_id
    try:
        event_id = scalar.cast(flag_id, pto.i64)
        return unwrap_surface_value(scalar.index_cast(event_id))
    except TypeError as exc:
        raise TypeError(f"{context} expects an integer-like flag_id") from exc


def _ascend_cross_core_flag(mode_id, pipe, flag_id, *, is_set):
    if mode_id not in _CROSS_CORE_MODES:
        raise ValueError(
            f"ascend cross-core flag mode_id must be one of 0/1/2/4, got {mode_id}"
        )

    event_id = _event_id_operand(
        flag_id,
        context=f"ascend_cross_core_{'set' if is_set else 'wait'}_flag",
    )
    if mode_id == 0:
        op = pto.set_cross_block if is_set else pto.wait_cross_block
        op(pipe, event_id)
        return
    if mode_id == 4:
        op = pto.set_intra_block if is_set else pto.wait_intra_block
        op(pipe, event_id)
        return

    # Modes 0 and 4 use named block operations. FFTS modes 1 and 2 intentionally
    # remain on the low-level pto.sync.set/wait surface.
    op = mlir_pto.sync_set if is_set else mlir_pto.sync_wait
    op(pipe, event_id, ffts_mode=mode_id)


def ascend_cross_core_set_flag(mode_id, pipe, flag_id):
    """Lower TileLang's cross-core set flag for modes 0, 1, 2, and 4."""
    _ascend_cross_core_flag(mode_id, pipe, flag_id, is_set=True)


def ascend_cross_core_wait_flag(mode_id, pipe, flag_id):
    """Lower TileLang's cross-core wait flag for modes 0, 1, 2, and 4."""
    _ascend_cross_core_flag(mode_id, pipe, flag_id, is_set=False)
