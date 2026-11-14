"""Ascend pipeline, cross-core, and pipe-buffer synchronization primitives."""

from __future__ import annotations

from tvm import tirx


# Pipe names accepted by codegen_ascend.cc::GetPipePair (HardEvent source and
# destination) and by EmitPipeBarrier_ (the PIPE_* barrier whitelist). Keeping
# the same sets here turns a late C++ ICHECK into a clear Python error.
_ASCEND_HARD_EVENT_PIPES = ("MTE1", "MTE2", "MTE3", "M", "V", "S", "FIX")
_ASCEND_PIPE_BARRIERS = (
    "PIPE_ALL", "PIPE_V", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3",
    "PIPE_S", "PIPE_FIX",
)
_ASCEND_CROSS_CORE_PIPES = tuple("PIPE_" + x for x in _ASCEND_HARD_EVENT_PIPES)


def _normalize_hard_event(hard_event, *, api):
    if not isinstance(hard_event, str):
        raise TypeError(f"{api} requires hard_event to be a string, got {type(hard_event)}")
    parts = hard_event.split("_")
    if len(parts) != 2 or any(x not in _ASCEND_HARD_EVENT_PIPES for x in parts):
        raise ValueError(
            f"{api} requires hard_event to be '<SRC>_<DST>' with both pipes in "
            f"{_ASCEND_HARD_EVENT_PIPES}, got {hard_event!r}"
        )
    return hard_event


def _normalize_sync_pipe(pipe, *, api, allowed):
    if not isinstance(pipe, str):
        raise TypeError(f"{api} requires pipe to be a string, got {type(pipe)}")
    if pipe not in allowed:
        raise ValueError(f"{api} requires pipe to be one of {allowed}, got {pipe!r}")
    return pipe


def _normalize_mode_id(mode_id, *, api):
    # bool is an int subclass: True would otherwise become mode 1 (subblock).
    # Non-constant expressions reach GetCrossCoreSyncScope, which LOG(FATAL)s.
    if isinstance(mode_id, bool):
        raise TypeError(f"{api} requires mode_id to be a plain int, got a bool")
    if isinstance(mode_id, int):
        value = mode_id
    elif isinstance(mode_id, tirx.IntImm):
        value = int(mode_id.value)
    else:
        raise TypeError(
            f"{api} requires mode_id to be a constant int (0, 1, 2, or 4), "
            f"got {type(mode_id)}"
        )
    if value not in (0, 1, 2, 4):
        raise ValueError(f"{api} requires mode_id to be one of 0, 1, 2, 4, got {value}")
    return tirx.IntImm("int32", value)


def _normalize_lock_mode(mode, *, api):
    # `1 if mode else 0` silently treats non-empty strings (e.g. "False") as 1.
    if isinstance(mode, bool):
        return 1 if mode else 0
    if isinstance(mode, int) and mode in (0, 1):
        return mode
    if isinstance(mode, tirx.IntImm) and int(mode.value) in (0, 1):
        return int(mode.value)
    raise TypeError(f"{api} requires mode to be a bool or a constant 0/1 flag, got {mode!r}")


def ascend_pipe_barrier(pipe_t: str) -> None:
    """Insert an AscendC pipeline barrier.

    Generates: AscendC::PipeBarrier<pipe_t::XXX>();

    Parameters
    ----------
    pipe_t : str
        Pipeline type string, e.g., "PIPE_ALL", "PIPE_V", "PIPE_M",
        "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_S".

    Example
    -------
    >>> T.ascend_pipe_barrier("PIPE_ALL")
    """
    pipe_t = _normalize_sync_pipe(pipe_t, api="T.ascend_pipe_barrier", allowed=_ASCEND_PIPE_BARRIERS)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_pipe_barrier"), pipe_t)


_EVENT_ID_MAX = 0xFFFF  # hardware event-ID slot width


def _normalize_event_id(event_id):
    """Box a Python int event id, range-checked against the hardware slot."""
    if isinstance(event_id, bool) or not isinstance(event_id, int):
        if not isinstance(event_id, tirx.PrimExpr):
            raise TypeError(f"event_id must be an int or PrimExpr, got {type(event_id)}")
        return event_id
    if not 0 <= event_id <= _EVENT_ID_MAX:
        raise ValueError(f"event_id must be within [0, {_EVENT_ID_MAX}], got {event_id}")
    return tirx.IntImm("int32", event_id)


def ascend_set_flag(hard_event: str, event_id) -> None:
    """Insert an AscendC SetFlag for pipeline synchronization.

    Generates: AscendC::SetFlag<AscendC::HardEvent::XXX>(eventID);

    Parameters
    ----------
    hard_event : str
        HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
    event_id : int or PrimExpr
        Event ID for synchronization.

    Example
    -------
    >>> T.ascend_set_flag("S_MTE3", event_id)
    """
    hard_event = _normalize_hard_event(hard_event, api="T.ascend_set_flag/wait_flag")
    event_id = _normalize_event_id(event_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_set_flag"), hard_event, event_id)


def ascend_wait_flag(hard_event: str, event_id) -> None:
    """Insert an AscendC WaitFlag for pipeline synchronization.

    Generates: AscendC::WaitFlag<AscendC::HardEvent::XXX>(eventID);

    Parameters
    ----------
    hard_event : str
        HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
    event_id : int or PrimExpr
        Event ID for synchronization.

    Example
    -------
    >>> T.ascend_wait_flag("S_MTE3", event_id)
    """
    hard_event = _normalize_hard_event(hard_event, api="T.ascend_set_flag/wait_flag")
    event_id = _normalize_event_id(event_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_wait_flag"), hard_event, event_id)


def ascend_sync_inter_arrive(pipe: str, flag_id) -> None:
    """Signal an Ascend inter-core synchronization event.

    Generates: AscendC::CrossCoreSetFlag<0, pipe>(flag_id);

    Parameters
    ----------
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Inter-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
    """
    return ascend_cross_core_set_flag(0, pipe, flag_id)


def ascend_sync_inter_wait(pipe: str, flag_id) -> None:
    """Wait for an Ascend inter-core synchronization event.

    Generates: AscendC::CrossCoreWaitFlag<0, pipe>(flag_id);

    Parameters
    ----------
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Inter-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_sync_inter_wait("PIPE_MTE3", flag_id)
    """
    return ascend_cross_core_wait_flag(0, pipe, flag_id)


def ascend_threadfence() -> None:
    """Insert an AscendC thread fence.

    Generates: asc_threadfence();

    Example
    -------
    >>> T.ascend_threadfence()
    """
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_threadfence"))


def ascend_cross_core_set_flag(mode_id: int, pipe: str, flag_id) -> None:
    """Insert an AscendC CrossCoreSetFlag for cross-core synchronization.

    Generates: AscendC::CrossCoreSetFlag<modeId, pipe>(flagId);

    Parameters
    ----------
    mode_id : int
        Cross-core sync mode: 0, 1, 2, or 4.
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Cross-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
    """
    pipe = _normalize_sync_pipe(pipe, api="T.ascend_cross_core_set_flag/wait_flag", allowed=_ASCEND_CROSS_CORE_PIPES)
    mode_id = _normalize_mode_id(mode_id, api="T.ascend_cross_core_set_flag/wait_flag")
    if isinstance(flag_id, int):
        flag_id = tirx.IntImm("int32", flag_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_cross_core_set_flag"), mode_id, pipe, flag_id)


def ascend_cross_core_wait_flag(mode_id: int, pipe: str, flag_id) -> None:
    """Insert an AscendC CrossCoreWaitFlag for cross-core synchronization.

    Generates: AscendC::CrossCoreWaitFlag<modeId, pipe>(flagId);

    Parameters
    ----------
    mode_id : int
        Cross-core sync mode: 0, 1, 2, or 4.
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Cross-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
    """
    pipe = _normalize_sync_pipe(pipe, api="T.ascend_cross_core_set_flag/wait_flag", allowed=_ASCEND_CROSS_CORE_PIPES)
    mode_id = _normalize_mode_id(mode_id, api="T.ascend_cross_core_set_flag/wait_flag")
    if isinstance(flag_id, int):
        flag_id = tirx.IntImm("int32", flag_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_cross_core_wait_flag"), mode_id, pipe, flag_id)


def ascend_get_buf(pipe, buf_id, mode=False):
    """Acquire a pipe buffer for double-buffer management.

    Generates: asc_lock(pipe, buf_id, mode);

    Parameters
    ----------
    pipe : str
        Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
    buf_id : int or PrimExpr
        Buffer slot index, typically (sk & 1) for double-buffering.
    mode : bool
        Optional mode flag (default False).

    Example
    -------
    >>> T.ascend_get_buf("PIPE_MTE1", sk & 1)
    """
    if isinstance(buf_id, int):
        buf_id = tirx.IntImm("int32", buf_id)
    mode_val = tirx.IntImm("int32", _normalize_lock_mode(mode, api="T.ascend_get_buf/rls_buf"))
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_get_buf"), pipe, buf_id, mode_val)


def ascend_rls_buf(pipe, buf_id, mode=False):
    """Release a pipe buffer for double-buffer management.

    Generates: asc_unlock(pipe, buf_id, mode);

    Parameters
    ----------
    pipe : str
        Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
    buf_id : int or PrimExpr
        Buffer slot index, typically (sk & 1) for double-buffering.
    mode : bool
        Optional mode flag (default False).

    Example
    -------
    >>> T.ascend_rls_buf("PIPE_M", sk & 1)
    """
    if isinstance(buf_id, int):
        buf_id = tirx.IntImm("int32", buf_id)
    mode_val = tirx.IntImm("int32", _normalize_lock_mode(mode, api="T.ascend_get_buf/rls_buf"))
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_rls_buf"), pipe, buf_id, mode_val)


__all__ = [
    "ascend_pipe_barrier",
    "ascend_set_flag",
    "ascend_wait_flag",
    "ascend_sync_inter_arrive",
    "ascend_sync_inter_wait",
    "ascend_threadfence",
    "ascend_cross_core_set_flag",
    "ascend_cross_core_wait_flag",
    "ascend_get_buf",
    "ascend_rls_buf",
]
