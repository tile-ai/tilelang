"""Kernel-side IKET event helpers."""

from __future__ import annotations

import threading
import zlib
from dataclasses import dataclass
from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import evaluate as T_evaluate

from .metadata import MAX_EVENT_NAME_BYTES, Event, encode_event


_RANGE_END_EVENT_ID = 31
_EVENT_EXTERN = "TL_IKET_EVENT"
_EVENT_PAYLOAD_U32_EXTERN = "TL_IKET_EVENT_PAYLOAD_U32"
_EVENT_PAYLOAD_F32_EXTERN = "TL_IKET_EVENT_PAYLOAD_F32"


@dataclass(frozen=True)
class PayloadSpec:
    """IKET payload value descriptor for a TileLang CUDA event."""

    expr: Any
    dtype: str
    iket_id: int


_events: dict[tuple[str, str], Event] = {}
_ranges: dict[str, int] = {}
_range_stacks = threading.local()
_registry_lock = threading.RLock()
_next_event_id = 1


def _range_stack() -> list[str]:
    stack = getattr(_range_stacks, "value", None)
    if stack is None:
        stack = []
        _range_stacks.value = stack
    return stack


def reset() -> None:
    """Reset frontend event-id allocation for kernels built afterward."""
    global _next_event_id
    with _registry_lock:
        _events.clear()
        _ranges.clear()
        _range_stack().clear()
        _next_event_id = 1


def event_table() -> list[dict[str, int | str]]:
    """Return events registered while constructing recent kernels."""
    from .session import runtime_payloads_enabled

    runtime_payloads = runtime_payloads_enabled()
    with _registry_lock:
        return [
            {
                "name": event.name,
                "event_id": event.event_id,
                "kind": event.kind,
                "range_id": event.range_id,
                "payload_type": event.payload_type,
                "payload_iket_id": event.payload_iket_id,
                "runtime_payload_iket_id": event.payload_iket_id if runtime_payloads else 0,
            }
            for event in sorted(_events.values(), key=lambda item: item.event_id)
        ]


def payload(expr: Any, dtype: str | None = None) -> PayloadSpec:
    """Attach a supported scalar payload value to an IKET event."""
    resolved_dtype = _normalize_payload_dtype(dtype or _infer_payload_dtype(expr))
    _validate_runtime_payload_dtype(resolved_dtype)
    return PayloadSpec(expr=expr, dtype=resolved_dtype, iket_id=_payload_iket_id(resolved_dtype))


def mark(name: str, payload: Any = None):
    """Emit an IKET instant marker at the current program point."""
    payload_spec = _payload_spec(payload)
    event = _get_event(name, "mark", payload_spec=payload_spec)
    return _event_call(event, payload_spec)


def range_push(name: str, payload: Any = None):
    """Emit an IKET range-start event at the current program point."""
    payload_spec = _payload_spec(payload)
    event = _get_event(name, "range", payload_spec=payload_spec)
    _range_stack().append(name)
    return _event_call(event, payload_spec)


def range_pop(name: str | None = None):
    """Emit an IKET range-end event for the current warp-local range."""
    stack = _range_stack()
    if not stack:
        raise RuntimeError("iket.range_pop() called without a matching iket.range_push()")
    started = stack.pop()
    if name is not None and name != started:
        raise RuntimeError(f"iket.range_pop({name!r}) does not match active range {started!r}")
    return tirx.call_extern("handle", _EVENT_EXTERN, _RANGE_END_EVENT_ID)


def range_start(name: str, payload: Any = None):
    """Alias for :func:`range_push`."""
    return range_push(name, payload=payload)


def range_end(name: str | None = None):
    """Alias for :func:`range_pop`."""
    return range_pop(name)


class _RangeScope:
    def __init__(self, name: str, payload: Any = None):
        self.name = name
        self.payload = payload

    def __enter__(self):
        T_evaluate(range_push(self.name, payload=self.payload))
        return self

    def __exit__(self, exc_type, exc, tb):
        T_evaluate(range_pop(self.name))
        return False


def range(name: str, payload: Any = None) -> _RangeScope:
    """Return a scope that emits IKET range-start and range-end events."""
    return _RangeScope(name, payload=payload)


def _payload_spec(value: Any) -> PayloadSpec | None:
    if value is None:
        return None
    if isinstance(value, PayloadSpec):
        return value
    return payload(value)


def _get_event(name: str, kind: str, payload_spec: PayloadSpec | None = None) -> Event:
    if not isinstance(name, str) or not name:
        raise ValueError("IKET event name must be a non-empty string")
    if len(name.encode("utf-8")) > MAX_EVENT_NAME_BYTES:
        raise ValueError(f"IKET event name must be at most {MAX_EVENT_NAME_BYTES} UTF-8 bytes: {name!r}")
    if kind not in ("mark", "range"):
        raise ValueError(f"Unsupported IKET event kind: {kind}")

    key = (kind, name)
    with _registry_lock:
        if key in _events:
            event = _events[key]
            _validate_payload_compat(event, payload_spec)
            return event

        global _next_event_id
        while _next_event_id == _RANGE_END_EVENT_ID:
            _next_event_id += 1
        event_id = _next_event_id
        _next_event_id += 1

        range_id = 0
        if kind == "range":
            range_id = _range_id(name)
            _ranges[name] = range_id

        event = Event(
            name=name,
            event_id=event_id,
            kind=kind,
            range_id=range_id,
            payload_type=payload_spec.dtype if payload_spec is not None else "NoPayload",
            payload_iket_id=payload_spec.iket_id if payload_spec is not None else 0,
        )
        _events[key] = event
        return event


def _validate_payload_compat(event: Event, payload_spec: PayloadSpec | None) -> None:
    payload_type = payload_spec.dtype if payload_spec is not None else "NoPayload"
    if event.payload_type != payload_type:
        raise ValueError(
            f"IKET event {event.name!r} was first registered with payload type "
            f"{event.payload_type!r}, but is now used with {payload_type!r}"
        )


def _event_call(event: Event, payload_spec: PayloadSpec | None):
    token = encode_event(event)
    if payload_spec is None:
        return tirx.call_extern("handle", _EVENT_EXTERN, event.event_id, token)
    extern = _EVENT_PAYLOAD_F32_EXTERN if payload_spec.dtype == "float32" else _EVENT_PAYLOAD_U32_EXTERN
    return tirx.call_extern("handle", extern, event.event_id, payload_spec.expr, token)


def _infer_payload_dtype(expr: Any) -> str:
    dtype = getattr(expr, "dtype", None)
    if dtype is not None:
        return str(dtype)
    if isinstance(expr, bool):
        return "uint32"
    if isinstance(expr, int):
        return "int32"
    if isinstance(expr, float):
        return "float32"
    raise TypeError("Cannot infer IKET payload dtype. Use iket.payload(expr, dtype='int32') with one of int32/uint32/float32.")


def _normalize_payload_dtype(dtype: str) -> str:
    aliases = {"int": "int32", "uint": "uint32", "float": "float32"}
    normalized = aliases.get(dtype.lower(), dtype.lower())
    _payload_iket_id(normalized)
    return normalized


def _validate_runtime_payload_dtype(dtype: str) -> None:
    if dtype not in ("int32", "uint32", "float32"):
        raise NotImplementedError("TileLang IKET runtime payload capture currently supports only int32, uint32, and float32 payloads.")


def _payload_iket_id(dtype: str) -> int:
    ids = {"int32": 5, "uint32": 6, "float32": 13}
    key = dtype.lower()
    if key not in ids:
        raise ValueError(f"Unsupported IKET payload dtype: {dtype!r}")
    return ids[key]


def _range_id(name: str) -> int:
    value = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    return value or 1
