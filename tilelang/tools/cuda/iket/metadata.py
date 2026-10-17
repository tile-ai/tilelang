"""Self-contained IKET event metadata carried through TIR."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Any


TOKEN_PREFIX = "__tl_iket_v1_"
MAX_EVENT_NAME_BYTES = 32
_TOKEN_VERSION = 1


@dataclass(frozen=True)
class Event:
    """Metadata for one IKET event emitted by a TileLang kernel."""

    name: str
    event_id: int
    kind: str
    range_id: int = 0
    payload_type: str = "NoPayload"
    payload_iket_id: int = 0


def encode_event(event: Event) -> str:
    """Encode an event as a source-safe token stored in a TIR StringImm."""
    data = {"version": _TOKEN_VERSION, **asdict(event)}
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return TOKEN_PREFIX + encoded


def decode_event(token: str) -> Event:
    """Decode and validate an event token recovered from generated CUDA."""
    if not token.startswith(TOKEN_PREFIX):
        raise ValueError("Invalid TileLang IKET metadata token prefix")

    encoded = token.removeprefix(TOKEN_PREFIX)
    try:
        padding = "=" * (-len(encoded) % 4)
        data: dict[str, Any] = json.loads(base64.urlsafe_b64decode(encoded + padding).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid TileLang IKET metadata token") from exc

    expected_fields = {
        "version",
        "name",
        "event_id",
        "kind",
        "range_id",
        "payload_type",
        "payload_iket_id",
    }
    if set(data) != expected_fields or data["version"] != _TOKEN_VERSION:
        raise ValueError("Unsupported TileLang IKET metadata schema")
    if not isinstance(data["name"], str) or not data["name"]:
        raise ValueError("IKET event metadata must contain a non-empty name")
    if len(data["name"].encode("utf-8")) > MAX_EVENT_NAME_BYTES:
        raise ValueError(f"IKET event name must be at most {MAX_EVENT_NAME_BYTES} UTF-8 bytes")
    if data["kind"] not in ("mark", "range"):
        raise ValueError(f"Unsupported IKET event kind: {data['kind']!r}")
    for field in ("event_id", "range_id", "payload_iket_id"):
        if not isinstance(data[field], int) or isinstance(data[field], bool) or data[field] < 0:
            raise ValueError(f"Invalid IKET metadata field {field!r}")
    if data["event_id"] == 0 or data["event_id"] == 31:
        raise ValueError(f"Invalid IKET event id: {data['event_id']}")
    if not isinstance(data["payload_type"], str):
        raise ValueError("Invalid IKET payload type metadata")

    data.pop("version")
    return Event(**data)
