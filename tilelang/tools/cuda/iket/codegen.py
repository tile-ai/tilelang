"""CUDA source instrumentation for IKET events."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .metadata import TOKEN_PREFIX, Event, decode_event, encode_event


_RANGE_END_EVENT_ID = 31
_IKET_MAGIC = [157, 241, 190, 186]
_INJECTION_MARKER = "__tilelang_iket_tool__"
_TOKEN_PATTERN = re.compile(re.escape(TOKEN_PREFIX) + r"[A-Za-z0-9_-]+")
_EVENT_CALL_PATTERN = re.compile(
    r"(?P<prefix>\bTL_IKET_EVENT(?:_PAYLOAD_(?:U32|F32))?\(\s*)"
    r"(?P<event_id>[0-9]+)"
    r"(?P<suffix>[^\n]*?\"(?P<token>" + re.escape(TOKEN_PREFIX) + r"[A-Za-z0-9_-]+)\"\s*\))"
)


def _canonicalize_cuda_events(code: str) -> tuple[str, list[Event]]:
    """Recover events and assign module-wide IDs to their generated calls."""
    by_key: dict[tuple[str, str], Event] = {}
    token_ids: dict[str, int] = {}
    canonical_tokens: dict[str, str] = {}
    next_event_id = 1
    matches = list(_EVENT_CALL_PATTERN.finditer(code))
    if len(matches) != len(_TOKEN_PATTERN.findall(code)):
        raise ValueError("IKET metadata token is not attached to a supported event call")

    for match in matches:
        token = match.group("token")
        event = decode_event(token)
        key = (event.kind, event.name)
        previous_key = by_key.get(key)
        if previous_key is not None:
            if replace(previous_key, event_id=event.event_id) != event:
                raise ValueError(f"Conflicting IKET metadata for event {event.name!r}")
            token_ids[token] = previous_key.event_id
            canonical_tokens[token] = encode_event(previous_key)
            continue

        while next_event_id == _RANGE_END_EVENT_ID:
            next_event_id += 1
        canonical = replace(event, event_id=next_event_id)
        next_event_id += 1
        by_key[key] = canonical
        token_ids[token] = canonical.event_id
        canonical_tokens[token] = encode_event(canonical)

    def rewrite_id(match: re.Match[str]) -> str:
        token = match.group("token")
        suffix = match.group("suffix").replace(token, canonical_tokens[token])
        return match.group("prefix") + str(token_ids[token]) + suffix

    return _EVENT_CALL_PATTERN.sub(rewrite_id, code), list(by_key.values())


def _safe_symbol(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return safe


def _u32(value: int) -> list[int]:
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


def _attrs(size: int, assignments: dict[int, list[int]], name: str | None = None) -> list[int]:
    data = [0] * size
    data[0:4] = _u32(size)
    for offset, bytes_ in assignments.items():
        data[offset : offset + len(bytes_)] = bytes_
    if name is not None:
        encoded = name.encode("utf-8")
        if len(encoded) > size - 28:
            raise ValueError(f"IKET event name is too long for minimal metadata: {name!r}")
        data[24:28] = _u32(len(encoded))
        data[28 : 28 + len(encoded)] = list(encoded)
    return data


def _range_attrs(name: str, range_id: int) -> list[int]:
    encoded = name.encode("utf-8")
    size = 72
    if len(encoded) > size - 36:
        raise ValueError(f"IKET range name is too long for minimal metadata: {name!r}")
    data = [0] * size
    data[0:4] = _u32(size)
    data[8:12] = _u32(range_id)
    data[12:16] = _u32(0xFFFFFFFF)
    data[16:20] = _u32(2)
    data[32:36] = _u32(len(encoded))
    data[36 : 36 + len(encoded)] = list(encoded)
    return data


def _c_array(symbol: str, data: list[int]) -> str:
    values = ", ".join(str(x) for x in data)
    return f"__device__ __attribute__((used, aligned(1))) unsigned char {symbol}[{len(data)}] = {{{values}}};"


def _metadata_decls(events: Iterable[Event], *, runtime_payloads: bool) -> str:
    events = list(events)
    if not events:
        return ""

    lines = ['extern "C" {']
    ranges: dict[str, int] = {}
    max_event_id = 0
    instrument_method = 3

    for event in events:
        max_event_id = max(max_event_id, event.event_id)
        assignments = {
            4: _u32(event.event_id),
            8: _u32(instrument_method),
            12: _u32(event.payload_iket_id if runtime_payloads else 0),
        }
        if event.kind == "range":
            assignments[16] = _u32(1)
            assignments[20] = _u32(event.range_id)
            ranges[event.name] = event.range_id
        symbol = f"__iket_evt_decl_{_safe_symbol(event.name)}_{event.event_id}_attrs"
        lines.append(_c_array(symbol, _attrs(60, assignments, event.name)))

    for name, range_id in ranges.items():
        symbol = f"__iket_range_decl_{_safe_symbol(name)}_{range_id}_attrs"
        lines.append(_c_array(symbol, _range_attrs(name, range_id)))

    meta = [0] * 48
    meta[0:4] = _u32(48)
    meta[8:12] = _u32(len(events) + len(ranges))
    meta[12:16] = _u32(_RANGE_END_EVENT_ID)
    meta[16:20] = _u32(32)
    meta[20:24] = _u32(60)
    meta[24:28] = _IKET_MAGIC
    meta[32:36] = _u32(max_event_id)
    lines.append(_c_array("__iket_meta_info", meta))
    lines.append("}")
    return "\n".join(lines)


def _target_sm(target: Any) -> int | None:
    attrs = getattr(target, "attrs", None)
    arch = None
    if attrs is not None:
        try:
            arch = attrs.get("arch", None)
        except (AttributeError, KeyError, TypeError):
            arch = None
    candidates = [str(arch)] if arch is not None else []
    candidates.append(str(target))
    for candidate in candidates:
        match = re.search(r"(?:sm|compute)_([0-9]+)", candidate)
        if match is not None:
            return int(match.group(1))
    return None


def _cluster_rank_instruction(target: Any) -> str:
    sm = _target_sm(target)
    if sm is not None and sm >= 90:
        return "mov.b32 r, %cluster_ctarank;"
    return "mov.u32 r, 0;"


def _event_macros(target: Any, *, runtime_payloads: bool) -> str:
    rank_instruction = _cluster_rank_instruction(target)
    event = rf"""
#define TL_IKET_EVENT(ID, ...) \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" ::: "memory")
"""
    if not runtime_payloads:
        return (
            event
            + r"""
#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE, ...) TL_IKET_EVENT(ID)
#define TL_IKET_EVENT_PAYLOAD_F32(ID, VALUE, ...) TL_IKET_EVENT(ID)
"""
        )

    payload = rf"""
#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE, ...) do {{ \
  unsigned int __tl_iket_payload_u32 = (unsigned int)(VALUE); \
  /* Separate 32-bit stores avoid an STS.64 record rejected by IKET NativeDump. */ \
  asm volatile("{{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "}}" ::: "memory"); \
  asm volatile("{{\n\t" \
               ".reg .b32 r, p;\n\t" \
               "{rank_instruction}\n\t" \
               "mov.u32 p, %0;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x24;\n\t" \
               "st.volatile.shared.u32 [r], p;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}}" :: "r"(__tl_iket_payload_u32) : "memory"); \
}} while (0)

#define TL_IKET_EVENT_PAYLOAD_F32(ID, VALUE, ...) do {{ \
  union {{ float f; unsigned int u; }} __tl_iket_payload_f32_bits; \
  __tl_iket_payload_f32_bits.f = (float)(VALUE); \
  TL_IKET_EVENT_PAYLOAD_U32(ID, __tl_iket_payload_f32_bits.u); \
}} while (0)
"""
    return event + payload


def inject_iket_cuda(code: str, target: Any, *, runtime_payloads: bool) -> str:
    """Inject IKET declarations and event macros into generated CUDA source."""
    if _INJECTION_MARKER in code:
        return code
    code, events = _canonicalize_cuda_events(code)
    if not events:
        return code

    insert_at = code.find("#include <tl_templates")
    if insert_at < 0:
        insert_at = code.find("#include")
    if insert_at < 0:
        insert_at = 0

    prefix = (
        f"// {_INJECTION_MARKER}\n"
        "// Experimental IKET metadata emitted by tilelang.tools.cuda.iket.\n"
        + _metadata_decls(events, runtime_payloads=runtime_payloads)
        + "\n"
        + _event_macros(target, runtime_payloads=runtime_payloads)
        + "\n"
    )
    return code[:insert_at] + prefix + code[insert_at:]
