"""Experimental IKET frontend hooks for TileLang CUDA kernels.

This module exposes a small opt-in API that emits IKET-compatible event
markers through TileLang's regular CUDA backend. It is intentionally minimal:
events are represented as extern calls in TIR, and a CUDA post-processing
callback injects the matching metadata declarations and inline PTX helper.
"""

from __future__ import annotations

import re
import os
import shlex
import zlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

import tvm_ffi
from tvm import tirx
from tvm.tirx.script.builder import evaluate as T_evaluate

from tilelang.engine.callback import register_cuda_postproc, register_cuda_postproc_callback


_RANGE_END_EVENT_ID = 31
_IKET_MAGIC = [157, 241, 190, 186]


@dataclass
class _IketEvent:
    name: str
    event_id: int
    kind: str
    range_id: int = 0
    payload_type: str = "NoPayload"
    payload_iket_id: int = 0


@dataclass(frozen=True)
class PayloadSpec:
    """IKET payload value descriptor for TileLang CUDA events."""

    expr: Any
    dtype: str
    iket_id: int


_events: dict[tuple[str, str], _IketEvent] = {}
_ranges: dict[str, int] = {}
_range_stack: list[str] = []
_next_event_id = 1
_enabled = False
_previous_cuda_postproc: Callable[[str, Any], str] | None = None
_output_dir: Path | None = None
_runtime_payloads_enabled = False


def reset() -> None:
    """Reset the process-local IKET event registry used by the next compile."""
    global _next_event_id
    _events.clear()
    _ranges.clear()
    _range_stack.clear()
    _next_event_id = 1


def set_output_dir(path: str | os.PathLike[str] | None) -> Path | None:
    """Set the process-local IKET export directory used by helper APIs."""
    global _output_dir
    if path is None:
        _output_dir = None
        os.environ.pop("TL_IKET_OUTPUT_DIR", None)
        return None

    _output_dir = Path(path).expanduser().absolute()
    _output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TL_IKET_OUTPUT_DIR"] = str(_output_dir)
    return _output_dir


def output_dir(default: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the configured IKET export directory, if any."""
    if _output_dir is not None:
        return _output_dir
    env_dir = os.environ.get("TL_IKET_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().absolute()
    if default is None:
        return None
    return Path(default).expanduser().absolute()


def output_path(name: str, *, directory: str | os.PathLike[str] | None = None) -> Path:
    """Return a path under the configured IKET export directory."""
    base = output_dir(directory)
    if base is None:
        raise RuntimeError("IKET output directory is not configured")
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def trace_files(*, directory: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return generated IKET JSON traces in size-descending order."""
    base = output_dir(directory)
    if base is None or not base.exists():
        return []
    return sorted(base.glob("*.trace.json"), key=lambda path: path.stat().st_size, reverse=True)


def profile_command(
    command: str | Sequence[str],
    *,
    directory: str | os.PathLike[str],
    postprocess: str = "all",
    clobber: bool = True,
) -> str:
    """Build an IKET CLI command that exports traces to ``directory``."""
    parts = [
        "python",
        "-m",
        "iket.cli.main",
        "--output-dir",
        str(Path(directory).expanduser()),
    ]
    if clobber:
        parts.append("--clobber")
    parts.extend(["profile", "--postprocess", postprocess, "--"])
    command_prefix = " ".join(shlex.quote(str(item)) for item in parts)
    if isinstance(command, str):
        return f"{command_prefix} {command}"
    return " ".join([command_prefix, *(shlex.quote(str(item)) for item in command)])


def payload(expr: Any, dtype: str | None = None) -> PayloadSpec:
    """Attach a scalar payload value to an IKET event."""
    resolved_dtype = _normalize_payload_dtype(dtype or _infer_payload_dtype(expr))
    _validate_runtime_payload_dtype(resolved_dtype)
    return PayloadSpec(expr=expr, dtype=resolved_dtype, iket_id=_payload_iket_id(resolved_dtype))


def enable_runtime_payloads() -> None:
    """Emit IKET runtime payload metadata for payload events."""
    global _runtime_payloads_enabled
    _runtime_payloads_enabled = True


def disable_runtime_payloads() -> None:
    """Disable IKET payload metadata while keeping event instrumentation enabled."""
    global _runtime_payloads_enabled
    _runtime_payloads_enabled = False


def runtime_payloads_enabled() -> bool:
    """Return whether payload metadata emission is enabled."""
    return _runtime_payloads_enabled


def enable(*, override: bool = True) -> None:
    """Enable IKET CUDA source post-processing for subsequent compiles."""
    global _enabled, _previous_cuda_postproc
    if _enabled:
        return

    _previous_cuda_postproc = tvm_ffi.get_global_func("tilelang_callback_cuda_postproc", allow_missing=True)

    @register_cuda_postproc_callback(override=override)
    def tilelang_callback_cuda_postproc(code, _target):
        return _inject_iket_cuda(code)

    _enabled = True


def disable(*, restore: bool = True) -> None:
    """Disable IKET CUDA source post-processing for subsequent compiles."""
    global _enabled, _previous_cuda_postproc
    if not _enabled:
        return

    if restore and _previous_cuda_postproc is not None:
        register_cuda_postproc(_previous_cuda_postproc, override=True)
    else:
        register_cuda_postproc(_identity_cuda_postproc, override=True)

    _previous_cuda_postproc = None
    _enabled = False


def is_enabled() -> bool:
    """Return whether the IKET CUDA post-processing callback is registered."""
    return _enabled


def event_table() -> list[dict[str, int | str]]:
    """Return the currently registered IKET events for debugging/tests."""
    return [
        {
            "name": event.name,
            "event_id": event.event_id,
            "kind": event.kind,
            "range_id": event.range_id,
            "payload_type": event.payload_type,
            "payload_iket_id": event.payload_iket_id,
            "runtime_payload_iket_id": event.payload_iket_id if _runtime_payloads_enabled else 0,
        }
        for event in sorted(_events.values(), key=lambda item: item.event_id)
    ]


def mark(name: str, payload: Any = None):
    """Emit an IKET instant marker at the current program point."""
    payload_spec = _payload_spec(payload)
    event = _get_event(name, "mark", payload_spec=payload_spec)
    return _event_call(event, payload_spec)


def range_push(name: str, payload: Any = None):
    """Emit an IKET range-start event at the current program point."""
    payload_spec = _payload_spec(payload)
    event = _get_event(name, "range", payload_spec=payload_spec)
    _range_stack.append(name)
    return _event_call(event, payload_spec)


def range_pop(name: str | None = None):
    """Emit an IKET range-end event for the current warp-local range."""
    if not _range_stack:
        raise RuntimeError("T.iket.range_pop() called without a matching T.iket.range_push()")
    started = _range_stack.pop()
    if name is not None and name != started:
        raise RuntimeError(f"T.iket.range_pop({name!r}) does not match active range {started!r}")
    return tirx.call_extern("handle", "TL_IKET_EVENT", _RANGE_END_EVENT_ID)


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
    """Return a Python context manager that emits ``range_push/pop``.

    This is intended for TileLang frontend code:

    .. code-block:: python

        with T.iket.range("compute"):
            ...
    """
    return _RangeScope(name, payload=payload)


class _Session:
    def __init__(
        self,
        *,
        reset_events: bool = True,
        override: bool = True,
        disable_on_exit: bool = True,
        output_dir: str | os.PathLike[str] | None = None,
        runtime_payloads: bool | None = None,
    ):
        self.reset_events = reset_events
        self.override = override
        self.disable_on_exit = disable_on_exit
        self.output_dir = output_dir
        self.runtime_payloads = runtime_payloads
        self._previous_output_dir: Path | None = None
        self._previous_env_output_dir: str | None = None
        self._previous_runtime_payloads: bool | None = None

    def __enter__(self):
        self._previous_output_dir = _output_dir
        self._previous_env_output_dir = os.environ.get("TL_IKET_OUTPUT_DIR")
        self._previous_runtime_payloads = _runtime_payloads_enabled
        if self.output_dir is not None:
            set_output_dir(self.output_dir)
        if self.runtime_payloads is not None:
            if self.runtime_payloads:
                enable_runtime_payloads()
            else:
                disable_runtime_payloads()
        if self.reset_events:
            reset()
        enable(override=self.override)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.disable_on_exit:
            disable(restore=True)
        _restore_output_dir(self._previous_output_dir, self._previous_env_output_dir)
        _restore_runtime_payloads(self._previous_runtime_payloads)
        return False


def session(
    *,
    reset_events: bool = True,
    override: bool = True,
    disable_on_exit: bool = True,
    output_dir: str | os.PathLike[str] | None = None,
    runtime_payloads: bool | None = None,
) -> _Session:
    """Host-side helper for ``reset(); enable()`` around one compile."""
    return _Session(
        reset_events=reset_events,
        override=override,
        disable_on_exit=disable_on_exit,
        output_dir=output_dir,
        runtime_payloads=runtime_payloads,
    )


# Backward-compatible flat names for existing experiments.
iket_enable = enable
iket_disable = disable
iket_set_output_dir = set_output_dir
iket_enable_runtime_payloads = enable_runtime_payloads
iket_disable_runtime_payloads = disable_runtime_payloads
iket_reset = reset
iket_mark = mark
iket_range_push = range_push
iket_range_pop = range_pop
iket_range_start = range_start
iket_range_end = range_end


def _payload_spec(value: Any) -> PayloadSpec | None:
    if value is None:
        return None
    if isinstance(value, PayloadSpec):
        return value
    return payload(value)


def _identity_cuda_postproc(code: str, _target: Any) -> str:
    return code


def _restore_output_dir(previous_output_dir: Path | None, previous_env_output_dir: str | None) -> None:
    global _output_dir
    _output_dir = previous_output_dir
    if previous_env_output_dir is None:
        os.environ.pop("TL_IKET_OUTPUT_DIR", None)
    else:
        os.environ["TL_IKET_OUTPUT_DIR"] = previous_env_output_dir


def _restore_runtime_payloads(previous_runtime_payloads: bool | None) -> None:
    global _runtime_payloads_enabled
    if previous_runtime_payloads is not None:
        _runtime_payloads_enabled = previous_runtime_payloads


def _get_event(name: str, kind: str, payload_spec: PayloadSpec | None = None) -> _IketEvent:
    if not isinstance(name, str) or not name:
        raise ValueError("IKET event name must be a non-empty string")
    if kind not in ("mark", "range"):
        raise ValueError(f"Unsupported IKET event kind: {kind}")

    key = (kind, name)
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

    event = _IketEvent(
        name=name,
        event_id=event_id,
        kind=kind,
        range_id=range_id,
        payload_type=payload_spec.dtype if payload_spec is not None else "NoPayload",
        payload_iket_id=payload_spec.iket_id if payload_spec is not None else 0,
    )
    _events[key] = event
    return event


def _validate_payload_compat(event: _IketEvent, payload_spec: PayloadSpec | None) -> None:
    payload_type = payload_spec.dtype if payload_spec is not None else "NoPayload"
    if event.payload_type != payload_type:
        raise ValueError(
            f"IKET event {event.name!r} was first registered with payload type "
            f"{event.payload_type!r}, but is now used with {payload_type!r}"
        )


def _event_call(event: _IketEvent, payload_spec: PayloadSpec | None):
    if not _runtime_payloads_enabled or payload_spec is None:
        return tirx.call_extern("handle", "TL_IKET_EVENT", event.event_id)
    macro = "TL_IKET_EVENT_PAYLOAD_F32" if payload_spec.dtype == "float32" else "TL_IKET_EVENT_PAYLOAD_U32"
    return tirx.call_extern("handle", macro, event.event_id, payload_spec.expr)


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
    raise TypeError(
        "Cannot infer IKET payload dtype. Use T.iket.payload(expr, dtype='int32') "
        "with one of int32/uint32/int64/uint64/float32/float64/pointer."
    )


def _normalize_payload_dtype(dtype: str) -> str:
    aliases = {
        "int": "int32",
        "uint": "uint32",
        "float": "float32",
        "handle": "pointer",
        "void*": "pointer",
    }
    normalized = aliases.get(dtype.lower(), dtype.lower())
    _payload_iket_id(normalized)
    return normalized


def _validate_runtime_payload_dtype(dtype: str) -> None:
    if dtype not in ("int32", "uint32", "float32"):
        raise NotImplementedError(
            "TileLang IKET runtime payload capture currently supports only "
            "int32, uint32, and float32 payloads."
        )


def _payload_iket_id(dtype: str) -> int:
    ids = {
        "nopayload": 0,
        "int8": 1,
        "uint8": 2,
        "int16": 3,
        "uint16": 4,
        "int32": 5,
        "uint32": 6,
        "int64": 7,
        "uint64": 16,
        "float16": 11,
        "bfloat16": 12,
        "float32": 13,
        "float64": 14,
        "pointer": 15,
    }
    key = dtype.lower()
    if key not in ids:
        raise ValueError(f"Unsupported IKET payload dtype: {dtype!r}")
    return ids[key]


def _range_id(name: str) -> int:
    # IKET range ids are 32-bit values. CRC32 is deterministic and adequate for
    # this experimental frontend; collisions are possible but unlikely here.
    value = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    return value or 1


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
    return (
        f"__device__ __attribute__((used, aligned(1))) "
        f"unsigned char {symbol}[{len(data)}] = {{{values}}};"
    )


def _metadata_decls() -> str:
    if not _events:
        return ""

    lines = ['extern "C" {']
    decl_count = 0
    max_event_id = 0
    instrument_method = 3

    for event in _events.values():
        max_event_id = max(max_event_id, event.event_id)
        assignments = {
            4: _u32(event.event_id),
            8: _u32(instrument_method),
            12: _u32(event.payload_iket_id if _runtime_payloads_enabled else 0),
        }
        if event.kind == "range":
            assignments[16] = _u32(1)
            assignments[20] = _u32(event.range_id)
        symbol = f"__iket_evt_decl_{_safe_symbol(event.name)}_{event.event_id}_attrs"
        lines.append(_c_array(symbol, _attrs(60, assignments, event.name)))
        decl_count += 1

    for name, range_id in _ranges.items():
        symbol = f"__iket_range_decl_{_safe_symbol(name)}_{range_id}_attrs"
        lines.append(_c_array(symbol, _range_attrs(name, range_id)))
        decl_count += 1

    meta = [0] * 48
    meta[0:4] = _u32(48)
    meta[8:12] = _u32(decl_count)
    meta[12:16] = _u32(_RANGE_END_EVENT_ID)
    meta[16:20] = _u32(32)
    meta[20:24] = _u32(60)
    meta[24:28] = _IKET_MAGIC
    meta[32:36] = _u32(max_event_id)
    lines.append(_c_array("__iket_meta_info", meta))
    lines.append("}")
    return "\n".join(lines)


_IKET_EVENT_MACRO = r'''
#define TL_IKET_EVENT(ID) \
  asm volatile("{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "mov.b32 r, %cluster_ctarank;\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}" ::: "memory")

#define TL_IKET_EVENT_PAYLOAD_U32(ID, VALUE) do { \
  unsigned int __tl_iket_payload = (unsigned int)(VALUE); \
  /* Keep timestamp and payload as separate 32-bit STS instructions. IKET's \
     NativeDump payload patcher does not accept ptxas-fused STS.64 records. */ \
  asm volatile("{\n\t" \
               ".reg .b32 r, t;\n\t" \
               "mov.b32 r, %cluster_ctarank;\n\t" \
               "mov.u32 t, %globaltimer_lo;\n\t" \
               "or.b32 t, t, " #ID ";\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x20;\n\t" \
               "st.weak.shared.u32 [r], t;\n\t" \
               "}" ::: "memory"); \
  asm volatile("{\n\t" \
               ".reg .b32 r, p;\n\t" \
               "mov.b32 r, %cluster_ctarank;\n\t" \
               "mov.u32 p, %0;\n\t" \
               "mad.lo.u32 r, r, 0x1000000, 0x24;\n\t" \
               "st.volatile.shared.u32 [r], p;\n\t" \
               "pmevent.mask " #ID ";\n\t" \
               "}" :: "r"(__tl_iket_payload) : "memory"); \
} while (0)

#define TL_IKET_EVENT_PAYLOAD_F32(ID, VALUE) do { \
  union { float f; unsigned int u; } __tl_iket_payload; \
  __tl_iket_payload.f = (float)(VALUE); \
  TL_IKET_EVENT_PAYLOAD_U32(ID, __tl_iket_payload.u); \
} while (0)
'''


def _inject_iket_cuda(code: str) -> str:
    if not _events or "__tilelang_iket_frontend__" in code:
        return code

    insert_at = code.find("#include <tl_templates")
    if insert_at < 0:
        insert_at = code.find("#include")
    if insert_at < 0:
        insert_at = 0

    prefix = (
        "// __tilelang_iket_frontend__\n"
        "// Experimental IKET metadata emitted from TileLang frontend APIs.\n"
        + _metadata_decls()
        + "\n"
        + _IKET_EVENT_MACRO
        + "\n"
    )
    return code[:insert_at] + prefix + code[insert_at:]
