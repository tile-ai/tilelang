"""Native layout annotation boundary shared by the frontend and lowering."""

from tilelang import _ffi_api


def encode_layout_payload(layout):
    """Return version 1's five-slot native Tuple, with bound coordinate variables."""
    return _ffi_api.EncodeLayoutPayload(layout)


def decode_layout_payload(payload, kind, schema_version=1):
    """Recover a Layout, Fragment, or PartialFragment from a native payload."""
    return _ffi_api.DecodeLayoutPayload(payload, kind, schema_version)
