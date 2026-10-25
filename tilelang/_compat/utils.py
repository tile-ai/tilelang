"""Small host-side utilities independent of the compiler package layout."""


def py_str(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value
