"""Small compiler-process helpers owned by TileLang."""


def py_str(output: bytes) -> str:
    """Decode compiler output with the existing UTF-8 diagnostic contract."""
    return output.decode("utf-8")
