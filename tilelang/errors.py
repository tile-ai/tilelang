"""Render source-location hints embedded in compiler error messages.

C++ passes append a machine-readable location line to error messages when the
relevant IR node carries a source span (see `src/span_utils.h`):

    \n  --> /abs/path/to/kernel.py:21:1

`enrich_error` rewrites the exception message in place (same exception type)
to a clang/python-style snippet:

    tvm.error.InternalError: The layout for fragment C_local ...
      --> /abs/path/to/kernel.py:21:1
       |
    21 |     C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
       |     ^
"""

import os
import re

# The file part is lazy so the trailing ":line[:col]" is split off from the
# right: greedy matching would swallow ":line" into the file group, while
# lazy matching still accommodates paths with spaces ("/path with spaces/")
# and Windows drive letters ("C:\work\kernel.py") via backtracking.
_MARKER_RE = re.compile(r"\n\s*-->\s+(?P<file>.+?):(?P<line>\d+)(?::(?P<col>\d+))?\s*$")


def _render_snippet(file: str, line: int, col: int) -> str | None:
    """Render the `--> file:line:col` header plus the source line and caret."""
    try:
        if not os.path.isfile(file):
            return None
        with open(file, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if not (1 <= line <= len(lines)):
            return None
        text = lines[line - 1]
    except OSError:
        return None
    gutter = str(line)
    # Line-level spans carry no column; point at the first non-blank
    # character instead of the line start.
    caret_col = max(col, 1) if col > 1 else len(text) - len(text.lstrip()) + 1
    return f"  --> {file}:{line}:{col}\n   {' ' * len(gutter)}|\n {gutter} | {text}\n   {' ' * len(gutter)}| {' ' * (caret_col - 1)}^"


def enrich_error(exc: BaseException) -> BaseException:
    """Append a rendered source snippet to a compiler error, when available.

    The exception object is returned unchanged (type and traceback preserved);
    only its message args are rewritten. Rendering failures fall back to the
    original message silently.
    """
    try:
        if not exc.args or not isinstance(exc.args[0], str):
            return exc
        message = exc.args[0]
        match = _MARKER_RE.search(message)
        if match is None:
            return exc
        file = match.group("file")
        line = int(match.group("line"))
        col = int(match.group("col") or 1)
        snippet = _render_snippet(file, line, col)
        if snippet is None:
            return exc
        # Replace the bare marker line with the full snippet.
        message = message[: match.start()] + "\n" + snippet
        exc.args = (message, *exc.args[1:])
    except Exception:
        pass
    return exc
