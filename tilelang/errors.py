"""Canonical TileLang compiler exceptions and source-location hints.

The named exceptions retain the established catch hierarchy while the native
provider and TVM FFI own the compiler objects and error transport. Registration
is installed by the provider bootstrap, after checking the provider identity.

C++ passes append a machine-readable location line to error messages when the
relevant IR node carries a source span (see `src/span_utils.h`):

    \n  --> /abs/path/to/kernel.py:21:1

`enrich_error` rewrites the exception message in place (same exception type)
to a clang/python-style snippet:

    tilelang.errors.InternalError: The layout for fragment C_local ...
      --> /abs/path/to/kernel.py:21:1
       |
    21 |     C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
       |     ^
"""

import os
import re

# Re-executing this module must not invalidate an exception class already held
# by user code or registered with TVM FFI. Keep ordinary top-level class names
# for traceback display and pickling, and restore their canonical bindings on
# reload even if a caller has overwritten a module attribute.
if "_ERROR_CLASSES" not in globals():

    class TVMError(RuntimeError):
        """Base class for named TileLang compiler errors."""

    class InternalError(TVMError):
        """An internal compiler invariant was violated."""

    class RPCError(TVMError):
        """A remote invocation failed."""

    class RPCSessionTimeoutError(RPCError, TimeoutError):
        """A remote invocation exceeded its session lifetime."""

    class OpError(TVMError):
        """Base class for operator errors."""

    class OpNotImplemented(OpError, NotImplementedError):
        """An operator has no supported implementation."""

    class OpAttributeRequired(OpError, AttributeError):
        """An operator is missing a required attribute."""

    class OpAttributeInvalid(OpError, AttributeError):
        """An operator attribute has an invalid value."""

    class OpAttributeUnImplemented(OpError, NotImplementedError):
        """An operator attribute is unsupported."""

    class DiagnosticError(TVMError):
        """Compiler diagnostics reported an error."""

    _ERROR_CLASSES = {
        cls.__name__: cls
        for cls in (
            TVMError,
            InternalError,
            RPCError,
            RPCSessionTimeoutError,
            OpError,
            OpNotImplemented,
            OpAttributeRequired,
            OpAttributeInvalid,
            OpAttributeUnImplemented,
            DiagnosticError,
        )
    }
else:
    globals().update(_ERROR_CLASSES)


def _ensure_error_registration():
    """Register the named subclasses without replacing a foreign binding.

    TVM FFI's public registration API overwrites existing entries. Its fixed
    registry tables therefore need a complete preflight before any write.
    TVMError itself was not registered by the previous compiler: an exact
    TVMError instance still travels as RuntimeError, as do unknown errors.
    """
    import tvm_ffi

    by_name = tvm_ffi.core.ERROR_NAME_TO_TYPE
    by_type = tvm_ffi.core.ERROR_TYPE_TO_NAME
    registered = {name: cls for name, cls in _ERROR_CLASSES.items() if name != "TVMError"}
    for name, cls in registered.items():
        if name in by_name and by_name[name] is not cls:
            raise ImportError(f"TVM FFI error name {name!r} is already owned by another exception class")
        if cls in by_type and by_type[cls] != name:
            raise ImportError(f"TileLang error class {name!r} is already registered under another name")
        if any(other_cls is not cls and other_name == name for other_cls, other_name in by_type.items()):
            raise ImportError(f"TVM FFI error name {name!r} has a conflicting reverse registration")
        if any(other_name != name and other_cls is cls for other_name, other_cls in by_name.items()):
            raise ImportError(f"TileLang error class {name!r} has a conflicting name alias")
    for name, cls in registered.items():
        if by_name.get(name) is not cls or by_type.get(cls) != name:
            tvm_ffi.register_error(name, cls)


__all__ = [*_ERROR_CLASSES, "enrich_error"]

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
