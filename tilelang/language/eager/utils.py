from __future__ import annotations
import ast
import inspect
import os
from typing import Any, Literal
from collections.abc import Callable, Sequence
from tilelang import env
from hashlib import sha256
from tvm import tirx
import linecache


def disk_compile(source, name):
    cache_dir = env.TILELANG_CACHE_DIR
    if cache_dir is not None:
        import os

        save_dir = os.path.join(cache_dir, "py-cache")
        os.makedirs(save_dir, exist_ok=True)
        hash_sfx = sha256(source.encode("utf-8")).hexdigest()[:8]
        path = os.path.join(save_dir, f"{name}.{hash_sfx}.py")
        with open(path, "w") as f:
            f.write(source)
    linecache.cache[path] = (len(source), None, source.splitlines(), path)
    return compile(source, path, "exec")


def _remove_leading_ident(source: str):
    lines = source.splitlines()
    if not lines:
        return source
    ident_size = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join([line[ident_size:] if len(line) >= ident_size else line for line in lines])


def get_func_nonlocals(func):
    """A modified version of `inspect.getclosurevars`"""

    if inspect.ismethod(func):
        func = func.__func__

    if not inspect.isfunction(func):
        raise TypeError(f"{func!r} is not a Python function")

    code = func.__code__
    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    nonlocal_vars = {}
    if func.__closure__ is not None:
        for var, cell in zip(code.co_freevars, func.__closure__):
            try:
                nonlocal_vars[var] = cell.cell_contents
            except ValueError as err:
                # cell_contents may raise ValueError if the cell is empty.
                if "empty" not in str(err):
                    raise
    return nonlocal_vars


def get_ast(func: Callable):
    _, start = inspect.getsourcelines(func)
    filename = inspect.getsourcefile(func) or inspect.getfile(func)
    # Normalize to an absolute path so that source spans and remapped
    # tracebacks are clickable links regardless of the invocation cwd.
    filename = os.path.abspath(filename)
    source = inspect.getsource(func)
    source = _remove_leading_ident(source)
    source = "\n" * (start - 1) + source
    tree = ast.parse(source, filename=filename)
    return tree


CompileMethod = Literal["direct", "disk"]


def get_compiled_object(source: str | ast.AST, name: str, filename: str = None, globals: dict[str, Any] = None):
    if isinstance(source, ast.AST):
        assert filename is not None, "filename must be provided when source is an AST"
    try:
        if isinstance(source, ast.AST):
            ast.fix_missing_locations(source)
            compiled = compile(source, filename, "exec")
        else:
            compiled = disk_compile(source, name)
    except Exception as e:
        source_str = source if isinstance(source, str) else ast.unparse(source)
        raise RuntimeError(f"Failed to compile source for {name}, Error: {e}:\n{source_str}") from e
    locs = {}
    exec(compiled, globals, locs)
    return locs[name]


def construct_strides(shape: Sequence[Any], allow_prim_expr: bool = True) -> tuple[Any, ...]:
    """Construct contiguous row-major strides from ``shape``.

    This is the single source of truth for default (contiguous) strides in the
    Python frontend; ``T.Tensor``, ``T.out`` and ``retrieve_stride`` all route
    here so they cannot drift apart.

    The result always has the same rank as ``shape``. In particular a rank-0
    (scalar) shape yields an empty stride tuple, matching the TIR convention
    that ``len(buffer.strides)`` is either ``0`` or ``len(buffer.shape)``.
    Entries may be ``int`` or ``PrimExpr`` depending on ``shape``; pass
    ``allow_prim_expr=False`` to require fully static strides.
    """
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
        if not allow_prim_expr and isinstance(stride, tirx.PrimExpr):
            raise ValueError("Cannot construct strides with PrimExpr when allow_prim_expr is False.")
    return tuple(reversed(strides))
