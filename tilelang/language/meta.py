"""Turn Python classes/methods into TIR, in both eager and lazy modes.

Two DSL primitives -- ``inline`` (method -> inlined TIR) and ``meta_class``
(class of JIT-time stateful helpers) -- that behave the same in eager
(``@tilelang.jit``) and lazy (``@T.prim_func``) frontends. See each function's
docstring for the details.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, TypeVar

_C = TypeVar("_C", bound=type)


class _InlineMethod:
    """Descriptor that inlines a method, dispatching eager vs lazy per call."""

    def __init__(self, func: Callable) -> None:
        self._func = func
        self._eager = None  # eager macro, built lazily
        self._lazy = None  # TVMScript parser inline, built lazily
        self.__name__ = getattr(func, "__name__", "inline")
        self.__doc__ = getattr(func, "__doc__", None)

    def _eager_macro(self):
        if self._eager is None:
            from tilelang.language.eager.builder import macro

            self._eager = macro(self._func)
        return self._eager

    def _lazy_macro(self):
        if self._lazy is None:
            from tvm.tirx.script.parser.entry import inline as _tvm_inline

            self._lazy = _tvm_inline(self._func)
        return self._lazy

    def _dispatch(self, obj: Any, *args: Any, **kwargs: Any):
        from tilelang.language.eager.builder import Builder

        if Builder.current() is not None:
            return self._eager_macro()(obj, *args, **kwargs)
        return self._lazy_macro()(obj, *args, **kwargs)

    def __get__(self, obj: Any, owner: type | None = None):
        if obj is None:
            return self
        return lambda *args, **kwargs: self._dispatch(obj, *args, **kwargs)


def inline(func: Callable) -> _InlineMethod:
    """Decorator: lower a method body to TIR, inlined at each call site.

    ``self`` is bound automatically. The lowering engine is chosen per call so
    the same method works in both frontends:

    - eager mode (an eager ``Builder`` is active) -> the eager ``macro`` engine;
    - lazy mode (no eager builder; a TVMScript parser is active) -> TVM's
      parser-level ``inline`` (``TIRInline``).

    Both engines are built lazily on first use of the respective mode.

    Inside an inlined method, read scalar state via ``self.x[0]`` and write via
    ``self.x[0] = ...``; the store lowers to ``BufferStore`` through whichever
    engine is active, so no core builder/parser changes are needed.
    """
    return _InlineMethod(func)


def _emits_store(func: Callable) -> bool:
    """True if ``func`` contains a buffer store (subscript-target assignment).

    A subscript target ``self.x[...] = ...`` (or its augmented form) is how state
    methods lower to ``BufferStore``, so such methods must be inlined. Methods
    without any store only build/return ``PrimExpr`` values and are kept plain
    Python (they work in both modes and can be reused statelessly).
    """
    try:
        src = textwrap.dedent(inspect.getsource(func))
        fn = ast.parse(src).body[0]
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for tgt in targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Subscript):
                        return True
        return False
    except Exception:  # noqa: BLE001 - if we cannot tell, default to inlining
        return True


def _name_buffer(prefix: str, attr: str, value: Any) -> None:
    """Give a state buffer a readable name ``{prefix}_{attr}`` in the IR."""
    from tvm.script.ir_builder import IRBuilder
    from tvm.tirx import Buffer

    if not isinstance(value, Buffer):
        return
    with contextlib.suppress(Exception):  # naming is best-effort, never fatal
        IRBuilder.name(f"{prefix}_{attr}", value)


def _install_prefix_naming(cls: type) -> None:
    """Make ``self.<attr> = <buffer>`` auto-name the buffer using ``prefix``.

    ``prefix`` is read from the constructor argument of the same name (if any),
    so existing ``__init__(self, prefix, ...)`` signatures just work.
    """
    if cls.__dict__.get("_tl_prefix_naming_installed", False):
        return

    orig_init = cls.__init__
    init_sig = inspect.signature(orig_init)

    def __init__(self, *args, **kwargs):
        prefix = None
        try:
            bound = init_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            prefix = bound.arguments.get("prefix")
        except TypeError:
            prefix = None
        object.__setattr__(self, "_meta_prefix", prefix)
        orig_init(self, *args, **kwargs)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        prefix = getattr(self, "_meta_prefix", None)
        if prefix and not name.startswith("_"):
            _name_buffer(prefix, name, value)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__
    cls._tl_prefix_naming_installed = True


def meta_class(cls: _C) -> _C:
    """Class decorator for JIT-time stateful helpers (e.g. tile schedulers).

    Instances exist only during JIT tracing / parsing and hold ``T.alloc_var``
    buffers as state (``T.alloc_var`` emits into the active IR frame in both
    modes). The decorator has three responsibilities:

    1. Mark the class with ``_is_meta_class``. The lazy parser needs this to
       bind ``sched = Sched(...)`` as a (non-TIR) instance in its scope instead
       of trying to turn it into a constant.
    2. Auto-``inline`` every TIR-emitting method, so methods need no per-method
       ``@inline``. A method is considered TIR-emitting iff it contains a
       buffer store, i.e. a subscript-target assignment ``self.x[...] = ...``
       (see ``_emits_store``). Left as plain Python:

       - dunders (``__init__`` allocates state as plain Python),
       - ``staticmethod`` / ``classmethod`` / ``property``,
       - methods already decorated with ``@inline``,
       - methods with no buffer store -- pure compile-time helpers that only
         build/return ``PrimExpr`` expressions (e.g. ``valid`` returning a
         loop condition, or a ``coord(tile_id)`` decode returning ``(m, n)``).
         These stay plain so they can return values and be reused statelessly.
         A method that emits TIR only through control flow / nested calls (no
         direct store) should be marked explicitly with ``@inline``.
    3. Auto-name state buffers ``{prefix}_{attr}`` in the generated IR, where
       ``prefix`` is the constructor argument named ``prefix`` (so
       ``Sched("sched", ...)`` yields ``sched_m_idx`` etc.). This wraps
       ``__init__`` to capture ``prefix`` and installs a ``__setattr__`` that
       names any ``Buffer`` assigned to a non-underscore attribute; non-buffers
       (compile-time ints like ``num_n_tiles``) and underscore attributes are
       skipped, and naming is best-effort (never fatal). It runs at
       construction while an IR builder is active, so it works in both modes.
    """
    cls._is_meta_class = True
    _install_prefix_naming(cls)
    for name, attr in list(cls.__dict__.items()):
        if name.startswith("__"):
            continue
        if isinstance(attr, _InlineMethod):
            continue
        if isinstance(attr, (staticmethod, classmethod, property)):
            continue
        if inspect.isfunction(attr) and _emits_store(attr):
            setattr(cls, name, inline(attr))
    return cls
