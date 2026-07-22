"""Language dialect registry and overlay helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from types import ModuleType
from collections.abc import Iterable

from tilelang.env import env


@dataclass(frozen=True)
class LanguageDialect:
    """A named language surface layered on top of TileLang core syntax."""

    name: str
    module: str
    aliases: tuple[str, ...] = ()
    description: str = ""


_DIALECTS: dict[str, LanguageDialect] = {}
_ALIASES: dict[str, str] = {}


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if not normalized:
        return "core"
    if normalized in {"default", "generic", "base", "none", "off"}:
        return "core"
    return normalized


def register_language_dialect(
    name: str,
    module: str,
    *,
    aliases: Iterable[str] = (),
    description: str = "",
    override: bool = False,
) -> LanguageDialect:
    """Register a language dialect module.

    The module should export a language namespace, typically by re-exporting
    ``tilelang.language`` and adding backend-specific operators or intrinsics.
    """

    canonical = _normalize_name(name)
    if canonical in _DIALECTS and not override:
        raise ValueError(f"Language dialect {canonical!r} is already registered.")

    alias_tuple = tuple(_normalize_name(alias) for alias in aliases)
    spec = LanguageDialect(canonical, module, alias_tuple, description)
    _DIALECTS[canonical] = spec
    _ALIASES[canonical] = canonical
    for alias in alias_tuple:
        if alias in _ALIASES and _ALIASES[alias] != canonical and not override:
            raise ValueError(f"Language dialect alias {alias!r} is already registered.")
        _ALIASES[alias] = canonical
    return spec


def list_language_dialects() -> tuple[LanguageDialect, ...]:
    """Return the registered language dialects in stable name order."""

    return tuple(_DIALECTS[name] for name in sorted(_DIALECTS))


def resolve_language_dialect(name: str | None = None) -> LanguageDialect:
    """Resolve a dialect name or alias to its registration record."""

    if name is None:
        name = get_default_language_dialect()
    normalized = _normalize_name(name)
    canonical = _ALIASES.get(normalized, normalized)
    try:
        return _DIALECTS[canonical]
    except KeyError as exc:
        available = ", ".join(sorted(_DIALECTS))
        raise ValueError(f"Unknown TileLang language dialect {name!r}. Available dialects: {available}.") from exc


def resolve_language_module(name: str | None = None) -> ModuleType:
    """Import and return the module implementing a language dialect."""

    spec = resolve_language_dialect(name)
    return importlib.import_module(spec.module)


def get_default_language_dialect() -> str:
    """Return the default dialect requested by ``TILELANG_DEFAULT_DIALECT``."""

    value = env.TILELANG_DEFAULT_DIALECT
    if value is None:
        return "core"
    return resolve_language_dialect(str(value)).name


def set_default_language_dialect(name: str) -> ModuleType:
    """Set the process-local default dialect and return its module."""

    spec = resolve_language_dialect(name)
    env.TILELANG_DEFAULT_DIALECT = spec.name
    module = resolve_language_module(spec.name)
    tilelang_pkg = sys.modules.get("tilelang")
    if tilelang_pkg is not None:
        tilelang_pkg.language = module
    return module


def export_module_symbols(module: ModuleType, namespace: dict[str, object]) -> None:
    """Copy exported module attributes into ``namespace``.

    If the source module has ``__all__``, it is honored. Otherwise all non-dunder
    names are copied, including TileLang spellings like ``__ldg``.
    """

    names = getattr(module, "__all__", None)
    if names is None:
        names = [
            name
            for name in dir(module)
            if not (name.startswith("__") and name.endswith("__")) and (not name.startswith("_") or name.startswith("__"))
        ]
    for name in names:
        namespace[name] = getattr(module, name)


def export_core_language_symbols(namespace: dict[str, object]) -> None:
    """Copy the current core ``tilelang.language`` surface into ``namespace``."""

    core = importlib.import_module("tilelang.language")
    export_module_symbols(core, namespace)


register_language_dialect(
    "core",
    "tilelang.language",
    aliases=("generic", "base"),
    description="Backend-neutral TileLang language surface.",
    override=True,
)
register_language_dialect(
    "cuda",
    "tilelang.cuda.language",
    aliases=("nvidia", "cu"),
    description="TileLang core language plus CUDA-specific operators and intrinsics.",
    override=True,
)
register_language_dialect(
    "rocm",
    "tilelang.rocm.language",
    aliases=("hip", "amd"),
    description="TileLang core language plus ROCm/HIP-specific operators and intrinsics.",
    override=True,
)
register_language_dialect(
    "cpu",
    "tilelang.cpu.language",
    aliases=("c", "llvm"),
    description="TileLang core language for CPU targets.",
    override=True,
)
register_language_dialect(
    "metal",
    "tilelang.metal.language",
    description="TileLang core language plus Metal-specific operators and intrinsics.",
    override=True,
)
register_language_dialect(
    "webgpu",
    "tilelang.webgpu.language",
    aliases=("wgsl",),
    description="TileLang core language for WebGPU/WGSL targets.",
    override=True,
)
