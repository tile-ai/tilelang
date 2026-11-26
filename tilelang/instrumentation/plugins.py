"""Discover optional compiler tools through Python package entry points.

The ``tilelang.instrumentation_tools`` group contains zero-argument activation
callbacks. TileLang calls them once during normal package import, after compiler
APIs are available. Plugins own their configuration and should register tool
factories only when enabled; a disabled plugin should avoid heavy imports.
"""

from __future__ import annotations

import warnings


def load_instrumentation_plugins() -> None:
    """Activate installed tools without making them required dependencies."""
    from importlib.metadata import entry_points

    for entry in sorted(entry_points(group="tilelang.instrumentation_tools"), key=lambda item: item.name):
        try:
            entry.load()()
        except Exception as error:
            warnings.warn(
                f"Could not activate TileLang instrumentation plugin {entry.name!r}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
