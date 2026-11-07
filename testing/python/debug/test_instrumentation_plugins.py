"""Optional tools register through entry points without a devkit dependency."""

from importlib import metadata
import sys
from types import ModuleType

import pytest

from tilelang.instrumentation import (
    PassInstrumentationTool,
    compile_pass_instrumentation,
    register_pass_instrumentation_tool,
    unregister_pass_instrumentation_tool,
)
from tilelang.instrumentation.plugins import load_instrumentation_plugins


def test_plugin_registers_per_compile_tool(monkeypatch):
    plugin = ModuleType("test_instrumentation_plugin")
    plugin.activate = lambda: register_pass_instrumentation_tool("test-plugin", PassInstrumentationTool)
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    entry = metadata.EntryPoint(name="test-plugin", value=f"{plugin.__name__}:activate", group="tilelang.instrumentation_tools")
    monkeypatch.setattr(metadata, "entry_points", lambda *, group: [entry] if group == entry.group else [])

    try:
        load_instrumentation_plugins()
        with compile_pass_instrumentation() as first:
            first_tool = first.find_tool(PassInstrumentationTool)
            assert first_tool is not None
            with compile_pass_instrumentation() as nested:
                assert nested is first
        with compile_pass_instrumentation() as second:
            assert second.find_tool(PassInstrumentationTool) is not first_tool
    finally:
        unregister_pass_instrumentation_tool("test-plugin")


def test_no_installed_plugins(monkeypatch):
    monkeypatch.setattr(metadata, "entry_points", lambda *, group: [])
    load_instrumentation_plugins()
    with compile_pass_instrumentation(include_default_tools=False) as session:
        assert session.tools == ()


def test_broken_plugin_does_not_block_other_tools(monkeypatch):
    plugin = ModuleType("test_instrumentation_plugin")
    activated = []
    plugin.activate = lambda: activated.append(True)
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    entries = [
        metadata.EntryPoint(name="a-broken", value="_tilelang_missing_plugin:activate", group="tilelang.instrumentation_tools"),
        metadata.EntryPoint(name="b-working", value=f"{plugin.__name__}:activate", group="tilelang.instrumentation_tools"),
    ]
    monkeypatch.setattr(metadata, "entry_points", lambda *, group: entries)
    with pytest.warns(RuntimeWarning, match="a-broken"):
        load_instrumentation_plugins()
    assert activated == [True]
