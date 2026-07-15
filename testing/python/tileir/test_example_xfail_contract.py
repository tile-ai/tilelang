"""Tests for the TileIR example-suite expected-failure contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tilelang.tileir.errors import TileIRLoweringNotImplementedError


def _load_examples_conftest():
    path = Path(__file__).parents[3] / "examples" / "conftest.py"
    spec = importlib.util.spec_from_file_location("tilelang_examples_conftest_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tileir_known_failure_markers_are_strict_and_exception_scoped(monkeypatch):
    conftest = _load_examples_conftest()

    class FakeItem:
        nodeid = "gemv/test_example_gemv.py::test_example_gemv"

        def __init__(self):
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    item = FakeItem()
    monkeypatch.setenv("TILELANG_TARGET", "tileir")

    conftest.pytest_collection_modifyitems(None, [item])

    assert len(item.markers) == 1
    marker = item.markers[0]
    assert marker.name == "xfail"
    assert marker.kwargs["strict"] is True
    assert marker.kwargs["raises"] is TileIRLoweringNotImplementedError
