"""Tests for the TileIR example-suite expected-failure contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tilelang.tileir.errors import TileIRLoweringNotImplementedError


def _load_examples_conftest():
    path = Path(__file__).parents[3] / "examples" / "conftest.py"
    spec = importlib.util.spec_from_file_location("tilelang_examples_conftest_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "target,nodeid,known_failure",
    [
        ("tileir", "gemv/test_example_gemv.py::test_example_gemv", "gemv/test_example_gemv.py::test_example_gemv"),
        (
            "tileir -arch=sm_100",
            "examples/gemv/test_example_gemv.py::test_example_gemv[128]",
            "gemv/test_example_gemv.py::test_example_gemv",
        ),
        (
            "tileir",
            "examples/flash_attention/test_example_flash_attention.py::test_example_gqa_bwd_tma_reduce_varlen",
            "flash_attention/test_example_flash_attention.py::test_example_gqa_bwd_tma_reduce_varlen",
        ),
        ("tileir", "gemv/test_example_gemv.py::test_example_gemv_new_variant", None),
        ("tileir", "examples/aws/test_example_aws.py::test_example_gemm", None),
        ("tileir", "examples/aws/test_example_aws.py::test_example_gemm_manual", None),
        ("tileir", "examples/deepseek_mhc/test_example_mhc.py::test_mhc_pre", None),
        ("tileir", "examples/attention_sink/test_example_attention_sink.py::test_example_gqa_sink_bwd_bhsd", None),
        ("cuda", "gemv/test_example_gemv.py::test_example_gemv", None),
    ],
)
def test_tileir_known_failure_markers_are_strict_and_exception_scoped(monkeypatch, target, nodeid, known_failure):
    conftest = _load_examples_conftest()

    class FakeItem:
        def __init__(self):
            self.nodeid = nodeid
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    item = FakeItem()
    monkeypatch.setenv("TILELANG_TARGET", target)

    conftest.pytest_collection_modifyitems(None, [item])

    if known_failure is None:
        assert item.markers == []
        return

    assert len(item.markers) == 1
    marker = item.markers[0]
    assert marker.name == "xfail"
    assert marker.kwargs["strict"] is True
    assert marker.kwargs["raises"] is TileIRLoweringNotImplementedError
    assert marker.kwargs["reason"] == f"TileIR: {conftest.TILEIR_KNOWN_FAILURES[known_failure]}"
