"""Debug source loss never changes an already loaded module's executable identity."""

from pathlib import Path

import pytest
from tvm_tirx import ir, tirx

from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter


_MODULE = """
_native_host_schema = 1
_native_cutedsl_schema = 1
_function_names = []
_kernel_specs = {}
_cutedsl_compile_specs = {}
_cutedsl_device_source = "# actual device source"
_cutedsl_arch = "sm_100"

def call(kernels, *, stream=0, device_id=0):
    return 17
"""


def _restore(path):
    function = tirx.Function([], ir.SeqStmt([ir.Evaluate(ir.const(0))]), ir.TupleType([]))
    function = function.with_attr("global_symbol", "main")
    return CuTeDSLKernelAdapter.from_database(
        params=[],
        result_idx=[],
        target={"kind": "cutedsl", "arch": "sm_100"},
        func_or_mod=function,
        host_kernel_source="# host wrapper",
        device_kernel_source="# device kernel",
        kernel_lib_path=str(path),
    )


def _assert_separate_sources(adapter, expected):
    assert adapter.get_generated_module_source() == expected
    assert adapter.get_host_source() == "# host wrapper"
    assert adapter.get_kernel_source(kernel_only=True) == "# device kernel"
    assert adapter.get_kernel_source(kernel_only=False) == "# device kernel\n\n# host wrapper"
    assert adapter.pymodule.call({}) == 17


def test_real_import_survives_later_source_removal(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text(_MODULE + "\nfrom pathlib import Path\nPath(__file__).unlink()\n")
    restored = _restore(path)
    assert not path.exists()
    _assert_separate_sources(restored, None)


@pytest.mark.parametrize("error", [PermissionError("debug source denied"), OSError("debug source unavailable")])
def test_debug_source_oserror_does_not_replace_a_real_loaded_module(tmp_path, monkeypatch, error):
    path = tmp_path / "kernel.py"
    path.write_text(_MODULE)
    original = Path.read_text

    def read_text(value, *args, **kwargs):
        if value == path:
            raise error
        return original(value, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    restored = _restore(path)
    assert path.exists()
    _assert_separate_sources(restored, None)


def test_present_source_is_read_as_the_actual_generated_module(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text(_MODULE)
    restored = _restore(path)
    _assert_separate_sources(restored, _MODULE)


@pytest.mark.parametrize("failure", ["missing", "syntax", "schema"])
def test_actual_module_loading_failures_are_never_ignored(tmp_path, failure):
    path = tmp_path / "kernel.py"
    expected = FileNotFoundError
    if failure == "syntax":
        path.write_text("def invalid syntax\n")
        expected = SyntaxError
    elif failure == "schema":
        path.write_text(_MODULE.replace("_native_host_schema = 1", "_native_host_schema = 999"))
        expected = ValueError
    with pytest.raises(expected):
        _restore(path)


def test_non_oserror_from_source_read_is_not_suppressed(tmp_path, monkeypatch):
    path = tmp_path / "kernel.py"
    path.write_text(_MODULE)
    original = Path.read_text

    def read_text(value, *args, **kwargs):
        if value == path:
            raise UnicodeDecodeError("utf8", b"\xff", 0, 1, "invalid source")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(UnicodeDecodeError):
        _restore(path)
