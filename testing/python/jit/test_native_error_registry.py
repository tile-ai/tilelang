"""The native provider retains TileLang's named exception and catch contracts."""

import builtins
import importlib
import pickle
import subprocess
import sys
import textwrap

import pytest
import tvm_ffi
import tvm_ffi.testing
import tvm_tirx

import tilelang
from tilelang import errors
from tilelang import _provider


ERROR_BASES = {
    "InternalError": ("TVMError",),
    "RPCError": ("TVMError",),
    "RPCSessionTimeoutError": ("RPCError", "TimeoutError"),
    "OpError": ("TVMError",),
    "OpNotImplemented": ("OpError", "NotImplementedError"),
    "OpAttributeRequired": ("OpError", "AttributeError"),
    "OpAttributeInvalid": ("OpError", "AttributeError"),
    "OpAttributeUnImplemented": ("OpError", "NotImplementedError"),
    "DiagnosticError": ("TVMError",),
}


@pytest.mark.parametrize("name", ERROR_BASES)
def test_named_error_crosses_real_ffi_in_both_directions(name):
    cls = getattr(errors, name)
    assert cls.__bases__ == tuple(getattr(errors, base, getattr(builtins, base, None)) for base in ERROR_BASES[name])
    assert issubclass(cls, errors.TVMError)
    assert errors.TVMError.__bases__ == (RuntimeError,)

    converted = tvm_ffi.convert(cls("Python error"))
    assert converted.kind == name
    assert type(converted.py_error()) is cls
    assert converted.py_error().args == ("Python error",)

    with pytest.raises(cls, match=r"C\+\+ error") as raised:
        tvm_ffi.get_global_func("testing.test_raise_error")(name, "C++ error")
    assert type(raised.value) is cls
    assert raised.value.__tvm_ffi_error__.kind == name
    assert raised.value.__traceback__ is not None

    def callback():
        raise cls("callback error")

    with pytest.raises(cls, match="callback error") as callback_error:
        tvm_ffi.get_global_func("testing.apply")(callback)
    assert type(callback_error.value) is cls
    traceback = callback_error.value.__traceback__
    frames = []
    while traceback:
        frames.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "callback" in frames


@pytest.mark.parametrize(
    "name", ("RuntimeError", "ValueError", "TypeError", "AttributeError", "KeyError", "IndexError", "AssertionError", "MemoryError")
)
def test_builtin_error_registrations_keep_their_exact_types(name):
    cls = getattr(builtins, name)
    assert type(tvm_ffi.core.Error(name, "built-in", "").py_error()) is cls
    assert tvm_ffi.convert(cls("built-in")).kind == name
    with pytest.raises(cls) as raised:
        tvm_ffi.get_global_func("testing.test_raise_error")(name, "built-in")
    assert type(raised.value) is cls


def test_base_and_unknown_errors_do_not_gain_a_named_subclass():
    assert "TVMError" not in tvm_ffi.core.ERROR_NAME_TO_TYPE
    assert errors.TVMError not in tvm_ffi.core.ERROR_TYPE_TO_NAME
    assert tvm_ffi.convert(errors.TVMError("base")).kind == "RuntimeError"
    assert type(tvm_ffi.core.Error("UnregisteredCompilerError", "unknown", "").py_error()) is RuntimeError
    with pytest.raises(RuntimeError) as raised:
        tvm_ffi.get_global_func("testing.test_raise_error")("RuntimeError", "InternalError: text is not an error kind")
    assert type(raised.value) is RuntimeError


def test_real_compiler_error_has_the_canonical_type():
    assert not tvm_tirx.ir.Op.get("tirx.call_packed").has_attr("tilelang.testing.missing_error_attr")
    with pytest.raises(errors.InternalError, match="is not registered") as raised:
        tvm_tirx.ir.Op.get("tirx.call_packed").get_attr("tilelang.testing.missing_error_attr")
    assert type(raised.value) is errors.InternalError
    assert isinstance(raised.value, errors.TVMError)


def test_reload_and_repeated_bootstrap_preserve_canonical_classes():
    names = ("TVMError", *ERROR_BASES)
    previous = {name: getattr(errors, name) for name in names}
    instances = {name: cls("before reload") for name, cls in previous.items()}
    provider, library = _provider.load_provider()
    importlib.reload(errors)
    errors._ensure_error_registration()
    next_provider, next_library = _provider.load_provider()
    assert provider is next_provider is tvm_tirx
    assert library is next_library
    for name, cls in previous.items():
        assert getattr(errors, name) is cls
        assert type(pickle.loads(pickle.dumps(instances[name]))) is cls
    assert importlib.import_module("tilelang.errors") is errors
    assert not hasattr(tilelang, "tvm")
    assert "error" not in vars(tvm_tirx)
    assert "TVMError" not in vars(tvm_tirx)
    assert "tvm" not in sys.modules


def _subprocess(script, tmp_path):
    code = textwrap.dedent(script)
    (tmp_path / "case.py").write_text(code)
    command = [sys.executable, "-B", str(tmp_path / "case.py")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=40)
    (tmp_path / "stdout.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("first", ["tvm_tirx", "tilelang"])
def test_import_order_uses_one_native_provider_and_error_namespace(first, tmp_path):
    _subprocess(
        f"""
        import {first}
        import sys
        import importlib
        import tilelang
        import tvm_tirx
        import tvm_ffi
        from tilelang import errors
        from tilelang.errors import InternalError
        provider, library = tilelang._provider.load_provider()
        assert provider is tvm_tirx
        assert library is tilelang._LIB
        assert not hasattr(tilelang, "tvm")
        assert "error" not in vars(tvm_tirx)
        assert "TVMError" not in vars(tvm_tirx)
        assert InternalError is errors.InternalError
        assert type(tvm_ffi.core.Error("InternalError", "message", "").py_error()) is InternalError
        assert "tvm" not in sys.modules
        assert not any(name == "tilelang.tvm" or name.startswith("tilelang.tvm.") for name in sys.modules)
        assert importlib.import_module("tilelang.errors") is errors
        """,
        tmp_path,
    )


def test_preexisting_foreign_error_rejects_import_without_partial_registration(tmp_path):
    _subprocess(
        """
        import tvm_ffi
        class ForeignError(RuntimeError):
            pass
        tvm_ffi.register_error("DiagnosticError", ForeignError)
        before_names = dict(tvm_ffi.core.ERROR_NAME_TO_TYPE)
        before_types = dict(tvm_ffi.core.ERROR_TYPE_TO_NAME)
        try:
            import tilelang
        except ImportError as error:
            assert "DiagnosticError" in str(error)
        else:
            raise AssertionError("foreign error registration was accepted")
        assert tvm_ffi.core.ERROR_NAME_TO_TYPE == before_names
        assert tvm_ffi.core.ERROR_TYPE_TO_NAME == before_types
        """,
        tmp_path,
    )


@pytest.mark.parametrize("kind", ["name", "reverse", "reverse_foreign", "name_alias"])
def test_registration_conflict_is_rejected_without_writes(kind, monkeypatch):
    class ForeignError(RuntimeError):
        pass

    by_name = tvm_ffi.core.ERROR_NAME_TO_TYPE
    by_type = tvm_ffi.core.ERROR_TYPE_TO_NAME
    if kind == "name":
        monkeypatch.setitem(by_name, "DiagnosticError", ForeignError)
    elif kind == "reverse":
        monkeypatch.setitem(by_type, errors.DiagnosticError, "ForeignName")
    elif kind == "reverse_foreign":
        monkeypatch.setitem(by_type, ForeignError, "DiagnosticError")
    else:
        monkeypatch.setitem(by_name, "ForeignName", errors.DiagnosticError)
    before = dict(by_name), dict(by_type)
    with pytest.raises(ImportError, match="DiagnosticError"):
        errors._ensure_error_registration()
    assert (dict(by_name), dict(by_type)) == before


@pytest.mark.parametrize("kind", ["error", "TVMError"])
def test_bootstrap_preserves_foreign_provider_attributes(kind, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(tvm_tirx, kind, sentinel, raising=False)
    before = dict(tvm_ffi.core.ERROR_NAME_TO_TYPE), dict(tvm_ffi.core.ERROR_TYPE_TO_NAME)
    provider, library = _provider.load_provider()
    assert provider is tvm_tirx and library is tilelang._LIB
    assert (dict(tvm_ffi.core.ERROR_NAME_TO_TYPE), dict(tvm_ffi.core.ERROR_TYPE_TO_NAME)) == before
    assert getattr(tvm_tirx, kind) is sentinel


def test_enrichment_retains_identity_traceback_cause_and_other_args(tmp_path):
    directory = tmp_path / "path with spaces"
    directory.mkdir()
    source = directory / "kernel.py"
    source.write_text("def kernel():\n    invalid = 1\n")
    cause = ValueError("cause")
    try:
        raise errors.InternalError(f"compiler failed\n  --> {source}:2:5", "extra") from cause
    except errors.InternalError as error:
        traceback = error.__traceback__
        assert errors.enrich_error(error) is error
        assert error.__traceback__ is traceback
        assert error.__cause__ is cause
        assert error.args[1:] == ("extra",)
        assert "2 |     invalid = 1" in error.args[0]
        assert "^" in error.args[0]
        assert type(error) is errors.InternalError


@pytest.mark.parametrize("cls", [RuntimeError, TypeError, ValueError])
def test_enrichment_fallback_leaves_builtin_errors_unchanged(cls):
    error = cls("message without source marker", 17)
    assert errors.enrich_error(error) is error
    assert error.args == ("message without source marker", 17)
    assert type(error) is cls


def test_location_marker_keeps_windows_drive_and_optional_column():
    marker = errors._MARKER_RE.search("\n  --> C:\\work path\\kernel.py:12:3")
    assert marker.groupdict() == {"file": "C:\\work path\\kernel.py", "line": "12", "col": "3"}
    marker = errors._MARKER_RE.search("\n  --> /work path/kernel.py:12")
    assert marker.groupdict() == {"file": "/work path/kernel.py", "line": "12", "col": None}
