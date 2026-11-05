"""AutoDD subprocesses must keep the selected interpreter and compiler provider."""

import asyncio
import json
import os
from pathlib import Path
import sys
import textwrap

import pytest
import tilelang
import tvm_ffi
import tvm_tirx

from tilelang.autodd import SubProcRunner


@pytest.mark.skipif(os.name == "nt", reason="This PATH counterexample uses a POSIX executable; Windows runs in its platform matrix")
def test_subprocess_keeps_native_environment_when_path_selects_another_python(tmp_path, monkeypatch):
    alternate = tmp_path / "another-environment"
    alternate.mkdir()
    for name in ("python", "python3"):
        executable = alternate / name
        executable.write_text("#!/bin/sh\necho wrong-interpreter >&2\nexit 97\n")
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(alternate) + os.pathsep + os.environ.get("PATH", ""))
    source = textwrap.dedent("""\
        import json
        import sys
        import tilelang
        import tvm_ffi
        import tvm_tirx

        function = tvm_tirx.tirx.Function([], [tvm_tirx.ir.Evaluate(0)])
        print("AUTODD_IDENTITY=" + json.dumps({
            "python": sys.executable,
            "tilelang": tilelang.__file__,
            "tirx": tvm_tirx.__file__,
            "ffi": tvm_ffi.__file__,
            "compiler": dict(tvm_ffi.get_global_func("tl.CompilerBuildInfo")()),
            "libraries": {name: library._name for name, library in tvm_tirx.base._LOADED_LIBS.items()},
            "function_type": type(function).__name__,
            "legacy_imported": "tvm" in sys.modules,
        }))
    """)
    stdout, stderr, success = asyncio.run(SubProcRunner().run(source, timeout=30.0))
    assert success, stdout + stderr
    messages = [line.removeprefix("AUTODD_IDENTITY=") for line in stdout.splitlines() if line.startswith("AUTODD_IDENTITY=")]
    assert len(messages) == 1, stdout
    actual = json.loads(messages[0])
    (tmp_path / "child_identity.json").write_text(json.dumps(actual, indent=2) + "\n")
    assert actual["python"] == sys.executable
    for key, module in (("tilelang", tilelang), ("tirx", tvm_tirx), ("ffi", tvm_ffi)):
        assert Path(actual[key]).resolve() == Path(module.__file__).resolve()
    assert actual["compiler"] == dict(tvm_ffi.get_global_func("tl.CompilerBuildInfo")())
    assert actual["libraries"] == {name: library._name for name, library in tvm_tirx.base._LOADED_LIBS.items()}
    assert actual["function_type"] == "Function"
    assert not actual["legacy_imported"]
    assert "wrong-interpreter" not in stderr
