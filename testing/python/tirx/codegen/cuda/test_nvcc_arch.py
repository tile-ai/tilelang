from pathlib import Path
from types import SimpleNamespace

import pytest

from tilelang.contrib import nvcc
from tilelang._compat.target import native_backend_target, normalize_target
from tvm_tirx.backend.cuda.backend_config import resolve_backend_config
from tvm_tirx.target import Target, TargetKind


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100a", "sm_103a", "sm_100f"])
@pytest.mark.parametrize("output_format", ["fatbin", "ptx", "cubin"])
@pytest.mark.parametrize("explicit", [True, False])
def test_nvcc_preserves_architecture_qualifiers(monkeypatch, arch, output_format, explicit):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"test-compiler-output")
        return SimpleNamespace(returncode=0, communicate=lambda **kwargs: (b"", None))

    monkeypatch.setattr(nvcc.subprocess, "Popen", run)
    monkeypatch.setattr(nvcc, "get_nvcc_compiler", lambda: "nvcc")
    monkeypatch.setattr(nvcc, "get_cplus_compiler", lambda: None)
    with Target({"kind": "cuda", "arch": arch}):
        artifact = nvcc.compile_cuda("", output_format, arch if explicit else None)
    assert artifact == b"test-compiler-output"
    command = commands[0]
    virtual_arch = "compute_" + arch.removeprefix("sm_")
    if output_format == "fatbin":
        assert command[command.index("-gencode") + 1] == f"arch={virtual_arch},code=[{arch},{virtual_arch}]"
        assert "-arch" not in command
    elif output_format == "ptx":
        assert command[command.index("-arch") + 1] == virtual_arch
        assert command[command.index("-code") + 1] == virtual_arch
    else:
        assert command[command.index("-arch") + 1] == arch


def test_cuda_code_uses_owned_backend_config():
    target = normalize_target("cuda -arch=sm_100f -code=sm_100a,sm_103a")
    assert "code" not in TargetKind.options_from_name("cuda")
    assert nvcc.get_target_arch_and_code(target) == ("100f", ["sm_100a", "sm_103a"])
    native = native_backend_target(target)
    assert "tilelang" not in native.attrs["backend_config"]
    assert resolve_backend_config(target=native)["arch"] == "sm_100f"
    assert nvcc.get_target_arch_and_code(target) == ("100f", ["sm_100a", "sm_103a"])
