import io
import subprocess

from tvm.target import Target

from tilelang.contrib import bisheng
from tilelang.jit.adapter import libgen


def test_target_npu_arch_priority(monkeypatch):
    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-env")

    target = Target({"kind": "ascend", "arch": "dav-arch", "mcpu": "dav-mcpu"})
    assert bisheng.get_target_npu_arch(target) == "dav-arch"

    target = Target({"kind": "ascend", "mcpu": "dav-mcpu"})
    assert bisheng.get_target_npu_arch(target) == "dav-mcpu"
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-env"

    monkeypatch.delenv("ASCEND_NPU_ARCH")
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-3510"


def test_cython_compile_uses_target_npu_arch(monkeypatch, tmp_path):
    source_path = tmp_path / "kernel.asc"
    captured_command = []

    class SourceFile(io.StringIO):
        name = str(source_path)

    source_file = SourceFile()

    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-env")
    monkeypatch.setattr(bisheng, "find_bisheng_path", lambda: "bisheng")
    monkeypatch.setattr(
        libgen.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: source_file,
    )

    def fake_run(command, **kwargs):
        captured_command.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(libgen.subprocess, "run", fake_run)

    generator = libgen.LibraryGenerator(Target({"kind": "ascend", "arch": "dav-target"}))
    generator.update_lib_code('extern "C" void kernel() {}')
    generator.compile_lib()

    assert [option for option in captured_command if option.startswith("--npu-arch=")] == ["--npu-arch=dav-target"]
