import io
import subprocess
from types import SimpleNamespace

import pytest

from tvm.target import Target

from tilelang.contrib import bisheng
from tilelang.jit.adapter import libgen
from tilelang.jit import kernel as jit_kernel
from tilelang.transform import PassConfigKey


def test_target_npu_arch_priority(monkeypatch):
    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-env")

    target = Target({"kind": "ascend", "arch": "dav-arch", "mcpu": "dav-mcpu"})
    assert bisheng.get_target_npu_arch(target) == "dav-arch"

    target = Target({"kind": "ascend", "mcpu": "dav-mcpu"})
    assert bisheng.get_target_npu_arch(target) == "dav-mcpu"
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-env"

    monkeypatch.delenv("ASCEND_NPU_ARCH")
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-3510"


@pytest.mark.parametrize(
    "configured_flags",
    [None, "-mllvm -cce-aicore-dcpreload-args=false -include configured_header.h"],
    ids=["no-configured-flags", "configured-flags"],
)
@pytest.mark.parametrize(
    "compile_flags",
    [
        None,
        ["-mllvm", "-cce-aicore-dcpreload-args=true", "-mllvm", "-inline-threshold=128", "-include", "explicit_header.h"],
    ],
    ids=["no-explicit-flags", "explicit-flags"],
)
@pytest.mark.parametrize("via_jit", [False, True], ids=["direct", "jit"])
def test_cython_compile_preserves_target_and_options(monkeypatch, tmp_path, configured_flags, compile_flags, via_jit):
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

    target = Target({"kind": "ascend", "arch": "dav-target"})
    pass_configs = {PassConfigKey.TL_DEVICE_COMPILE_FLAGS: configured_flags}
    source = 'extern "C" void kernel() {}'

    def compile_library(**kwargs):
        generator = libgen.LibraryGenerator(kwargs["target"])
        generator.assign_pass_configs(kwargs["pass_configs"])
        generator.assign_compile_flags(kwargs["compile_flags"])
        generator.update_lib_code(kwargs["device_kernel_source"])
        generator.compile_lib()
        return generator

    if via_jit:
        kernel = object.__new__(jit_kernel.JITKernel)
        kernel.target = target
        kernel.target_host = None
        kernel.execution_backend = "cython"
        kernel.verbose = False
        kernel.pass_configs = pass_configs
        kernel.compile_flags = compile_flags
        artifact = SimpleNamespace(params=[], host_mod=None, device_mod=None, kernel_source=source)
        monkeypatch.setattr(kernel, "_compile_artifact", lambda *args: artifact)
        monkeypatch.setattr(jit_kernel, "CythonKernelAdapter", compile_library)
        kernel._compile_and_create_adapter(SimpleNamespace(attrs={"global_symbol": "kernel"}), [])
    else:
        compile_library(target=target, pass_configs=pass_configs, compile_flags=compile_flags, device_kernel_source=source)

    assert [option for option in captured_command if option.startswith("--npu-arch=")] == ["--npu-arch=dav-target"]

    expected_llvm_options = ["-cce-aicore-dcpreload-args=false"]
    if configured_flags is not None:
        expected_llvm_options.append("-cce-aicore-dcpreload-args=false")
    if compile_flags is not None:
        expected_llvm_options.extend(["-cce-aicore-dcpreload-args=true", "-inline-threshold=128"])
    llvm_pairs = [captured_command[index : index + 2] for index, option in enumerate(captured_command) if option == "-mllvm"]
    assert llvm_pairs == [["-mllvm", option] for option in expected_llvm_options]

    expected_headers = []
    if configured_flags is not None:
        expected_headers.append("configured_header.h")
    if compile_flags is not None:
        expected_headers.append("explicit_header.h")
    include_pairs = [captured_command[index : index + 2] for index, option in enumerate(captured_command) if option == "-include"]
    assert include_pairs == [["-include", header] for header in expected_headers]
