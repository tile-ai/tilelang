import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


def load_tool(name):
    path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_tool("prepare_tirx")
repair = load_tool("repair_tirx_wheel")
installation = load_tool("check_tirx_wheel_install")


@pytest.mark.parametrize("overlap", [False, True])
def test_binary_boundary_rejects_native_implementation_interposition(monkeypatch, overlap):
    monkeypatch.setattr(installation.sys, "platform", "linux")
    monkeypatch.setattr(installation.shutil, "which", lambda name: "/usr/bin/nm")

    def exported(command, text):
        name = "native_impl" if overlap or "libtvm" in command[-1] else "tilelang_impl"
        return f"{name} T 0 10\nheader_inline W 0 10\n__bss_start B 0 0\n_edata D 0 0\n_end B 0 0\n"

    monkeypatch.setattr(installation.subprocess, "check_output", exported)
    libraries = {Path("libtilelang.so"), Path("libtvm_tirx_compiler.so")}
    if overlap:
        with pytest.raises(RuntimeError, match="native_impl"):
            installation.check_symbol_ownership(libraries)
    else:
        installation.check_symbol_ownership(libraries)


def wheel(tmp_path, distribution="apache-tvm-tirx", version="0.1.0a2", files=()):
    path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fixture.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n")
        for name in files:
            archive.writestr(name, b"")
    return path


@pytest.mark.parametrize("revision", ["main", "9af5544", "", "HEAD~1", "x" * 40])
def test_dependency_revision_must_be_immutable(revision):
    with pytest.raises(ValueError, match="full, pinned"):
        prepare.validate_revision(revision)


def test_source_dependency_cannot_return_to_thirdparty(tmp_path):
    source = tmp_path / "3rdparty" / "renamed-tirx"
    with pytest.raises(ValueError, match="3rdparty"):
        prepare.validate_source(source, "a" * 40, tmp_path)


def test_source_symlink_cannot_return_to_thirdparty(tmp_path):
    project = tmp_path / "tilelang"
    thirdparty = project / "3rdparty"
    thirdparty.mkdir(parents=True)
    dependency = tmp_path / "tvm-tirx"
    dependency.mkdir()
    source = thirdparty / "tvm"
    try:
        source.symlink_to(dependency, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this host")
    with pytest.raises(ValueError, match="3rdparty"):
        prepare.validate_source(source, "a" * 40, project=project)


def source_fixture(directory):
    source = directory / "dependency"
    for filename in ("include/tvm/tirx/type.h", "python/tvm_tirx/__init__.py", "3rdparty/tvm-ffi/CMakeLists.txt"):
        path = source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return source


def test_source_without_git_cannot_bypass_the_upstream_pin(tmp_path):
    source = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="Git checkout"):
        prepare.validate_source(source, "a" * 40)


@pytest.mark.parametrize("dirty", [" M src/ir/prim.cc\n", "?? src/extension.cc\n", " M 3rdparty/tvm-ffi\n", ""])
def test_source_pin_also_requires_clean_files_and_submodules(tmp_path, monkeypatch, dirty):
    source = source_fixture(tmp_path)
    (source / ".git").touch()
    revision = "a" * 40

    def run(command, **options):
        if "rev-parse" in command:
            return SimpleNamespace(stdout=revision + "\n")
        assert "--untracked-files=all" in command
        assert "--ignore-submodules=none" in command
        return SimpleNamespace(stdout=dirty)

    monkeypatch.setattr(prepare, "run", run)
    if dirty:
        with pytest.raises(ValueError, match="unmodified"):
            prepare.validate_source(source, revision)
    else:
        assert prepare.validate_source(source, revision) == source


def test_command_line_cannot_disable_required_extension_exports():
    with pytest.raises(ValueError, match="CMAKE_PROJECT_tvm_tirx_INCLUDE"):
        prepare.definitions_from_environment({}, command_line=["CMAKE_PROJECT_tvm_tirx_INCLUDE="])


@pytest.mark.parametrize("setting", ["ON", "OFF"])
def test_dependency_no_longer_requires_a_tirx_export_option(setting):
    definitions = prepare.definitions_from_environment({})
    assert "TVM_TIRX_EXPORT_ALL_SYMBOLS" not in definitions
    assert Path(definitions["CMAKE_PROJECT_tvm_tirx_INCLUDE"]) == prepare.EXTENSION_EXPORTS
    with pytest.raises(ValueError, match="obsolete"):
        prepare.definitions_from_environment({}, command_line=[f"TVM_TIRX_EXPORT_ALL_SYMBOLS={setting}"])


def test_command_line_backend_overrides_are_validated():
    definitions = prepare.definitions_from_environment({"USE_CUDA": "OFF"}, command_line=["TVM_TIRX_USE_CUDA=/cuda/path"])
    assert definitions["TVM_TIRX_USE_CUDA"] == "/cuda/path"


@pytest.mark.parametrize("version", ["0.1.0a1", "0.1.0a2+tilelang.1", "0.1.0a2+different"])
def test_only_upstream_wheel_version_is_accepted(tmp_path, version):
    with pytest.raises(ValueError, match="Expected apache-tvm-tirx"):
        prepare.validate_wheel(wheel(tmp_path, version=version), prepare.required_version())


def test_upstream_wheel_version_matches_dependency_pin(tmp_path):
    prepare.validate_wheel(wheel(tmp_path), prepare.required_version())


@pytest.mark.parametrize("backend", prepare.BACKENDS)
def test_dependency_backend_settings_preserve_sdk_paths(backend):
    definitions = prepare.definitions_from_environment({f"USE_{backend}": "/toolkit/path with spaces"})
    assert definitions[f"TVM_TIRX_USE_{backend}"] == "/toolkit/path with spaces"


def test_explicit_dependency_settings_override_tilelang_defaults():
    definitions = prepare.definitions_from_environment({"USE_LLVM": "OFF", "TVM_TIRX_USE_LLVM": "/usr/bin/llvm-config-18"})
    assert definitions["TVM_TIRX_USE_LLVM"] == "/usr/bin/llvm-config-18"


def test_missing_requested_runtime_is_not_silently_accepted(tmp_path):
    definitions = prepare.definitions_from_environment({"USE_ROCM": "ON", "USE_CUDA": "OFF"})
    with pytest.raises(ValueError, match="requested ROCM runtime"):
        prepare.validate_backends({"library_dir": str(tmp_path)}, definitions)
    (tmp_path / "libtvm_runtime_rocm.so").touch()
    prepare.validate_backends({"library_dir": str(tmp_path)}, definitions)


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_tilelang_repair_excludes_independent_libraries(platform):
    command = repair.repair_command("input.whl", "output", "tilelang", platform=platform)
    for stem in repair.DEPENDENCY_STEMS:
        assert any(stem in argument for argument in command)


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_tirx_repair_retains_its_own_libraries(platform):
    command = repair.repair_command("input.whl", "output", "tirx", platform=platform)
    assert any("tvm_ffi" in argument for argument in command)
    assert not any("tvm_runtime" in argument or "tvm_tirx_compiler" in argument for argument in command)


@pytest.mark.parametrize(
    "library", ["libtvm_tirx_compiler.so", "libtvm_runtime_cuda-1234.so.1", "tvm_runtime.dll", "tvm_ffi.dll", "libtvm_runtime_metal.dylib"]
)
def test_repaired_tilelang_wheel_cannot_bundle_a_dependency(tmp_path, library):
    path = wheel(tmp_path, "tilelang", files=[f"tilelang.libs/{library}"])
    with pytest.raises(ValueError, match="Independent dependency"):
        repair.verify_boundary(path, "tilelang")


def test_repaired_tirx_wheel_owns_the_native_compiler(tmp_path):
    path = wheel(tmp_path, files=["tvm_tirx/lib/libtvm_tirx_compiler.so", "tvm_tirx/lib/libtvm_runtime.so"])
    repair.verify_boundary(path, "tirx")
