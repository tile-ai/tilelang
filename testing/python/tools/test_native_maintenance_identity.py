"""Maintenance subprocesses and result caches retain the selected native stack."""

import asyncio
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

from tilelang.autodd import SubProcRunner


@pytest.fixture
def regression_tool(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "maint" / "scripts"))
    return importlib.import_module("run_current_regression")


def test_subprocess_uses_current_python_without_path_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    out, err, success = asyncio.run(SubProcRunner().run("import sys; print(sys.executable)"))
    assert success, err
    assert out.strip() == sys.executable


def test_subprocess_failure_and_timeout_keep_their_oracle():
    _, err, success = asyncio.run(SubProcRunner().run("raise ValueError('native-runner-negative')"))
    assert not success and "ValueError: native-runner-negative" in err
    _, err, success = asyncio.run(SubProcRunner().run("while True: pass", timeout=0.1))
    assert not success and err.startswith("TimeoutError:")


def test_performance_identity_reads_loaded_native_compiler(regression_tool):
    import tvm_ffi

    from tilelang._provider import FFI_REF

    meta = regression_tool._environment_metadata()
    assert "tilelang_error" not in meta, meta.get("tilelang_error")
    compiler = meta["compiler"]
    assert compiler["kind"] == "native-tirx"
    assert compiler["build"] == dict(tvm_ffi.get_global_func("tl.CompilerBuildInfo")())
    assert compiler["build"]["ffi_ref"] == FFI_REF
    assert "tvm_tirx_compiler:" in compiler["libraries"]
    assert "tvm_ffi:" in compiler["libraries"]
    # A frontend version can stay unchanged across provider builds. They must
    # not share a cached benchmark result.
    other = {**meta, "compiler": {**compiler, "libraries": compiler["libraries"] + ":different-build"}}
    assert regression_tool._environment_key(meta) != regression_tool._environment_key(other)


def test_dirty_fingerprint_tracks_content_not_only_git_status(tmp_path, regression_tool):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True)

    git("init", "-q")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("x = 0\n")
    git("add", "tracked.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    tracked.write_text("x = 1\n")
    first = regression_tool._git_state(tmp_path)["commit_key"]
    status = git("status", "--porcelain=v1")
    tracked.write_text("x = 2\n")
    second = regression_tool._git_state(tmp_path)["commit_key"]
    assert git("status", "--porcelain=v1") == status
    assert first != second

    untracked = tmp_path / "new.py"
    untracked.write_text("x = 3\n")
    third = regression_tool._git_state(tmp_path)["commit_key"]
    status = git("status", "--porcelain=v1")
    untracked.write_text("x = 4\n")
    assert git("status", "--porcelain=v1") == status
    assert third != regression_tool._git_state(tmp_path)["commit_key"]


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "commit.gpgsign=false", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *map(str, args)],
        cwd=repo,
        text=True,
    )


def _initialize_repo(path, *, commit=True):
    path.mkdir()
    _git(path, "init", "-q")
    (path / "tracked.py").write_text("x = 0\n")
    (path / ".gitignore").write_text("ignored.txt\n")
    _git(path, "add", ".")
    if commit:
        _git(path, "commit", "-qm", "fixture")
    return path


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("filename", ["tracked.py", "untracked.py"])
def test_dependency_content_changes_key_even_with_ignored_gitlinks(tmp_path, regression_tool, depth, filename):
    parent = _initialize_repo(tmp_path / "parent")
    upstream = _initialize_repo(tmp_path / "upstream")
    if depth == 2:
        leaf = _initialize_repo(tmp_path / "leaf")
        _git(upstream, "-c", "protocol.file.allow=always", "submodule", "add", "-q", leaf, "ffi")
        _git(upstream, "commit", "-qam", "nested dependency")
    _git(parent, "-c", "protocol.file.allow=always", "submodule", "add", "-q", upstream, "compiler")
    _git(parent, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
    _git(parent, "commit", "-qam", "dependency")
    clean = regression_tool._git_state(parent)
    assert not clean["dirty"] and clean["commit_key"] == clean["commit"]
    _git(parent, "config", "submodule.compiler.ignore", "all")
    child = parent / "compiler"
    if depth == 2:
        _git(child, "config", "submodule.ffi.ignore", "all")
        child /= "ffi"
    keys = []
    for value in (1, 2):
        (child / filename).write_text(f"x = {value}\n")
        assert not _git(parent, "status", "--porcelain=v1").strip()
        assert not _git(parent, "diff", "--binary", "HEAD").strip()
        state = regression_tool._git_state(parent)
        assert state["dirty"]
        keys.append(state["commit_key"])
        assert regression_tool._git_state(parent)["commit_key"] == keys[-1]
    assert len(set(keys + [clean["commit_key"]])) == 3
    (child / "ignored.txt").write_text("build scratch must not affect source identity")
    assert regression_tool._git_state(parent)["commit_key"] == keys[-1]


@pytest.mark.parametrize("commit", [True, False])
def test_untracked_nested_repository_is_source_not_a_file(tmp_path, regression_tool, commit):
    parent = _initialize_repo(tmp_path / "parent")
    nested = _initialize_repo(parent / "nested repository", commit=commit)
    first = regression_tool._git_state(parent)["commit_key"]
    (nested / "tracked.py").write_text("x = 1\n")
    second = regression_tool._git_state(parent)["commit_key"]
    assert first != second
    (nested / ".git" / "description").write_text("private repository metadata is not compiler source")
    (nested / "ignored.txt").write_text("ignored build output")
    assert regression_tool._git_state(parent)["commit_key"] == second


def test_untracked_symlink_and_newline_path_use_literal_contents(tmp_path, regression_tool):
    if sys.platform == "win32":
        pytest.skip("This filename and symlink fixture requires POSIX filesystem semantics")
    parent = _initialize_repo(tmp_path / "parent")
    link = parent / "new\nlink"
    link.symlink_to("missing-a")
    first = regression_tool._git_state(parent)["commit_key"]
    link.unlink()
    link.symlink_to("missing-b")
    assert regression_tool._git_state(parent)["commit_key"] != first
