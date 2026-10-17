from pathlib import Path
import pytest

from tilelang.contrib import nvcc
from tilelang.contrib.nvcc import discover_cuda_include_paths
from tilelang.contrib.nvcc import get_nvrtc_include_options


def _make_include_tree(cuda_home: Path, relative_paths: list[str]) -> list[str]:
    paths = []
    for relative_path in relative_paths:
        path = cuda_home / relative_path
        path.mkdir(parents=True)
        paths.append(str(path))
    return paths


def test_discovers_flat_pip_include_layout(tmp_path, monkeypatch):
    expected = _make_include_tree(tmp_path, ["include", "include/cccl"])

    assert discover_cuda_include_paths(str(tmp_path), machine="x86_64", system="linux") == expected
    monkeypatch.setattr(nvcc, "CUDA_HOME", str(tmp_path))
    for major in (12, 13):
        options = [f"-I{path}" for path in expected] + [f"-D__CUDACC_VER_MAJOR__={major}"]
        if major < 13:
            options.append(f"-I{tmp_path / 'include' / 'cuda' / 'std'}")
        assert get_nvrtc_include_options(major) == options
    monkeypatch.setattr(nvcc, "CUDA_HOME", "")
    with pytest.raises(RuntimeError, match="CUDA_HOME"):
        get_nvrtc_include_options(13)


def test_discovers_target_specific_system_include_layout(tmp_path):
    expected = _make_include_tree(
        tmp_path,
        ["targets/x86_64-linux/include", "targets/x86_64-linux/include/cccl"],
    )

    assert discover_cuda_include_paths(str(tmp_path), machine="x86_64", system="linux") == expected


def test_discovers_flat_and_target_specific_include_layouts(tmp_path):
    expected = _make_include_tree(
        tmp_path,
        [
            "include",
            "include/cccl",
            "targets/sbsa-linux/include",
            "targets/sbsa-linux/include/cccl",
        ],
    )

    assert discover_cuda_include_paths(str(tmp_path), machine="aarch64", system="linux") == expected


def test_preserves_legacy_paths_when_cuda_layout_is_missing(tmp_path):
    assert discover_cuda_include_paths(str(tmp_path), machine="x86_64", system="linux") == [
        str(tmp_path / "include"),
        str(tmp_path / "targets/x86_64-linux/include"),
        str(tmp_path / "targets/x86_64-linux/include/cccl"),
    ]


def test_preserves_flat_windows_include_layout(tmp_path):
    assert discover_cuda_include_paths(str(tmp_path), system="win32") == [
        str(tmp_path / "include"),
        str(tmp_path / "include/cccl"),
    ]
