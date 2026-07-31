from pathlib import Path

from tilelang.jit.adapter.nvrtc.include_paths import discover_cuda_include_paths


def _make_include_tree(cuda_home: Path, relative_paths: list[str]) -> list[str]:
    paths = []
    for relative_path in relative_paths:
        path = cuda_home / relative_path
        path.mkdir(parents=True)
        paths.append(str(path))
    return paths


def test_discovers_flat_pip_include_layout(tmp_path):
    expected = _make_include_tree(tmp_path, ["include", "include/cccl"])

    assert discover_cuda_include_paths(str(tmp_path), machine="x86_64") == expected


def test_discovers_target_specific_system_include_layout(tmp_path):
    expected = _make_include_tree(
        tmp_path,
        ["targets/x86_64-linux/include", "targets/x86_64-linux/include/cccl"],
    )

    assert discover_cuda_include_paths(str(tmp_path), machine="x86_64") == expected


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

    assert discover_cuda_include_paths(str(tmp_path), machine="aarch64") == expected


def test_preserves_legacy_paths_when_cuda_layout_is_missing(tmp_path):
    assert discover_cuda_include_paths(str(tmp_path), machine="x86_64") == [
        str(tmp_path / "include"),
        str(tmp_path / "targets/x86_64-linux/include"),
        str(tmp_path / "targets/x86_64-linux/include/cccl"),
    ]


def test_preserves_flat_windows_include_layout(tmp_path):
    assert discover_cuda_include_paths(str(tmp_path), system="win32") == [
        str(tmp_path / "include"),
        str(tmp_path / "include/cccl"),
    ]
