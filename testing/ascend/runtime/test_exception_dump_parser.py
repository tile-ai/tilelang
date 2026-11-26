# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""Host-only tests: neither a compiled TileLang library nor an NPU is needed."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Load this standalone tool without initializing TileLang's compiler/runtime.
_SOURCE = Path(__file__).resolve().parents[3] / "tilelang" / "ascend" / "exception_dump_bin.py"
_SPEC = importlib.util.spec_from_file_location("ascend_exception_dump_bin", _SOURCE)
dump_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dump_tool)


def _dump(root, name="main_kernel", timestamp="20261106123456000", device="0"):
    path = root / "extra-info" / "data-dump" / device / f"{name}.custom.{timestamp}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"CANN dump placeholder")
    return path


@pytest.fixture
def mock_parser(monkeypatch):
    monkeypatch.setattr(dump_tool, "_find_msaicerr", lambda: "/cann/tools/msaicerr/msaicerr.py")
    monkeypatch.setattr(dump_tool, "subprocess", SimpleNamespace(run=subprocess.run, TimeoutExpired=subprocess.TimeoutExpired))

    def install(writer):
        def run(command, **kwargs):
            assert kwargs["timeout"] == 60
            assert kwargs["capture_output"]
            source = Path(command[command.index("-d") + 1])
            destination = Path(command[command.index("-out") + 1])
            assert source.is_absolute() and destination.is_absolute()
            assert Path(kwargs["cwd"]) == destination
            writer(source, destination)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(dump_tool.subprocess, "run", run)

    return install


@pytest.mark.parametrize("extension", ["bin", "npy"])
def test_parse_recovers_tensors_without_deleting_existing_files(tmp_path, mock_parser, extension):
    source = _dump(tmp_path, name="k.[x].y", timestamp="1")
    output = tmp_path / "parsed"
    output.mkdir()
    stale = output / "unrelated.bin"
    stale.write_bytes(b"keep this file")
    expected = np.array([1.5, -2.0, 3.25], dtype=np.float16)

    def write(selected, destination):
        assert selected == source
        for index in (10, 2):
            path = destination / f"{source.name}.input.{index}.float16.{extension}"
            if extension == "npy":
                np.save(path, expected)
            else:
                expected.tofile(path)
        np.save(destination / "unrelated.npy", expected + 1)

    mock_parser(write)
    first = dump_tool.parse_exception_dump(tmp_path, "k.[x].y", output, wait_seconds=0)
    second = dump_tool.parse_exception_dump(tmp_path, "k.[x].y", output, wait_seconds=0)
    assert [tensor["index"] for tensor in first] == [2, 10]
    assert stale.read_bytes() == b"keep this file"
    assert Path(first[0]["file"]).is_file()
    assert Path(first[0]["file"]).parent != Path(second[0]["file"]).parent
    for tensor in first + second:
        np.testing.assert_array_equal(tensor["data"], expected)
        assert tensor["dtype"] == "float16"
        assert tensor["type"] == "input"


def test_selects_latest_timestamp_across_devices(tmp_path, mock_parser):
    _dump(tmp_path, timestamp="9", device="9")
    latest = _dump(tmp_path, timestamp="10", device="0")
    _dump(tmp_path, name="other_kernel", timestamp="20")
    _dump(tmp_path, timestamp="30.not-a-dump")

    def write(selected, destination):
        assert selected == latest
        np.save(destination / f"{selected.name}.input.0.int32.npy", np.array([17], dtype=np.int32))

    mock_parser(write)
    result = dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)
    np.testing.assert_array_equal(result[0]["data"], [17])


@pytest.mark.parametrize("extension", ["bin", "npy"])
def test_bfloat16_values(tmp_path, mock_parser, extension):
    from ml_dtypes import bfloat16

    source = _dump(tmp_path)
    expected = np.array([1.5, -2.0], dtype=bfloat16)

    def write(_, destination):
        path = destination / f"{source.name}.input.0.bfloat16.{extension}"
        if extension == "npy":
            np.save(path, expected)
        else:
            expected.tofile(path)

    mock_parser(write)
    result = dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)
    assert result[0]["data"].dtype == expected.dtype
    np.testing.assert_array_equal(result[0]["data"].astype(np.float32), expected.astype(np.float32))


def test_unknown_raw_dtype_is_not_silently_interpreted_as_float(tmp_path, mock_parser):
    source = _dump(tmp_path)
    mock_parser(lambda _, output: (output / f"{source.name}.input.0.int4.bin").write_bytes(b"\x12"))
    with pytest.raises(ValueError, match="Unsupported raw exception dump dtype.*int4"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_numpy_object_data_cannot_load_pickle(tmp_path, mock_parser):
    source = _dump(tmp_path)
    mock_parser(lambda _, output: np.save(output / f"{source.name}.input.0.float32.npy", np.array([{}], dtype=object)))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_raw_header_and_shape(tmp_path):
    path = tmp_path / "tensor.bin"
    expected = np.arange(6, dtype=np.int16).reshape(2, 3)
    path.write_bytes(b"header" + expected.tobytes())
    data = dump_tool.read_msaicerr_bin(path, dtype=np.int16, shape=(2, 3), header_size=6)
    np.testing.assert_array_equal(data, expected)
    assert data.flags.writeable
    with pytest.raises(ValueError, match="header_size"):
        dump_tool.read_msaicerr_bin(path, header_size=-1)


def test_missing_dump(tmp_path):
    with pytest.raises(FileNotFoundError, match="main_kernel"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_missing_msaicerr(tmp_path, monkeypatch):
    _dump(tmp_path)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    with pytest.raises(RuntimeError, match="ASCEND_HOME_PATH"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_msaicerr_failure(tmp_path, mock_parser, monkeypatch):
    _dump(tmp_path)
    monkeypatch.setattr(
        dump_tool.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stdout="", stderr="invalid dump"),
    )
    with pytest.raises(RuntimeError, match="invalid dump"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_msaicerr_timeout(tmp_path, mock_parser, monkeypatch):
    _dump(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 60)

    monkeypatch.setattr(dump_tool.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_parser_requires_matching_tensor_output(tmp_path, mock_parser):
    _dump(tmp_path)
    mock_parser(lambda _, output: np.save(output / "unrelated.npy", np.ones(1)))
    with pytest.raises(RuntimeError, match="produced no tensor files"):
        dump_tool.parse_exception_dump(tmp_path, wait_seconds=0)


def test_relative_paths_are_resolved_before_changing_parser_directory(tmp_path, monkeypatch, mock_parser):
    monkeypatch.chdir(tmp_path)
    source = _dump(Path("dumps"))

    def write(selected, destination):
        assert selected == source.resolve()
        np.save(destination / f"{source.name}.input.0.int32.npy", np.array([23], dtype=np.int32))

    mock_parser(write)
    result = dump_tool.parse_exception_dump("dumps", output_dir="parsed", wait_seconds=0)
    np.testing.assert_array_equal(result[0]["data"], [23])
