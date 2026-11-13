# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""Read Ascend AI Core exception dumps with CANN's msaicerr tool."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}


def _numpy_dtype(name):
    if name == "bfloat16":
        from ml_dtypes import bfloat16

        return np.dtype(bfloat16)
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported raw exception dump dtype: {name!r}")
    return np.dtype(_DTYPE_MAP[name])


def read_msaicerr_bin(file_path, dtype=np.float32, shape=None, header_size=0):
    """Read a raw per-tensor ``.bin`` file, optionally skipping a byte header.

    Return an independent NumPy array. Without *shape*, the result is flat.
    Packed sub-byte types require explicit unpacking and are not supported.
    """
    if header_size < 0:
        raise ValueError("header_size must be nonnegative")
    with open(file_path, "rb") as file:
        file.seek(header_size)
        data = np.frombuffer(file.read(), dtype=dtype)
    if shape is not None:
        data = data.reshape(shape)
    return data.copy()


def _find_msaicerr():
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        return None
    path = Path(ascend_home) / "tools" / "msaicerr" / "msaicerr.py"
    return str(path) if path.is_file() else None


def parse_exception_dump(dump_path, kernel_name="main_kernel", output_dir=None, wait_seconds=2.0):
    """Find the latest dump for *kernel_name* and recover its tensor arrays.

    Requires CANN >= 9.2 and ``ASCEND_HOME_PATH`` pointing to its installation.
    Set ``ASCEND_DUMP_PATH`` and ``ASCEND_DUMP_SCENE=aic_err_brief_dump`` before
    ACL initialization when running the kernel. Call this function after
    catching the execution/synchronization exception.

    Dumps are searched under ``<dump_path>/extra-info/data-dump/<device>``.
    The newest numeric timestamp is selected across all devices. The callback
    is asynchronous; *wait_seconds* provides time for it to finish writing.

    Each invocation creates a fresh subdirectory in *output_dir* (default:
    ``<dump_path>/parsed``). Parsed files remain there for inspection; existing
    files are never removed. Both ``.bin`` and CANN 9.2's ``.npy`` are supported.
    Raw files use the dtype in their filename, including bfloat16 through
    ml-dtypes; unknown raw types raise ValueError. NumPy files retain their
    stored dtype and shape, and are loaded without pickle support.

    Returns a list of dictionaries with ``data`` (NumPy array), ``type``
    (input/output/workspace), ``index``, ``dtype`` (filename dtype), and ``file``.
    Results are sorted by tensor type and numeric index. All valid buffer
    arguments, including outputs, are currently labelled as inputs by the
    exception callback; indices refer to that compacted list of buffers.
    """
    time.sleep(wait_seconds)
    dump_root = Path(dump_path) / "extra-info" / "data-dump"
    prefix = f"{kernel_name}.custom."
    candidates = []
    for device_dir in dump_root.glob("*"):
        if not device_dir.is_dir():
            continue
        for path in device_dir.iterdir():
            if not path.is_file() or not path.name.startswith(prefix):
                continue
            timestamp = path.name[len(prefix) :]
            if timestamp.isascii() and timestamp.isdecimal():
                # Compare timestamps numerically across devices; path sorting
                # would let the device directory choose an older dump.
                candidates.append((int(timestamp), str(path)))
    if not candidates:
        raise FileNotFoundError(f"No exception dump file found for kernel {kernel_name!r} under {dump_root}")
    dump_file = Path(max(candidates)[1]).resolve()

    msaicerr_path = _find_msaicerr()
    if msaicerr_path is None:
        raise RuntimeError("msaicerr.py not found. Set ASCEND_HOME_PATH to your CANN install path.")

    output_root = Path(output_dir) if output_dir is not None else Path(dump_path) / "parsed"
    output_root.mkdir(parents=True, exist_ok=True)
    parsed_dir = Path(tempfile.mkdtemp(prefix="exception-dump-", dir=output_root)).resolve()
    # msaicerr may write logs relative to its cwd. Resolve its arguments first
    # and isolate each run so parser side effects stay out of the caller's tree.
    try:
        result = subprocess.run(
            [sys.executable, str(Path(msaicerr_path).resolve()), "-d", str(dump_file), "-out", str(parsed_dir)],
            cwd=parsed_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("msaicerr.py timed out after 60 seconds") from exc
    if result.returncode != 0:
        raise RuntimeError(f"msaicerr.py failed (exit {result.returncode}):\n{result.stderr or result.stdout}")

    # Match only outputs of the selected dump; regex metacharacters in a kernel
    # name are literal. Other files/logs in the parsed directory are not tensors.
    tensor_name = re.compile(re.escape(dump_file.name) + r"\.(input|output|workspace)\.(\d+)\.([^.]+)\.(bin|npy)$")
    tensors = []
    for path in parsed_dir.rglob("*"):
        match = tensor_name.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        tensor_type, index, dtype_name, extension = match.groups()
        if extension == "npy":
            data = np.load(path, allow_pickle=False)
            # NumPy serializes ml-dtypes bfloat16 arrays as opaque two-byte
            # values. Recover this dtype without converting the stored bits.
            if dtype_name == "bfloat16" and data.dtype == np.dtype("V2"):
                data = data.view(_numpy_dtype(dtype_name))
        else:
            data = read_msaicerr_bin(path, dtype=_numpy_dtype(dtype_name))
        tensors.append({"data": data, "type": tensor_type, "index": int(index), "dtype": dtype_name, "file": str(path)})
    if not tensors:
        raise RuntimeError(f"msaicerr.py produced no tensor files for {dump_file.name!r} in {parsed_dir}")
    tensors.sort(key=lambda tensor: (tensor["type"], tensor["index"], tensor["file"]))
    return tensors
