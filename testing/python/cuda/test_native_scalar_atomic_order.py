"""CUDA scalar atomics retain ordering when vector alignment is not proven."""

import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH


_DTYPES = {"float32": "float", "float16": "half_t", "bfloat16": "bfloat16_t"}
_ORDERS = ("relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst")
_CASES = tuple(itertools.product(_DTYPES, (False, True), _ORDERS))
_PUBLICATION_ORDERS = (
    ("release", "acquire"),
    ("release", "consume"),
    ("acq_rel", "acq_rel"),
    ("seq_cst", "seq_cst"),
)
_PUBLICATION_CASES = tuple(itertools.product(_DTYPES, _PUBLICATION_ORDERS))


def _name(dtype, previous, order):
    return f"scalar_{dtype}_{int(previous)}_{order}"


def _source():
    source = ["#include <tl_templates/cuda/atomic.h>"]
    for dtype, previous, order in _CASES:
        ty = _DTYPES[dtype]
        body = (
            f"(({ty}*)output)[index] = AtomicAddRet(({ty}*)address, 1.0f, cuda::memory_order_{order});"
            if previous
            else f"AtomicAdd(({ty}*)address, 1.0f, cuda::memory_order_{order});"
        )
        source.append(f"""extern "C" __global__ void {_name(dtype, previous, order)}(void* address, void* output) {{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  {body}
}}""")
    source.append('extern "C" int launch_scalar(int which, void* address, void* output, void* stream) { switch (which) {')
    for index, case in enumerate(_CASES):
        source.append(f"case {index}: {_name(*case)}<<<2, 32, 0, (cudaStream_t)stream>>>(address, output); break;")
    source.append("default: return -1; } return (int)cudaGetLastError(); }")
    for dtype, (write_order, read_order) in _PUBLICATION_CASES:
        ty = _DTYPES[dtype]
        name = f"publish_{dtype}_{write_order}_{read_order}"
        source.append(f"""extern "C" __global__ void {name}(void* flags, int* payload, int* observed, int epoch) {{
  int lane = threadIdx.x;
  auto* flag = (({ty}*)flags) + lane;
  if (blockIdx.x == 0) {{
    for (int j = 0; j < 8; ++j) payload[lane * 8 + j] = epoch + lane * 8 + j;
    AtomicAdd(flag, 1.0f, cuda::memory_order_{write_order});
  }} else {{
    unsigned long long start = clock64();
    while (float(AtomicAddRet(flag, 0.0f, cuda::memory_order_{read_order})) < 1.0f) {{
      if (clock64() - start > 100000000ULL) {{ observed[lane * 8] = -777; return; }}
    }}
    for (int j = 0; j < 8; ++j) observed[lane * 8 + j] = payload[lane * 8 + j];
  }}
}}""")
    source.append(
        'extern "C" int launch_publication(int which, void* flags, int* payload, int* observed, int epoch, void* stream) { switch (which) {'
    )
    for index, (dtype, (write_order, read_order)) in enumerate(_PUBLICATION_CASES):
        source.append(
            f"case {index}: publish_{dtype}_{write_order}_{read_order}<<<2, 32, 0, (cudaStream_t)stream>>>(flags, payload, observed, epoch); break;"
        )
    source.append("default: return -1; } return (int)cudaGetLastError(); }")
    return "\n".join(source)


@pytest.fixture(scope="module", params=("75", "80", "90", "100"))
def scalar_module(request, tmp_path_factory):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA scalar atomic validation needs nvcc")
    root = tmp_path_factory.mktemp("scalar_atomic_" + request.param)
    include = Path(os.environ.get("TILELANG_TEST_ATOMIC_INCLUDE", TILELANG_TEMPLATE_PATH)).resolve()
    header = include / "tl_templates/cuda/atomic.h"
    header_hash = hashlib.sha256(header.read_bytes()).hexdigest()
    cu = root / "scalar.cu"
    cu.write_text(_source())
    command = [nvcc, "-std=c++17", "-O3", "-I" + str(include), "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR]
    commands = [
        command + ["--ptx", "-arch=sm_" + request.param, str(cu), "-o", str(root / "scalar.ptx")],
        command
        + [
            "--shared",
            "-Xcompiler=-fPIC",
            "-gencode=arch=compute_" + request.param + ",code=compute_" + request.param,
            str(cu),
            "-o",
            str(root / "scalar.so"),
        ],
    ]
    records = []
    for index, cmd in enumerate(commands):
        result = subprocess.run(cmd, capture_output=True, text=True)
        (root / f"nvcc-{index}.log").write_text(result.stdout + result.stderr)
        records.append({"command": cmd, "exit_code": result.returncode})
        (root / "build.json").write_text(
            json.dumps(
                {
                    "commands": records,
                    "header": str(header),
                    "header_sha256": header_hash,
                    "source_sha256": hashlib.sha256(cu.read_bytes()).hexdigest(),
                    "device_macro_target": "sm_" + request.param,
                    "runtime_scope": "The PTX branch executes on the actual host GPU, not on emulated SM80/90 hardware.",
                },
                indent=2,
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert hashlib.sha256(header.read_bytes()).hexdigest() == header_hash
    ptx = (root / "scalar.ptx").read_text()
    entries = {}
    parts = re.split(r"(?=\.visible \.entry )", ptx)
    for part in parts:
        match = re.match(r"\.visible \.entry (\w+)\(", part)
        if match:
            entries[match.group(1)] = part
    return root, entries


@pytest.mark.parametrize("dtype,previous,order", _CASES)
def test_scalar_order_is_present_in_target_ptx(scalar_module, dtype, previous, order):
    _, entries = scalar_module
    source = entries[_name(dtype, previous, order)]
    lines = source.splitlines()
    atomics = [i for i, line in enumerate(lines) if re.search(r"\b(?:atom|red)\.", line)]
    barriers = [i for i, line in enumerate(lines) if re.search(r"\b(?:fence\.[^;]*gpu|membar\.gl)", line)]
    assert atomics, source
    ordered_atomic = "\n".join(lines[i] for i in atomics)
    if order == "relaxed":
        assert not barriers, source
        assert not any(word in ordered_atomic for word in (".acquire.", ".release.", ".acq_rel."))
    if order in ("release", "acq_rel", "seq_cst"):
        release = ".release." in ordered_atomic or ".acq_rel." in ordered_atomic
        assert release or any(index < min(atomics) for index in barriers), source
    if order in ("consume", "acquire", "acq_rel", "seq_cst"):
        acquire = ".acquire." in ordered_atomic or ".acq_rel." in ordered_atomic
        assert acquire or any(index > max(atomics) for index in barriers), source


@pytest.mark.parametrize("dtype,previous,order", _CASES)
def test_scalar_atomic_contention_previous_value_and_natural_alignment(scalar_module, dtype, previous, order):
    if not torch.cuda.is_available():
        pytest.skip("CUDA execution needs an available device")
    root, _ = scalar_module
    library = ctypes.CDLL(str(root / "scalar.so"))
    launch = library.launch_scalar
    launch.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    launch.restype = ctypes.c_int
    torch_dtype = getattr(torch, dtype)
    for offset in (0, 1, 3):
        storage = torch.full((8,), -16, dtype=torch_dtype, device="cuda")
        storage[offset] = 0
        before = storage.clone()
        previous_values = torch.full((64,), -16, dtype=torch_dtype, device="cuda")
        address = storage.data_ptr() + offset * storage.element_size()
        assert (
            launch(_CASES.index((dtype, previous, order)), address, previous_values.data_ptr(), torch.cuda.current_stream().cuda_stream)
            == 0
        )
        torch.cuda.synchronize()
        before[offset] = 64
        torch.testing.assert_close(storage, before, rtol=0, atol=0)
        expected = torch.arange(64, dtype=torch_dtype, device="cuda") if previous else torch.full_like(previous_values, -16)
        torch.testing.assert_close(previous_values.sort().values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,orders", _PUBLICATION_CASES)
def test_scalar_release_acquire_publishes_payload_between_blocks(scalar_module, dtype, orders):
    """A bounded publication litmus complements the PTX ordering check."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA publication validation needs an available device")
    root, _ = scalar_module
    library = ctypes.CDLL(str(root / "scalar.so"))
    launch = library.launch_publication
    launch.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    launch.restype = ctypes.c_int
    torch_dtype = getattr(torch, dtype)
    for offset in (1, 3):
        for epoch in (17, 1031):
            storage = torch.full((40,), -9, dtype=torch_dtype, device="cuda")
            storage[offset : offset + 32] = 0
            expected_flags = storage.clone()
            expected_flags[offset : offset + 32] = 1
            payload = torch.full((256,), -999, dtype=torch.int32, device="cuda")
            observed = torch.full_like(payload, -888)
            assert (
                launch(
                    _PUBLICATION_CASES.index((dtype, orders)),
                    storage.data_ptr() + offset * storage.element_size(),
                    payload.data_ptr(),
                    observed.data_ptr(),
                    epoch,
                    torch.cuda.current_stream().cuda_stream,
                )
                == 0
            )
            torch.cuda.synchronize()
            expected = torch.arange(256, dtype=torch.int32, device="cuda") + epoch
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            torch.testing.assert_close(storage, expected_flags, rtol=0, atol=0)


def test_scalar_feature_gates_compile_each_fatbin_architecture(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA scalar atomic validation needs nvcc")
    include = Path(os.environ.get("TILELANG_TEST_ATOMIC_INCLUDE", TILELANG_TEMPLATE_PATH)).resolve()
    header = include / "tl_templates/cuda/atomic.h"
    source = tmp_path / "scalar.cu"
    source.write_text(_source())
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-I" + str(include),
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
        "--fatbin",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        str(source),
        "-o",
        str(tmp_path / "scalar.fatbin"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    (tmp_path / "nvcc.log").write_text(result.stdout + result.stderr)
    (tmp_path / "build.json").write_text(
        json.dumps(
            {
                "command": command,
                "exit_code": result.returncode,
                "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": "Fatbin device compilation only; individual PTX branches have separate execution tests.",
            },
            indent=2,
        )
    )
    assert result.returncode == 0, result.stdout + result.stderr
