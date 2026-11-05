"""CUDA BF16 device types and pre-SM80 software code paths."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.contrib import nvcc


def bf16_program(mode):
    @T.prim_func
    def main(A: T.Tensor((128,), T.bfloat16), B: T.Tensor((128,), T.bfloat16)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((128,), T.bfloat16)
            for i in T.Parallel(128):
                shared[i] = A[i]
            for i in T.Parallel(128):
                if mode == "copy":
                    B[i] = shared[i]
                elif mode == "arithmetic":
                    B[i] = shared[i] + T.bfloat16(0.5)
                else:
                    B[i] = T.Cast("bfloat16", T.Cast("float32", shared[i]) * T.float32(1.5) + T.float32(0.25))

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_75", "sm_80", "sm_100a"])
@pytest.mark.parametrize("mode", ["copy", "arithmetic", "cast"])
def test_bf16_device_types_compile_and_execute(arch, mode, monkeypatch, tmp_path):
    # Preserve NVCC's selected architecture, but request textual PTX so the
    # selected software/hardware code path can also run through the driver
    # JIT on a newer device. This is not a claim of execution on SM75 silicon.
    major, minor = torch.cuda.get_device_capability(0)
    requested = int(arch.removeprefix("sm_").removesuffix("a"))
    available = major * 10 + minor
    if requested > available or (arch.endswith("a") and requested != available):
        pytest.skip(f"{arch} PTX needs a compatible execution GPU, found sm_{available}")
    original_compile = nvcc.compile_cuda
    original_popen = subprocess.Popen
    captured = []
    commands = []

    def recorded_popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and str(command[0]).endswith("nvcc"):
            commands.append([str(arg) for arg in command])
        return original_popen(command, *args, **kwargs)

    def compile_ptx(code, target_format="ptx", arch=None, options=None, **kwargs):
        source = tmp_path / f"kernel-{len(captured)}.cu"
        source.write_text(code)
        binary = original_compile(code, "ptx", arch=arch, options=options, **kwargs)
        ptx = tmp_path / f"kernel-{len(captured)}.ptx"
        ptx.write_bytes(binary)
        captured.append((source, ptx))
        return binary

    monkeypatch.setattr(nvcc, "compile_cuda", compile_ptx)
    monkeypatch.setattr(subprocess, "Popen", recorded_popen)
    kernel = tilelang.compile(bf16_program(mode), target={"kind": "cuda", "arch": arch}, execution_backend="tvm_ffi")
    source = kernel.get_kernel_source()
    assert "bfloat16_t*" in source
    values = (torch.arange(128, device="cuda", dtype=torch.float32) / 8 - 4).to(torch.bfloat16)
    result = torch.empty_like(values)
    kernel(values, result)
    torch.cuda.synchronize()
    if mode == "copy":
        expected = values
    elif mode == "arithmetic":
        expected = (values.float() + 0.5).to(torch.bfloat16)
    else:
        expected = (values.float() * 1.5 + 0.25).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert captured
    assert f".target {arch}" in captured[-1][1].read_text()
    if "CG_RESULT_DIR" in os.environ:
        destination = Path(os.environ["CG_RESULT_DIR"]) / "bf16-artifacts" / f"{arch}-{mode}"
        destination.mkdir(parents=True)
        records = []
        for _index, (cu, ptx) in enumerate(captured):
            for item in (cu, ptx):
                saved = destination / item.name
                saved.write_bytes(item.read_bytes())
                records.append({"path": saved.name, "sha256": hashlib.sha256(saved.read_bytes()).hexdigest()})
        (destination / "result.json").write_text(
            json.dumps(
                {
                    "nvcc_target": arch,
                    "execution": "PTX JIT on the recorded newer GPU",
                    "device": torch.cuda.get_device_name(0),
                    "capability": torch.cuda.get_device_capability(0),
                    "input_rule": "BF16(arange(128)/8-4)",
                    "mode": mode,
                    "oracle": "exact BF16 equality, atol=rtol=0",
                    "passed": True,
                    "artifacts": records,
                    "compile_commands": commands,
                },
                indent=2,
            )
            + "\n"
        )
