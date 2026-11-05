"""Public booleans retain their type while CUDA launch storage is explicit."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import tvm_ffi
from tvm_tirx import base, ir, tirx
from tvm_tirx.ir import prim

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner.param import AutotuneResult, CompileArgs


BACKENDS = ["tvm_ffi", "cython", "nvrtc", "cutedsl"]


def _function():
    @T.prim_func
    def main(
        enabled: T.bool,
        B: T.Tensor((4,), "float32"),
        bias: T.float32,
        A: T.Tensor((4,), "float32"),
        scale: T.int32,
        negated: T.bool,
        C: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx < 4:
                if enabled:
                    B[tx] = A[tx] + bias * T.float32(scale)
                else:
                    B[tx] = A[tx] - bias * T.float32(scale)
                C[tx] = tx + T.int32(enabled) * scale - T.int32(negated)

    return main


def _compile_args(backend):
    return CompileArgs(
        target={"kind": "cutedsl" if backend == "cutedsl" else "cuda", "arch": "sm_100a"},
        target_host="c",
        execution_backend=backend,
        out_idx=[6, 1],
        pass_configs={"tl.disable_warp_specialized": True},
    )


def _verify(kernel):
    # The original host order interleaves outputs, tensors and scalar types.
    # The device signature's pointer/scalar sort must not change the call API.
    assert kernel.out_idx == [6, 1]
    assert kernel.prim_func.params[0].ty == ir.PrimType("bool")
    assert kernel.prim_func.params[5].ty == ir.PrimType("bool")
    for enabled, negated in [(False, False), (False, True), (True, False), (True, True)]:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            source = torch.tensor([2, -5, 100, 1], dtype=torch.float32, device="cuda")
            integer_output, float_output = kernel(enabled, 1.25, source, 4, negated)
        stream.synchronize()
        expected_int = torch.arange(4, dtype=torch.int32, device="cuda") + int(enabled) * 4 - int(negated)
        torch.testing.assert_close(integer_output, expected_int, rtol=0, atol=0)
        torch.testing.assert_close(float_output, source + (5 if enabled else -5), rtol=0, atol=0)
    if kernel.execution_backend == "cython":
        # Preserve the original permissive Cython conversion, including float
        # truthiness. This patch must not unify backend input validation.
        source = torch.arange(4, dtype=torch.float32, device="cuda")
        for value in (2, 0.5, None):
            integer_output, float_output = kernel(value, 1.25, source, 4, False)
            torch.cuda.synchronize()
            torch.testing.assert_close(float_output, source + (5 if bool(value) else -5), rtol=0, atol=0)
            torch.testing.assert_close(
                integer_output,
                torch.arange(4, dtype=torch.int32, device="cuda") + int(bool(value)) * 4,
                rtol=0,
                atol=0,
            )
    if kernel.execution_backend == "cutedsl":
        # The baseline C++ launcher used integer storage, followed by i1
        # truncation in CuTe's Boolean kernel parameter. Keep that policy
        # separate from NVRTC/Cython truthiness and native packed checking.
        source = torch.arange(4, dtype=torch.float32, device="cuda")
        for value in (0, 1, 2, 3, -1, -2, 255, 256, 2**31, 2**32 + 1, 2**63 - 1):
            flag = bool(value & 1)
            integer_output, float_output = kernel(value, 1.25, source, 4, False)
            torch.cuda.synchronize()
            torch.testing.assert_close(float_output, source + (5 if flag else -5), rtol=0, atol=0)
            torch.testing.assert_close(
                integer_output,
                torch.arange(4, dtype=torch.int32, device="cuda") + int(flag) * 4,
                rtol=0,
                atol=0,
            )
        for value in (0.5, "true", None):
            with pytest.raises(TypeError, match="requires an integer"):
                kernel(value, 1.25, source, 4, False)
        for value in (1 << 63, -(1 << 63) - 1):
            with pytest.raises(OverflowError, match="int64 carrier"):
                kernel(value, 1.25, source, 4, False)
    if kernel.execution_backend == "tvm_ffi":
        for value in (0.5, "true", None):
            source = torch.ones(4, dtype=torch.float32, device="cuda")
            with pytest.raises(RuntimeError, match="expected boolean"):
                kernel(value, 1.25, source, 4, False)
    torch.testing.assert_close(torch.arange(4, device="cuda") + 1, torch.arange(1, 5, device="cuda"), rtol=0, atol=0)


def _check_ir(kernel):
    device = kernel.artifact.device_mod
    assert tirx.analysis.check_well_formed(device)
    for function in device.functions.values():
        if not isinstance(function, tirx.Function):
            continue
        assert all(parameter.ty != ir.PrimType("bool") for parameter in function.params)
        abi_parameters = [p for p in function.params if str(p.name).startswith(("enabled_abi", "negated_abi"))]
        assert len(abi_parameters) == 2
        assert all(p.ty == ir.PrimType("int32") for p in abi_parameters)
        casts = []
        tvm_ffi.structural_walk(function.body, (prim.Cast, casts.append))
        assert any(node.ty == ir.PrimType("bool") for node in casts)


def _cache_process(mode, folder, backend):
    from tilelang.jit.kernel import JITKernel

    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    args = _compile_args(backend)
    if mode == "save":
        kernel = args.compile_program(_function())
        _verify(kernel)
        _check_ir(kernel)
        (folder / "device.txt").write_text(kernel.get_kernel_source())
        (folder / "device.json").write_text(ir.save_json(kernel.artifact.device_mod))
        (folder / "host.json").write_text(ir.save_json(kernel.artifact.host_mod))
        AutotuneResult(latency=1.0, config={"bool_abi": True}, func=kernel.prim_func, kernel=kernel).save_to_disk(folder / "cache")
    else:

        def unexpected_compile(*_args, **_kwargs):
            raise AssertionError("CUDA bool cache must reload existing code")

        JITKernel._compile_and_create_adapter = unexpected_compile
        result = AutotuneResult.load_from_disk(folder / "cache", args)
        assert result is not None and result.config == {"bool_abi": True}
        _verify(result.kernel)
    compiler = Path(base._LOADED_LIBS["tvm_tirx_compiler"]._name).resolve()
    digest = hashlib.sha256(compiler.read_bytes()).hexdigest()
    if expected := os.environ.get("EXPECTED_COMPILER_SHA256"):
        assert digest == expected, str(compiler)
    record = {"mode": mode, "backend": backend, "compiler": str(compiler), "compiler_sha256": digest, "passed": True}
    if Path("/proc/self/maps").exists():
        record["libraries"] = sorted(
            {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libtvm" in Path(line.split()[-1]).name}
        )
    (folder / (mode + ".json")).write_text(json.dumps(record, indent=2) + "\n")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_cuda_boolean_control_arithmetic_output_order_and_reload(backend, tmp_path):
    for mode in ("save", "reload"):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path), backend],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=180,
        )
        (tmp_path / (mode + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((tmp_path / (mode + ".json")).read_text())["passed"]


if __name__ == "__main__":
    _cache_process(*sys.argv[1:])
