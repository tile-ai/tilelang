"""Raw CUDA entry points retain bool8 while runtime holders are int32."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tvm_tirx import base, ir

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner.param import AutotuneResult

from test_native_cuda_boolean_abi import _compile_args, _verify


RAW_SOURCE = r"""
extern "C" __global__ void main_kernel(const float* A, float* B, int* C,
                                        float bias, bool enabled,
                                        bool negated, int scale) {
  int i = threadIdx.x;
  if (i < 4) {
    B[i] = A[i] + (enabled ? bias * scale : -bias * scale);
    C[i] = i + (enabled ? scale : 0) - (negated ? 1 : 0);
  }
}
"""


def _raw_function():
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
            T.sblock_attr({"code_block_source": RAW_SOURCE, "code_block_entry_name": "main_kernel"})
            # The raw-source signature metadata captures the three tensors.
            # Add only the four scalar operands here: adding their data views
            # would create distinct native pointer captures. This placeholder
            # is replaced by RAW_SOURCE at the owning CUDA emitter boundary.
            T.evaluate(T.call_extern("int32", "source_capture", bias, enabled, negated, scale))

    return main


def _raw_cache_process(mode, folder, backend):
    from tilelang.jit.kernel import JITKernel

    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    args = _compile_args(backend)
    if mode == "save":
        kernel = args.compile_program(_raw_function())
        # Check the final argument layout before launching unparsed raw text.
        functions = list(kernel.artifact.device_mod.functions.values())
        assert len(functions) == 1
        device = functions[0]
        (folder / "device-before-launch.json").write_text(ir.save_json(kernel.artifact.device_mod))
        (folder / "device-before-launch.txt").write_text(str(kernel.artifact.device_mod))
        signature = [(str(p.name), str(p.ty)) for p in device.params]
        assert len(device.params) == 7, signature
        assert all(isinstance(p.ty, ir.PtrType) for p in device.params[:3])
        assert [str(p.ty) for p in device.params[3:]] == ["float32", "bool", "bool", "int32"]
        assert RAW_SOURCE in kernel.get_kernel_source()
        _verify(kernel)
        (folder / "device.txt").write_text(kernel.get_kernel_source())
        (folder / "device.json").write_text(ir.save_json(kernel.artifact.device_mod))
        (folder / "host.json").write_text(ir.save_json(kernel.artifact.host_mod))
        AutotuneResult(latency=1.0, config={"raw_bool_abi": True}, func=kernel.prim_func, kernel=kernel).save_to_disk(folder / "cache")
    else:

        def unexpected_compile(*_args, **_kwargs):
            raise AssertionError("Raw CUDA bool cache must reload existing code")

        JITKernel._compile_and_create_adapter = unexpected_compile
        result = AutotuneResult.load_from_disk(folder / "cache", args)
        assert result is not None and result.config == {"raw_bool_abi": True}
        _verify(result.kernel)
    compiler = Path(base._LOADED_LIBS["tvm_tirx_compiler"]._name).resolve()
    digest = hashlib.sha256(compiler.read_bytes()).hexdigest()
    if expected := os.environ.get("EXPECTED_COMPILER_SHA256"):
        assert digest == expected
    (folder / (mode + ".json")).write_text(
        json.dumps(
            {
                "mode": mode,
                "backend": backend,
                "compiler_sha256": digest,
                "raw_signature": "float*,float*,int*,float32,bool8,bool8,int32",
                "passed": True,
            },
            indent=2,
        )
        + "\n"
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", ["tvm_ffi", "cython", "nvrtc"])
def test_raw_cuda_bool8_mixed_order_and_cache(backend, tmp_path):
    for mode in ("save", "reload"):
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path), backend],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=180,
        )
        (tmp_path / (mode + ".log")).write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads((tmp_path / (mode + ".json")).read_text())["passed"]


if __name__ == "__main__":
    _raw_cache_process(*sys.argv[1:])
