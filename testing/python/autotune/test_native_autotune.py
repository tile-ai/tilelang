"""Autotuning shares the native argument, grouped compiler and export contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tvm_tirx import ir, runtime, tirx

import tilelang
import tilelang.cpu.language as C
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner.param import AutotuneResult, CompileArgs


LLVM_NOT_BUILT = pytest.mark.skipif(not runtime.enabled("llvm"), reason="LLVM support is not built")
HOST_KINDS = ["c", pytest.param("llvm", marks=LLVM_NOT_BUILT)]


def _target(kind):
    return {"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})}


def _cpu_function(callee):
    count = C.dynamic("count", "int64")

    @C.prim_func
    def main(B: C.Tensor((count,), "float32"), A: C.Buffer((count,), "float32", align=4), C0: C.Tensor((count,), "int32")):
        with C.Kernel(1):
            for i in C.Parallel(count):
                B[i] = A[i] + C.float32(1)
                C0[i] = C.int32(i) * 3

    return main.with_attr("tilelang_callee_allocated_outputs", callee)


def _verify_cpu(kernel):
    for count in (0, 1, 17, 257):
        actual = torch.arange(count + 5, dtype=torch.float32)[5:]
        outputs = kernel(actual)
        assert isinstance(outputs, list) and len(outputs) == 2
        torch.testing.assert_close(outputs[0], torch.arange(count, dtype=torch.int32) * 3, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], actual + 1, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="dtype"):
        kernel(torch.ones(3, dtype=torch.int32))


def _cpu_cache_process(mode, path, kind, callee, record):
    from tilelang.jit.kernel import JITKernel

    args = CompileArgs(
        target=_target(kind),
        target_host=_target(kind),
        execution_backend="tvm_ffi",
        out_idx=[2, 0],
    )
    path = Path(path)
    if mode == "save":
        kernel = args.compile_program(_cpu_function(callee))
        _verify_cpu(kernel)
        AutotuneResult(latency=1.0, config={"callee": callee}, ref_latency=2.0, func=kernel.prim_func, kernel=kernel).save_to_disk(path)
        assert (path / "executable.so").is_file()
    else:

        def unexpected_compile(*_args, **_kwargs):
            raise AssertionError("Autotune cache load must not invoke the compiler")

        JITKernel._compile_and_create_adapter = unexpected_compile
        result = AutotuneResult.load_from_disk(path, args)
        assert result is not None and isinstance(result.func, tirx.Function)
        assert result.config == {"callee": callee}
        assert result.latency == 1.0 and result.ref_latency == 2.0
        _verify_cpu(result.kernel)
    assert "tvm" not in sys.modules
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libtvm" in line})
    assert not any("/3rdparty/tvm/" in item for item in libraries)
    Path(record).write_text(json.dumps({"mode": mode, "kind": kind, "callee": callee, "libraries": libraries, "passed": True}))


@pytest.mark.parametrize("kind", HOST_KINDS)
@pytest.mark.parametrize("callee", [False, True])
def test_autotune_native_cpu_save_and_new_process_reload(kind, callee, tmp_path):
    for mode in ("save", "reload"):
        record = tmp_path / f"{mode}.json"
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path / "cache"), kind, str(int(callee)), str(record)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=180,
        )
        (tmp_path / f"{mode}.log").write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads(record.read_text())["passed"]


def _grouped_function(variant, callee):
    if variant < 0:
        raise ValueError("invalid candidate")
    count = T.dynamic("count", "int64")

    @T.prim_func
    def main(B: T.Tensor((count,), "float32"), A: T.Buffer((count,), "float32", align=4), bias: T.float32, C0: T.Tensor((count,), "int32")):
        with T.Kernel(T.ceildiv(count, 128), threads=128) as bx:
            for i in T.Parallel(128):
                index = bx * 128 + i
                if index < count:
                    B[index] = A[index] + bias + T.float32(variant)
                    C0[index] = T.int32(index) * 3 + variant

    return main.with_attr("tilelang_callee_allocated_outputs", callee)


def _check_zero_grid_rejection(kernel):
    # Baseline issue #1993 defines a zero CUDA grid as a launch error, even
    # when the tensor has no elements. Reject it before launching a CTA.
    empty = torch.empty(0, dtype=torch.float32, device="cuda")
    with pytest.raises((ValueError, RuntimeError), match=r"grid\.x must be a positive uint32 dimension"):
        kernel(empty, 0.25)
    torch.cuda.synchronize()
    healthy = torch.arange(8, device="cuda")
    torch.testing.assert_close(healthy + 1, torch.arange(1, 9, device="cuda"), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("callee", [False, True])
@pytest.mark.parametrize("host_kind", HOST_KINDS)
def test_grouped_compile_preserves_native_outputs_and_candidate_failures(callee, host_kind, monkeypatch):
    from tilelang.autotuner import grouped_compile

    original_codegen = grouped_compile.device_codegen
    invocations = []

    def observe_codegen(module, context):
        assert isinstance(module, ir.IRModule)
        invocations.append(module)
        return original_codegen(module, context)

    monkeypatch.setattr(grouped_compile, "device_codegen", observe_codegen)
    args = CompileArgs(
        target={"kind": "cuda", "arch": "sm_100a"},
        target_host=_target(host_kind),
        execution_backend="tvm_ffi",
        out_idx=[3, 0],
        pass_configs={"tl.disable_warp_specialized": True},
    )
    results = grouped_compile.compile_grouped_unit_tvm_ffi(
        [(0, {"variant": 0}), (1, {"variant": 1}), (2, {"variant": -1})],
        args,
        lambda variant: _grouped_function(variant, callee),
    )
    assert len(invocations) == 1, [(index, repr(error)) for index, _, _, error in results]
    by_index = {index: (config, kernel, error) for index, config, kernel, error in results}
    assert set(by_index) == {0, 1, 2}
    assert by_index[2][1] is None and isinstance(by_index[2][2], ValueError)
    assert "invalid candidate" in str(by_index[2][2])
    for variant in (0, 1):
        _, kernel, error = by_index[variant]
        assert error is None, repr(error)
        assert isinstance(kernel.artifact.host_mod, ir.IRModule)
        assert isinstance(kernel.artifact.device_mod, ir.IRModule)
        assert kernel.artifact.target.kind.name == "cuda"
        _check_zero_grid_rejection(kernel)
        for count in (1, 17, 257):
            values = torch.arange(count + 5, dtype=torch.float32, device="cuda")[5:]
            integer_output, float_output = kernel(values, 0.25)
            torch.testing.assert_close(integer_output, torch.arange(count, dtype=torch.int32, device="cuda") * 3 + variant, rtol=0, atol=0)
            torch.testing.assert_close(float_output, values + 0.25 + variant, rtol=0, atol=0)


def _verify_gpu(kernel):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _check_zero_grid_rejection(kernel)
    stream.synchronize()
    for count in (1, 17, 257):
        with torch.cuda.stream(stream):
            values = torch.arange(count + 5, dtype=torch.float32, device="cuda")[5:]
            outputs = kernel(values, 0.25)
        stream.synchronize()
        assert isinstance(outputs, list) and len(outputs) == 2
        torch.testing.assert_close(outputs[0], torch.arange(count, dtype=torch.int32, device="cuda") * 3 + 2, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], values + 2.25, rtol=0, atol=0)
    # The original Cython wrapper reports ValueError, whereas the packed FFI
    # binder reports RuntimeError. The native Python device adapters add an
    # explicit dtype guard to what was previously an unchecked pointer input.
    error, message = {
        "tvm_ffi": (RuntimeError, "dtype"),
        "cython": (ValueError, "dtype"),
        "nvrtc": (TypeError, "expected torch.float32"),
        "cutedsl": (TypeError, "expected torch.float32"),
    }[kernel.execution_backend]
    with pytest.raises(error, match=message):
        kernel(torch.ones(3, dtype=torch.int32, device="cuda"), 0.25)


def _gpu_cache_process(mode, path, backend, callee, record, host_kind):
    from tilelang.jit.kernel import JITKernel

    path = Path(path)
    args = CompileArgs(
        target={"kind": "cutedsl" if backend == "cutedsl" else "cuda", "arch": "sm_100a"},
        target_host=_target(host_kind),
        execution_backend=backend,
        out_idx=[3, 0],
        pass_configs={"tl.disable_warp_specialized": True},
    )
    if mode == "gpu_save":
        kernel = args.compile_program(_grouped_function(2, callee))
        _verify_gpu(kernel)
        AutotuneResult(latency=1.0, config={"callee": callee, "variant": 2}, func=kernel.prim_func, kernel=kernel).save_to_disk(path)
        assert AutotuneResult._is_complete_result_dir(path, backend)
        if backend == "cutedsl":
            assert list(path.glob("kernel.cutedsl/*/manifest.json"))
    else:

        def unexpected_compile(*_args, **_kwargs):
            raise AssertionError("Autotune cache reload must use its persisted device artifacts")

        JITKernel._compile_and_create_adapter = unexpected_compile
        if backend == "cutedsl":
            from tilelang.jit.adapter.cutedsl.runtime import CuTeDSLRuntime

            CuTeDSLRuntime._compile = unexpected_compile
            assert "cutlass" not in sys.modules
        result = AutotuneResult.load_from_disk(path, args)
        assert result is not None and isinstance(result.func, tirx.Function)
        assert result.config == {"callee": callee, "variant": 2}
        _verify_gpu(result.kernel)
        if backend == "cutedsl":
            assert "cutlass" not in sys.modules
    assert "tvm" not in sys.modules
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libtvm" in line})
    assert not any("/3rdparty/tvm/" in item for item in libraries)
    Path(record).write_text(
        json.dumps({"mode": mode, "backend": backend, "host_kind": host_kind, "callee": callee, "libraries": libraries, "passed": True})
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "backend,host_kind",
    [("tvm_ffi", "c"), pytest.param("tvm_ffi", "llvm", marks=LLVM_NOT_BUILT), ("cython", "c"), ("nvrtc", "c"), ("cutedsl", "c")],
)
@pytest.mark.parametrize("callee", [False, True])
def test_autotune_gpu_save_and_new_process_reload(backend, callee, host_kind, tmp_path):
    environment = dict(os.environ, TILELANG_CUTEDSL_HOST_LAUNCHER="cpp")
    for mode in ("gpu_save", "gpu_reload"):
        record = tmp_path / f"{mode}.json"
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                mode,
                str(tmp_path / "cache"),
                backend,
                str(int(callee)),
                str(record),
                host_kind,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        (tmp_path / f"{mode}.log").write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads(record.read_text())["passed"]


if __name__ == "__main__":
    if sys.argv[1].startswith("gpu_"):
        _gpu_cache_process(sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4])), sys.argv[5], sys.argv[6])
    else:
        _cpu_cache_process(sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4])), sys.argv[5])
