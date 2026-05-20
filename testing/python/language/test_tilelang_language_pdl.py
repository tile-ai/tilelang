import importlib.metadata
import os
import platform
import shutil
import statistics
import subprocess
import sys

import tilelang.testing
import tilelang.language as T
import pytest


def kernels_with_pdl_trigger(N, block_size=256, dtype=T.float32):
    """Create a TileLang kernel that triggers dependent PDL launches."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        """Trigger dependent PDL launches after writing the output tensor."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    B[idx] = A[idx] + 1.0
            T.pdl_trigger()

    return main


def kernels_with_pdl_sync(N, block_size=256, dtype=T.float32):
    """Create a TileLang kernel that waits for PDL dependencies."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        """Wait for PDL dependencies before writing the output tensor."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
            T.pdl_sync()
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                if idx < N:
                    B[idx] = A[idx] * 2.0

    return main


def kernels_with_pdl_pipeline(N, block_size=256, dtype=T.float32):
    """Create a two-kernel TileLang pipeline using PDL trigger and sync."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        """Launch a producer kernel and a PDL-dependent consumer kernel."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    B[idx] = A[idx] + 1.0
            T.pdl_trigger()

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
            T.pdl_sync()
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                if idx < N:
                    C[idx] = B[idx] * 2.0

    return main


def kernels_without_pdl_overlap_window(N, block_size=256, work_iters=8192, dtype=T.float32):
    """Create a strict same-stream two-kernel baseline for the PDL overlap benchmark."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        Tail: T.Tensor((N,), dtype),
        Pre: T.Tensor((N,), dtype),
    ):
        """Run producer tail work before the consumer can start."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                B[idx] = A[idx] + 1.0
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                x = T.alloc_var(T.float32, init=A[idx])
                for _ in T.serial(work_iters):
                    x = x * 1.000001 + 0.000001
                Tail[idx] = x

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                y = T.alloc_var(T.float32, init=A[idx])
                for _ in T.serial(work_iters):
                    y = y * 1.000002 + 0.000002
                Pre[idx] = y
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                C[idx] = B[idx] * 2.0

    return main


def kernels_with_pdl_overlap_window(N, block_size=256, work_iters=8192, dtype=T.float32):
    """Create a two-kernel PDL benchmark with real pre-sync overlap work."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        Tail: T.Tensor((N,), dtype),
        Pre: T.Tensor((N,), dtype),
    ):
        """Trigger the consumer early so its pre-sync work can overlap producer tail work."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                B[idx] = A[idx] + 1.0
            T.pdl_trigger()
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                x = T.alloc_var(T.float32, init=A[idx])
                for _ in T.serial(work_iters):
                    x = x * 1.000001 + 0.000001
                Tail[idx] = x

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                y = T.alloc_var(T.float32, init=A[idx])
                for _ in T.serial(work_iters):
                    y = y * 1.000002 + 0.000002
                Pre[idx] = y
            T.pdl_sync()
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                C[idx] = B[idx] * 2.0

    return main


def _get_sm90_cuda_device():
    """Return an SM90+ CUDA device for runtime PDL tests."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CuTeDSL PDL runtime test requires CUDA")
    for device_id in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(device_id) >= (9, 0):
            return torch.device("cuda", device_id)
    pytest.skip("CuTeDSL PDL runtime test requires an SM90+ CUDA device")


def _package_version(package_name: str) -> str:
    """Return an installed package version or a readable placeholder."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _run_version_command(command: list[str], preferred_substring: str | None = None) -> str:
    """Return the first non-empty version command line, or a readable failure."""
    executable = shutil.which(command[0])
    if executable is None:
        return "not-found"
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as err:
        return f"error: {err}"
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    if preferred_substring is not None:
        for line in lines:
            if preferred_substring in line:
                return line
    return lines[-1] if lines else f"exit={result.returncode}"


def _run_command_summary(command: list[str]) -> str:
    """Return all non-empty command output lines joined for compact benchmark logs."""
    executable = shutil.which(command[0])
    if executable is None:
        return "not-found"
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as err:
        return f"error: {err}"
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return "; ".join(lines) if lines else f"exit={result.returncode}"


def _print_pdl_benchmark_environment(device):
    """Print exact local environment details for opt-in PDL benchmark runs."""
    torch = pytest.importorskip("torch")
    print("PDL microbenchmark environment:")
    print(f"  python={sys.version.split()[0]}")
    print(f"  platform={platform.platform()}")
    print(f"  tilelang_package={_package_version('tilelang')}")
    print(f"  tilelang_git={_run_version_command(['git', 'rev-parse', '--short', 'HEAD'])}")
    print(f"  torch={torch.__version__}")
    print(f"  torch_cuda={torch.version.cuda}")
    print(f"  cuda_home={os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or 'unset'}")
    print(f"  nvcc={_run_version_command(['nvcc', '--version'], preferred_substring='release')}")
    print(f"  nvidia_smi={_run_version_command(['nvidia-smi', '--version'], preferred_substring='NVIDIA-SMI version')}")
    print(f"  nvidia_driver_cuda={_run_version_command(['nvidia-smi', '--version'], preferred_substring='CUDA Version')}")
    print(f"  nvidia_gpus={_run_command_summary(['nvidia-smi', '--query-gpu=driver_version,name,compute_cap', '--format=csv,noheader'])}")
    print(f"  nvidia_cutlass_dsl={_package_version('nvidia-cutlass-dsl')}")
    print(f"  nvidia_cutlass_dsl_libs_base={_package_version('nvidia-cutlass-dsl-libs-base')}")
    print(f"  cuda_python={_package_version('cuda-python')}")
    print(f"  cuda_bindings={_package_version('cuda-bindings')}")
    print(f"  benchmark_device={device.index}:{torch.cuda.get_device_name(device)}")
    print(f"  benchmark_device_capability={torch.cuda.get_device_capability(device)}")
    print(f"  TILELANG_DISABLE_CACHE={os.environ.get('TILELANG_DISABLE_CACHE', '0')}")


def _benchmark_kernel_ms(kernel, args, tensors_to_clear, warmup, repeats):
    """Measure a TileLang kernel with CUDA events and return per-run milliseconds."""
    torch = pytest.importorskip("torch")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for run_id in range(warmup + repeats):
        for tensor in tensors_to_clear:
            tensor.zero_()
        torch.cuda.synchronize()
        start.record()
        kernel(*args)
        end.record()
        end.synchronize()
        if run_id >= warmup:
            samples.append(start.elapsed_time(end))
    return samples


@tilelang.testing.requires_cuda
def test_pdl_trigger():
    """Verify CUDA codegen emits the PDL trigger intrinsic."""

    N = 64
    program = kernels_with_pdl_trigger(N)

    pdl_kernel = tilelang.compile(program, target="cuda -arch=sm_90")
    code = pdl_kernel.get_kernel_source()
    assert "cudaTriggerProgrammaticLaunchCompletion" in code


@tilelang.testing.requires_cuda
def test_pdl_sync():
    """Verify CUDA codegen emits the PDL sync intrinsic without restrict qualifiers."""

    N = 64
    program = kernels_with_pdl_sync(N)
    pdl_kernel = tilelang.compile(program, target="cuda -arch=sm_90")
    code = pdl_kernel.get_kernel_source()
    assert "cudaGridDependencySynchronize" in code
    assert "__restrict__" not in code


def _lower_cutedsl_for_pdl(program):
    """Lower a PDL program through CuTeDSL and build its host wrapper."""

    try:
        from cutlass.cute import arch as cute_arch
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available
        from tilelang.jit.adapter.cutedsl.wrapper import TLCuTeDSLSourceWrapper
        from tilelang.utils.target import determine_target

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")

    if not (hasattr(cute_arch, "griddepcontrol_launch_dependents") and hasattr(cute_arch, "griddepcontrol_wait")):
        pytest.skip("CuTeDSL PDL APIs are not available in this nvidia-cutlass-dsl build (introduced in 4.3.4)")

    # PDL is an SM90/Hopper feature. Use an explicit arch so this codegen test
    # does not depend on the default CUDA device, which may be sm_80 on mixed-GPU hosts.
    target = determine_target("cutedsl -arch=sm_90")
    with target:
        artifact = tilelang.lower(program, target=target)
    mod = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    wrapper = TLCuTeDSLSourceWrapper(mod, artifact.kernel_source, target, artifact.device_mod, artifact.host_mod)
    return artifact, wrapper


def test_cutedsl_pdl_codegen_and_launcher_support():
    """Verify CuTeDSL PDL lowering and host launcher generation."""

    trigger_artifact, _ = _lower_cutedsl_for_pdl(kernels_with_pdl_trigger(64, block_size=64))
    assert "tl.griddepcontrol_launch_dependents()" in trigger_artifact.kernel_source

    sync_artifact, sync_wrapper = _lower_cutedsl_for_pdl(kernels_with_pdl_sync(64, block_size=64))
    assert "tl.griddepcontrol_wait()" in sync_artifact.kernel_source
    assert "use_pdl=True" in sync_wrapper.host_func
    assert "--gpu-arch=sm_90" in sync_wrapper.host_func

    launcher_cpp = sync_wrapper.get_launcher_cpp_code()
    assert "CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION" in launcher_cpp
    assert "cuLaunchKernelEx" in launcher_cpp

    import tilelang.contrib.cutedsl as tl

    assert hasattr(tl, "griddepcontrol_launch_dependents")
    assert hasattr(tl, "griddepcontrol_wait")


@tilelang.testing.requires_cuda
def test_cutedsl_pdl_runtime_pipeline():
    """Run a CuTeDSL PDL pipeline and verify the produced tensors."""

    torch = pytest.importorskip("torch")
    device = _get_sm90_cuda_device()
    N = 64

    with torch.cuda.device(device):
        _lower_cutedsl_for_pdl(kernels_with_pdl_pipeline(N, block_size=64))
        kernel = tilelang.compile(
            kernels_with_pdl_pipeline(N, block_size=64),
            target="cutedsl -arch=sm_90",
        )
        a = torch.randn(N, dtype=torch.float32, device=device)
        b = torch.empty_like(a)
        c = torch.empty_like(a)
        kernel(a, b, c)
        torch.cuda.synchronize(device)

    ref_b = a + 1.0
    ref_c = ref_b * 2.0
    tilelang.testing.torch_assert_close(b, ref_b, atol=1e-5, rtol=1e-5)
    tilelang.testing.torch_assert_close(c, ref_c, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
@pytest.mark.perf
def test_cutedsl_pdl_overlap_microbenchmark():
    """Benchmark PDL overlap against a strict same-stream serial launch baseline.

    This is intentionally opt-in via ``--run-perf``. It validates correctness
    and prints timings, but does not assert a speedup threshold because kernel
    overlap depends on occupancy, clocks, driver scheduling, and host load.
    """

    torch = pytest.importorskip("torch")
    device = _get_sm90_cuda_device()
    block_size = int(os.environ.get("TILELANG_PDL_BENCH_BLOCK_SIZE", "256"))
    num_blocks = int(os.environ.get("TILELANG_PDL_BENCH_BLOCKS", "1"))
    work_iters = int(os.environ.get("TILELANG_PDL_BENCH_WORK_ITERS", "8192"))
    warmup = int(os.environ.get("TILELANG_PDL_BENCH_WARMUP", "20"))
    repeats = int(os.environ.get("TILELANG_PDL_BENCH_REPEATS", "80"))
    N = num_blocks * block_size

    _print_pdl_benchmark_environment(device)
    print("PDL microbenchmark config:")
    print("  target=cutedsl -arch=sm_90")
    print(f"  num_blocks={num_blocks}")
    print(f"  block_size={block_size}")
    print(f"  work_iters={work_iters}")
    print(f"  warmup={warmup}")
    print(f"  repeats={repeats}")

    with torch.cuda.device(device):
        serial_kernel = tilelang.compile(
            kernels_without_pdl_overlap_window(N, block_size=block_size, work_iters=work_iters),
            target="cutedsl -arch=sm_90",
        )
        pdl_kernel = tilelang.compile(
            kernels_with_pdl_overlap_window(N, block_size=block_size, work_iters=work_iters),
            target="cutedsl -arch=sm_90",
        )
        a = torch.randn(N, dtype=torch.float32, device=device)
        serial_outputs = [torch.empty_like(a) for _ in range(4)]
        pdl_outputs = [torch.empty_like(a) for _ in range(4)]

        # First calls generate CuTeDSL cubins. Keep compilation out of the timing window.
        serial_kernel(a, *serial_outputs)
        pdl_kernel(a, *pdl_outputs)
        torch.cuda.synchronize(device)

        serial_samples = _benchmark_kernel_ms(
            serial_kernel,
            (a, *serial_outputs),
            serial_outputs,
            warmup=warmup,
            repeats=repeats,
        )
        pdl_samples = _benchmark_kernel_ms(
            pdl_kernel,
            (a, *pdl_outputs),
            pdl_outputs,
            warmup=warmup,
            repeats=repeats,
        )

    ref_b = a + 1.0
    ref_c = ref_b * 2.0
    tilelang.testing.torch_assert_close(serial_outputs[0], ref_b, atol=1e-5, rtol=1e-5)
    tilelang.testing.torch_assert_close(serial_outputs[1], ref_c, atol=1e-5, rtol=1e-5)
    tilelang.testing.torch_assert_close(pdl_outputs[0], ref_b, atol=1e-5, rtol=1e-5)
    tilelang.testing.torch_assert_close(pdl_outputs[1], ref_c, atol=1e-5, rtol=1e-5)

    serial_mean = statistics.mean(serial_samples)
    pdl_mean = statistics.mean(pdl_samples)
    serial_median = statistics.median(serial_samples)
    pdl_median = statistics.median(pdl_samples)
    print("PDL microbenchmark results:")
    print(f"  serial_mean_ms={serial_mean:.6f}")
    print(f"  serial_median_ms={serial_median:.6f}")
    print(f"  serial_min_ms={min(serial_samples):.6f}")
    print(f"  pdl_mean_ms={pdl_mean:.6f}")
    print(f"  pdl_median_ms={pdl_median:.6f}")
    print(f"  pdl_min_ms={min(pdl_samples):.6f}")
    print(f"  speedup_mean={serial_mean / pdl_mean:.6f}")
    print(f"  speedup_median={serial_median / pdl_median:.6f}")


if __name__ == "__main__":
    tilelang.testing.main()
