# Test Plan: Disk Cache Verification using PostProc Callback
#
# Purpose: Reliably test disk cache in CI by using postproc callbacks to detect
#          whether compilation actually happened or cache was used.
#
# Strategy:
# - postproc is ONLY called during codegen (cache miss)
# - postproc is NOT called when loading from cache (cache hit)
# - Use a counter in postproc to distinguish these cases
#
# CI Safety:
# - Use isolated cache/tmp directories per test (pytest tmp_path)
# - Use unique kernel identifiers (UUID + global_symbol) to avoid collisions
# - Clear memory cache between passes to force disk I/O
# - os.replace() requires source and dest on same filesystem (atomic rename)
#
# Technical Details:
# - Cache key is based on func.script(show_meta=True) hash
# - Python comments do NOT affect cache key (not in TIR)
# - Must use .with_attr("global_symbol", ...) to create unique cache keys

import pytest
import tilelang
import tilelang.language as T
import tvm_ffi
import torch
import uuid
from pathlib import Path
from tilelang.env import env
from tilelang.cache import _dispatch_map

BACKENDS = [
    "tvm_ffi",
    "cython",
    "nvrtc",
    "cutedsl",
]


def _get_target_from_backend(backend: str):
    """Map backend to target string."""
    return "cutedsl" if backend == "cutedsl" else "auto"


class PostProcCounter:
    """Track postproc callback invocations with a simple counter."""

    def __init__(self):
        self.count = 0
        self.marker = None

    def register_callback(self, backend: str):
        """Register postproc callback for the given backend."""
        comment_prefix = "#" if backend == "cutedsl" else "//"
        global_func = "tilelang_callback_cutedsl_postproc" if backend == "cutedsl" else "tilelang_callback_cuda_postproc"

        def callback(code, _):
            self.count += 1
            self.marker = f"{comment_prefix} CACHE_TEST_MARKER_{self.count}"
            return self.marker + "\n" + code

        tvm_ffi.register_global_func(global_func, f=callback, override=True)
        return callback


@pytest.fixture(scope="module", autouse=True)
def setup_module_env():
    """Setup and restore module-level environment and cache state."""
    # Save original env values
    original_cache_dir = env.TILELANG_CACHE_DIR
    original_tmp_dir = env.TILELANG_TMP_DIR

    # Enable cache once for entire module
    tilelang.enable_cache()

    yield

    # Restore env at module end
    env.TILELANG_CACHE_DIR = original_cache_dir
    env.TILELANG_TMP_DIR = original_tmp_dir

    # Restore default postproc callbacks
    tvm_ffi.register_global_func("tilelang_callback_cuda_postproc", f=lambda code, _: code, override=True)
    tvm_ffi.register_global_func("tilelang_callback_cutedsl_postproc", f=lambda code, _: code, override=True)


@pytest.fixture(scope="function")
def clean_cache_env(tmp_path, request):
    """Provide isolated cache environment for each test.

    Creates isolated cache/tmp directories to ensure:
    - No interference from previous test runs
    - No interference between parallel tests
    - Clean slate for testing cache miss/hit behavior
    - No "Invalid cross-device link" errors (os.replace requires same filesystem)

    Technical notes:
    - TILELANG_TMP_DIR MUST be on same filesystem as TILELANG_CACHE_DIR because
      cache implementation uses os.replace() for atomic writes
    - Env restoration is handled by setup_module_env at module scope
    """
    # This fixture should ONLY be used with @pytest.mark.parametrize("backend", ...)
    backend = request.node.callspec.params["backend"]  # Will raise KeyError if missing

    cache_dir = tmp_path / "tilelang_cache"
    cache_dir.mkdir()

    tmp_dir = tmp_path / "tilelang_tmp"
    tmp_dir.mkdir()

    # Patch env variables to point to isolated directories
    env.TILELANG_CACHE_DIR = str(cache_dir)
    env.TILELANG_TMP_DIR = str(tmp_dir)

    # Clear memory caches to force disk I/O
    _dispatch_map[backend]._memory_cache.clear()

    return cache_dir


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_disk_cache_with_postproc(clean_cache_env, backend):
    """Test disk cache for multiple backends using postproc callback.

    Tests all CUDA-based backends: nvrtc, cutedsl
    (tvm_ffi, cython, torch use the same cuda_postproc callback as nvrtc)

    Verification logic:
    1. Pass 1: cache miss → postproc called → marker in kernel source
    2. Pass 2: cache hit → postproc NOT called → same marker still in source
    3. Verify cache files created on disk
    4. Verify functional correctness
    """
    counter = PostProcCounter()
    counter.register_callback(backend)

    # Use UUID in global_symbol to ensure unique cache key per test run
    unique_id = uuid.uuid4().hex[:8]
    M, N = 1024, 1024

    @T.prim_func
    def vector_add(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(M, threads=256) as bx:
            for i in T.serial(N):
                C[bx, i] = A[bx, i] + B[bx, i]

    kernel_func = vector_add.with_attr("global_symbol", f"vector_add_{backend}_{unique_id}")

    # === Pass 1: Cache miss (memory cache already cleared by fixture) ===
    kernel1 = tilelang.compile(
        kernel_func,
        out_idx=[2],
        target=_get_target_from_backend(backend),
        execution_backend=backend,
    )

    assert counter.count == 1, f"Cache miss: postproc should be called once, got {counter.count}"

    source1 = kernel1.get_kernel_source()
    assert counter.marker in source1, f"Expected marker '{counter.marker}' in kernel source"

    # Verify cache files created
    cache_files = list(Path(clean_cache_env).rglob("*.*"))
    assert len(cache_files) > 0, "Cache files should be created, found none"

    # === Pass 2: Cache hit (clear memory cache to force disk read) ===
    _dispatch_map[backend]._memory_cache.clear()

    kernel2 = tilelang.compile(
        kernel_func,
        out_idx=[2],
        target=_get_target_from_backend(backend),
        execution_backend=backend,
    )

    assert counter.count == 1, f"Cache hit: postproc should not be called again, got {counter.count} calls"

    source2 = kernel2.get_kernel_source()
    assert counter.marker in source2, f"Expected cached marker '{counter.marker}' in source"

    # === Verify functional correctness ===
    a = torch.randn(M, N, dtype=torch.float32).cuda()
    b = torch.randn(M, N, dtype=torch.float32).cuda()

    c1 = kernel1(a, b)
    c2 = kernel2(a, b)
    ref = a + b

    torch.testing.assert_close(c1, ref)
    torch.testing.assert_close(c2, ref)
    torch.testing.assert_close(c1, c2)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_cache_miss_detection(clean_cache_env, backend):
    """Verify cache correctly misses when function changes.

    This ensures our testing method is valid - different functions should
    produce different cache keys and trigger recompilation.
    """
    counter = PostProcCounter()
    counter.register_callback(backend)

    M, N = 512, 512

    # Kernel 1: A + 1.0
    @T.prim_func
    def func1(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(M, threads=128) as bx:
            for i in T.serial(N):
                B[bx, i] = A[bx, i] + 1.0

    unique_id_1 = uuid.uuid4().hex[:8]
    kernel_func1 = func1.with_attr("global_symbol", f"func1_{backend}_{unique_id_1}")

    tilelang.compile(
        kernel_func1,
        out_idx=[1],
        target=_get_target_from_backend(backend),
        execution_backend=backend,
    )
    assert counter.count == 1, f"First kernel: expected 1 call, got {counter.count}"

    # Kernel 2: A + 2.0 (different implementation)
    @T.prim_func
    def func2(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(M, threads=128) as bx:
            for i in T.serial(N):
                B[bx, i] = A[bx, i] + 2.0  # Different!

    unique_id_2 = uuid.uuid4().hex[:8]
    kernel_func2 = func2.with_attr("global_symbol", f"func2_{backend}_{unique_id_2}")

    tilelang.compile(
        kernel_func2,
        out_idx=[1],
        target=_get_target_from_backend(backend),
        execution_backend=backend,
    )

    assert counter.count == 2, f"Different function should cause cache miss, expected 2 calls, got {counter.count}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_cache_isolation_between_tests(clean_cache_env, backend):
    """Verify cache isolation between tests.

    Ensures clean_cache_env fixture provides independent cache directory for each test.
    """
    # Verify cache directory is empty
    cache_files = list(Path(clean_cache_env).rglob("*"))
    assert all(f.is_dir() for f in cache_files), f"Cache should be empty, found: {cache_files}"

    # Compile a kernel
    counter = PostProcCounter()
    counter.register_callback(backend)

    unique_id = uuid.uuid4().hex[:8]

    @T.prim_func
    def simple(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(128, threads=128) as i:
            B[i] = A[i] * 2.0

    kernel_func = simple.with_attr("global_symbol", f"simple_{backend}_{unique_id}")

    tilelang.compile(
        kernel_func,
        out_idx=[1],
        target=_get_target_from_backend(backend),
        execution_backend=backend,
    )

    # Should be cache miss (empty cache dir)
    assert counter.count == 1, f"Expected cache miss, got count={counter.count}"

    # Verify cache files created
    cache_files_after = list(Path(clean_cache_env).rglob("*.*"))
    assert len(cache_files_after) > 0, f"Cache files should be created, found: {cache_files_after}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
