"""T.alloc_global on the CuTeDSL backend: the Python host allocates the workspace
per call (the CUDA host codegen uses TVMBackendAllocWorkspace) and passes it to
every kernel that uses it."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def staged_copy(n, block, dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        C = T.alloc_global((n,), dtype)
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            T.copy(A[bx * block : (bx + 1) * block], C[bx * block : (bx + 1) * block])
            T.copy(C[bx * block : (bx + 1) * block], B[bx * block : (bx + 1) * block])

    return main


def split_then_combine(block, splits):
    """Two kernels share two workspaces whose size follows a dynamic dimension."""
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n, splits), "float32"), B: T.Tensor((n,), "float32")):
        partial = T.alloc_global((n, splits), "float32")
        count = T.alloc_global((n,), "int32")
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i, s in T.Parallel(block, splits):
                if bx * block + i < n:
                    partial[bx * block + i, s] = A[bx * block + i, s] * 2.0
            for i in T.Parallel(block):
                if bx * block + i < n:
                    count[bx * block + i] = splits
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                if bx * block + i < n:
                    acc = T.alloc_var("float32")
                    acc = 0.0
                    for s in T.serial(splits):
                        acc += partial[bx * block + i, s]
                    B[bx * block + i] = acc / T.Cast("float32", count[bx * block + i])

    return main


CUTEDSL_SM90_TARGET = {"kind": "cutedsl", "arch": "sm_90"}


def _call_signature(host_source):
    return next(line for line in host_source.splitlines() if line.startswith("def call("))


def test_cutedsl_alloc_global_host_wrapper():
    _require_cutedsl()
    kernel = tilelang.compile(staged_copy(1024, 128, "float16"), target=CUTEDSL_SM90_TARGET, execution_backend="cutedsl")
    host = kernel.get_host_source()
    # A and B come from the caller; the workspace is allocated inside call().
    assert _call_signature(host).count("_tl_arg_") == 2
    assert host.count("torch.empty((1024,), dtype=torch.float16, device=_ws_device)") == 1
    assert "with torch.cuda.stream(_ws_stream):" in host
    assert "else torch.cuda.default_stream(_ws_device)" in host


def test_cutedsl_alloc_global_dynamic_shape_host_wrapper():
    _require_cutedsl()
    kernel = tilelang.compile(split_then_combine(128, 4), target=CUTEDSL_SM90_TARGET, execution_backend="cutedsl")
    host = kernel.get_host_source()
    assert _call_signature(host).count("_tl_arg_") == 3
    assert host.count("torch.empty(") == 2
    assert "dtype=torch.int32" in host


@tilelang.testing.requires_cuda
def test_cutedsl_alloc_global_staged_copy():
    _require_cutedsl()
    kernel = tilelang.compile(staged_copy(1024, 128, "float16"), out_idx=[1], target="cutedsl")
    a = torch.randn(1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a)


@tilelang.testing.requires_cuda
def test_cutedsl_alloc_global_shared_by_two_kernels():
    _require_cutedsl()
    kernel = tilelang.compile(split_then_combine(128, 4), out_idx=[1], target="cutedsl")
    for n in (1000, 4096):
        a = torch.randn(n, 4, device="cuda")
        torch.testing.assert_close(kernel(a), a.sum(dim=1) * 2.0 / 4)


@tilelang.testing.requires_cuda
def test_cutedsl_alloc_global_on_side_stream():
    _require_cutedsl()
    kernel = tilelang.compile(split_then_combine(128, 4), out_idx=[1], target="cutedsl")
    a = torch.randn(4096, 4, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    out = kernel(a, stream=side.cuda_stream)
    side.synchronize()
    torch.testing.assert_close(out, a.sum(dim=1) * 2.0 / 4)


@tilelang.testing.requires_cuda
def test_cutedsl_alloc_global_on_explicit_default_stream_from_side_stream():
    _require_cutedsl()
    kernel = tilelang.compile(split_then_combine(128, 4), out_idx=[1], target="cutedsl")
    a = torch.randn(4096, 4, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        out = kernel(a, stream=0)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a.sum(dim=1) * 2.0 / 4)


if __name__ == "__main__":
    tilelang.testing.main()
