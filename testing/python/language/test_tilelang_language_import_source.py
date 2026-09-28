"""Tests for imported C source (``T.import_source`` and ``T.Kernel(prelude=...)``).

An import that does not end in a newline must not fuse with the code the
codegen emits next (a declaration followed by a preprocessor line used to
break nvcc with "#endif without #if"). Imports are emitted after the backend
headers, so they can use what ``tl_templates`` declares.
"""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T

# Deliberately lacks a trailing newline.
PRELUDE = "__device__ int my_helper(int x) { return x + 1; }"
# `TL_DEVICE` comes from tl_templates, so this only compiles after the headers.
TL_PRELUDE = "TL_DEVICE int my_helper(int x) { return x + 1; }"


def _make_import_source_kernel(prelude):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128):
            T.import_source(prelude)
            tid = T.get_thread_binding(0)
            B[tid] = T.call_extern("int32", "my_helper", A[tid])

    return main


def _make_prelude_kernel(prelude):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128, prelude=prelude):
            tid = T.get_thread_binding(0)
            B[tid] = T.call_extern("int32", "my_helper", A[tid])

    return main


def _check(kernel, prelude=PRELUDE):
    a = torch.arange(128, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    kernel(a, b)
    assert torch.equal(b, a + 1)

    source = kernel.get_kernel_source()
    assert f"{prelude}\n" in source
    assert source.index(prelude) > source.rindex("#include")
    assert "}#if" not in source
    assert "}#include" not in source


@pytest.mark.parametrize("prelude", [PRELUDE, PRELUDE + "\n"])
@tilelang.testing.requires_cuda
def test_import_source_ends_with_newline(prelude):
    _check(tilelang.compile(_make_import_source_kernel(prelude), target="cuda"))


@pytest.mark.parametrize("prelude", [PRELUDE, PRELUDE + "\n"])
@tilelang.testing.requires_cuda
def test_kernel_prelude_ends_with_newline(prelude):
    _check(tilelang.compile(_make_prelude_kernel(prelude), target="cuda"))


@pytest.mark.parametrize("make_kernel", [_make_import_source_kernel, _make_prelude_kernel])
@tilelang.testing.requires_cuda
def test_import_uses_tl_templates(make_kernel):
    _check(tilelang.compile(make_kernel(TL_PRELUDE), target="cuda"), TL_PRELUDE)


if __name__ == "__main__":
    tilelang.testing.main()
