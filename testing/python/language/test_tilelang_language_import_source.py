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
# A trailing backslash must not comment out the next imported helper.
BACKSLASH = " // comment \\"
WRAPPER = "__device__ int call_helper(int x) { return my_helper(x); }"


def _make_import_source_kernel(prelude):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128):
            T.import_source(prelude)
            with T.sblock():
                T.import_source(WRAPPER)
                tid = T.get_thread_binding(0)
                B[tid] = T.call_extern("int32", "call_helper", A[tid])

    return main


def _make_prelude_kernel(prelude):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128, prelude=prelude), T.sblock():
            T.import_source(WRAPPER)
            tid = T.get_thread_binding(0)
            B[tid] = T.call_extern("int32", "call_helper", A[tid])

    return main


def _check(kernel):
    a = torch.arange(128, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    kernel(a, b)
    assert torch.equal(b, a + 1)


PRELUDES = [PRELUDE, PRELUDE + "\n", PRELUDE + BACKSLASH, PRELUDE + BACKSLASH + "\n"]


@pytest.mark.parametrize("prelude", PRELUDES)
@tilelang.testing.requires_cuda
def test_import_source_ends_with_newline(prelude):
    _check(tilelang.compile(_make_import_source_kernel(prelude), target="cuda"))


@pytest.mark.parametrize("prelude", PRELUDES)
@tilelang.testing.requires_cuda
def test_kernel_prelude_ends_with_newline(prelude):
    _check(tilelang.compile(_make_prelude_kernel(prelude), target="cuda"))


@pytest.mark.parametrize("prelude", PRELUDES)
@tilelang.testing.requires_cuda
def test_import_attr_ends_with_newline(prelude):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128), T.attr(0, "pragma_import_c", prelude), T.attr(0, "pragma_import_c", WRAPPER):
            tid = T.get_thread_binding(0)
            B[tid] = T.call_extern("int32", "call_helper", A[tid])

    _check(tilelang.compile(main, target="cuda"))


@pytest.mark.parametrize("make_kernel", [_make_import_source_kernel, _make_prelude_kernel])
@tilelang.testing.requires_cuda
def test_import_uses_tl_templates(make_kernel):
    _check(tilelang.compile(make_kernel(TL_PRELUDE), target="cuda"))


if __name__ == "__main__":
    tilelang.testing.main()
