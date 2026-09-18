"""Shared utilities for Metal lowering tests."""

import tilelang
from tilelang import tvm as tvm


def lower_prim_to_metal(prim_func, *, target="metal") -> str:
    """Lower a TIR prim_func to Metal shader source.

    This is the canonical Metal lowering path shared by tests: it creates a
    TVM Metal target with an LLVM host target, lowers the function with host
    codegen and device compilation disabled, and returns the emitted MSL.
    Tests that only need the generated shader text should use this helper
    instead of inlining the same ``tilelang.lower`` call.
    """
    target = tvm.target.Target(target, tvm.target.Target("llvm"))
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            target_host="llvm",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact.kernel_source or ""
