"""Exercise CPU compilation, execution, and persistent cache loading from a wheel."""

import argparse
import json

import torch
import tilelang
import tilelang.cpu.language as T


@T.prim_func
def affine(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
    with T.Kernel(1):
        for i in T.serial(32):
            B[i] = A[i] * 2 + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-reload", action="store_true")
    args = parser.parse_args()
    if args.cache_reload:
        from tilelang.contrib import cc, msvc
        from tilelang.jit.adapter.libgen import LibraryGenerator

        def no_compile(*args, **kwargs):
            raise AssertionError("The fresh process must load the cached kernel without compiling")

        LibraryGenerator.compile_lib = no_compile
        cc.create_shared = msvc.create_shared = no_compile

    kernel = tilelang.compile(affine, target="c", out_idx=[1], execution_backend="cython")
    value = torch.arange(32, dtype=torch.float32)
    torch.testing.assert_close(kernel(value), value * 2 + 1)
    print(json.dumps({"package": tilelang.__file__, "backend": "cython", "cache_reload": args.cache_reload}))


if __name__ == "__main__":
    main()
