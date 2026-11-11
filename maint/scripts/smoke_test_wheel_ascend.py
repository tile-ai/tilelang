"""Ascend NPU smoke test for built tilelang wheels.

Runs against an *installed* tilelang (wheel), not the source tree: copy this
file somewhere outside the repository before running it, or run it with the
repository root absent from ``sys.path``, so ``import tilelang`` resolves to
site-packages. It compiles and launches a small pipelined SIMT vector
addition kernel on the NPU (mirroring examples/ascend/example_vecadd.py) and
checks the result against torch.

Requires an Ascend environment with CANN and a matching torch/torch_npu
stack, e.g. the CI container image used by .github/workflows/ci.yml.

Exits non-zero on any failure.
"""

import sys

import torch
import torch_npu  # noqa: F401  # registers the npu device with torch

import tilelang
import tilelang.ascend.language as T


def report_environment() -> None:
    print("=" * 60)
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__, "| torch_npu:", torch_npu.__version__)
    assert torch.npu.is_available(), "torch reports no NPU; an Ascend NPU is required for the wheel smoke test"
    print("npu:", torch.npu.get_device_name(0))
    print("tilelang:", tilelang.__version__)

    from tilelang.backend.target import determine_target

    target = determine_target(return_object=True)
    print("detected target:", target)
    assert "ascend" in str(target).lower(), f"expected an Ascend target, got {target}"
    print("=" * 60)


def vector_add(N: int):
    """Small pipelined SIMT vector addition, matching examples/ascend/example_vecadd.py."""
    num_blocks, num_threads, tile, num_stages = 64, 2048, 8192, 2
    assert N > 0 and N % (tile * num_blocks) == 0, f"N must be a positive multiple of {tile * num_blocks}"
    iterations = N // (tile * num_blocks)

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                with T.SimtVF(threads=num_threads):
                    for i in T.Parallel(tile):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + tile])

    return main


def main() -> None:
    report_environment()

    n = 2**20
    kernel = tilelang.compile(vector_add(n), out_idx=-1, target="ascend")

    a, b = torch.randn(n, device="npu"), torch.randn(n, device="npu")
    torch.testing.assert_close(kernel(a, b), a + b)
    print("correctness: PASS")
    print("WHEEL SMOKE TEST (Ascend): ALL PASS")


if __name__ == "__main__":
    main()
