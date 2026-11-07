"""Ascend RNG range and codegen state isolation across device kernels.

One ``CodeGenTileLangAscend`` instance serves every ``PrimFunc`` of the device
module, so the Philox state variable name emitted for ``tl::philox_rand*`` is
a C++ local of the function that ran ``tl.rng_init`` and must never leak into a
sibling kernel.
"""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm

N = 256
THREADS = 64


def _rand_only_kernel():
    @T.prim_func
    def main(out: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            with T.SimtVF(threads=THREADS):
                tx = T.get_thread_binding()
                for i in T.serial(N // THREADS):
                    ub[i * THREADS + tx] = T.cast(T.rng_rand_float(dist="uniform"), T.float32)
            T.copy(ub, out)

    return main


def _two_kernel_program(second_kernel_has_init):
    """Kernel A always initializes RNG and draws; kernel B only draws."""

    @T.prim_func
    def main(out_a: T.Tensor((N,), T.float32), out_b: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            ub_a = T.alloc_shared((N,), T.float32)
            with T.SimtVF(threads=THREADS):
                tx = T.get_thread_binding()
                T.rng_init(42, seq=tx, off=0)
                for i in T.serial(N // THREADS):
                    ub_a[i * THREADS + tx] = T.cast(T.rng_rand_float(dist="uniform"), T.float32)
            T.copy(ub_a, out_a)

        with T.Kernel(1):
            ub_b = T.alloc_shared((N,), T.float32)
            with T.SimtVF(threads=THREADS):
                tx = T.get_thread_binding()
                if second_kernel_has_init:
                    T.rng_init(42, seq=tx, off=0)
                for i in T.serial(N // THREADS):
                    ub_b[i * THREADS + tx] = T.cast(T.rng_rand_float(dist="uniform"), T.float32)
            T.copy(ub_b, out_b)

    return main


@tilelang.testing.requires_ascend(support_required="compile-only")
def test_rng_rand_without_rng_init_is_rejected_at_codegen():
    with pytest.raises(tvm.error.InternalError, match="preceding T.rng_init"):
        tilelang.lower(_rand_only_kernel(), target="ascend")


@tilelang.testing.requires_ascend(support_required="compile-only")
def test_rng_state_does_not_leak_between_kernels():
    # If kernel A is emitted first, B must not inherit A's RNG state variable
    # name; if B is emitted first, B fails on its own. Both module iteration
    # orders must surface the same clear codegen error instead of generating
    # AscendC source that references another function's C++ local.
    with pytest.raises(tvm.error.InternalError, match="preceding T.rng_init"):
        tilelang.lower(_two_kernel_program(second_kernel_has_init=False), target="ascend")


@tilelang.testing.requires_ascend
def test_two_kernels_each_with_rng_init_get_isolated_state():
    kernel = tilelang.compile(_two_kernel_program(second_kernel_has_init=True), target="ascend")
    source = kernel.get_kernel_source()
    assert source.count("tl::AscendPhiloxState") == 2
    assert source.count("tl::philox_init(") == 2

    # Both kernels run the identical RNG program (same seed/seq/off), so their
    # outputs must match bitwise; this also proves the sibling kernel's state
    # reset did not disturb the stream of the first kernel.
    out_a = torch.empty(N, dtype=torch.float32, device="npu")
    out_b = torch.empty(N, dtype=torch.float32, device="npu")
    kernel(out_a, out_b)
    torch.npu.synchronize()
    a, b = out_a.cpu(), out_b.cpu()
    assert bool(((a > 0) & (a <= 1)).all())
    assert bool(((b > 0) & (b <= 1)).all())
    assert torch.equal(a, b)
    kernel(out_a, out_b)
    torch.npu.synchronize()
    assert torch.equal(a, out_a.cpu())


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_philox_uniform_endpoints(target):
    # Inverting Philox4x32-10 gives this aligned stream position whose first
    # block is [0, 0xffffff7f, 0xffffff80, 0xffffffff]. Check the raw words too
    # so the endpoint coverage cannot silently disappear if the stream changes.
    seed = 39
    seq = 904192493351575326
    off = 436980385747072128

    @T.prim_func
    def main(words: T.Tensor((4,), "uint32"), uniform: T.Tensor((4,), "float32")):
        with T.Kernel(1):
            words_ub = T.alloc_shared((4,), "uint32")
            uniform_ub = T.alloc_shared((4,), "float32")
            with T.SimtVF(threads=1):
                T.rng_init(seed, seq=T.int64(seq), off=T.int64(off))
                for i in T.serial(4):
                    words_ub[i] = T.rng_rand()
                T.rng_init(seed, seq=T.int64(seq), off=T.int64(off))
                for i in T.serial(4):
                    uniform_ub[i] = T.rng_rand_float()
            T.copy(words_ub, words)
            T.copy(uniform_ub, uniform)

    kernel = tilelang.compile(main, target=target, out_idx=[0, 1])
    words, uniform = kernel()
    torch.npu.synchronize()
    assert torch.equal(words.cpu(), torch.tensor([0, 0xFFFFFF7F, 0xFFFFFF80, 0xFFFFFFFF], dtype=torch.uint32))
    # The smallest draw is positive. The last two words round up to one,
    # while the preceding word yields the largest float32 below one.
    expected = torch.tensor([2.0**-33, 1.0 - 2.0**-24, 1.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(uniform.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
