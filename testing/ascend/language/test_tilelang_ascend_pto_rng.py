"""PTO RNG state isolation across device kernels.

One ``CodeGenTileLangPTO`` instance serves every ``PrimFunc`` of the device
module, so all RNG emission state is function-local by contract: the Philox
state/counter SSA names are Python locals of the function that ran
``tl.rng_init`` and must never leak into a sibling kernel. Regression tests for
the state reset in ``CodeGenTileLangPTO::AddFunction``.
"""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.backend.target import determine_target
from tilelang.engine.lower import lower

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
    def main(
        out_a: T.Tensor((N,), T.float32),
        out_b: T.Tensor((N,), T.float32),
    ):
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


def _jit_rng_kernel(dist):
    @T.prim_func
    def main(out: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            with T.SimtVF(threads=THREADS):
                tx = T.get_thread_binding()
                T.rng_init(42, seq=tx, off=0)
                for i in T.serial(N // THREADS):
                    ub[i * THREADS + tx] = T.cast(T.rng_rand_float(dist=dist), T.float32)
            T.copy(ub, out)

    return main


def _jit_functions(source):
    funcs = [f for f in source.split("@pto.jit") if f.strip()]
    assert funcs, "generated PTODSL contains no @pto.jit kernel"
    return funcs


@pytest.mark.pto
def test_pto_rng_rand_without_rng_init_is_rejected_at_codegen():
    program = _rand_only_kernel()
    with determine_target("pto", return_object=True), pytest.raises(Exception, match="without prior tl.rng_init"):
        lower(program, target="pto")


@pytest.mark.pto
def test_pto_rng_state_does_not_leak_between_kernels():
    # If kernel A is emitted first, B must not inherit A's rng-initialized
    # flag; if B is emitted first, B fails on its own. Both module iteration
    # orders must surface the same clear codegen error instead of generating
    # PTODSL that references another function's Python locals.
    program = _two_kernel_program(second_kernel_has_init=False)
    with determine_target("pto", return_object=True), pytest.raises(Exception, match="without prior tl.rng_init"):
        lower(program, target="pto")


@pytest.mark.pto
def test_pto_two_kernels_each_with_rng_init_get_isolated_state():
    program = _two_kernel_program(second_kernel_has_init=True)
    with determine_target("pto", return_object=True):
        artifact = lower(program, target="pto")
    source = str(artifact.kernel_source)

    assert source.count("tl.PhiloxRNG(") == 2
    # Every kernel that uses RNG state defines it first: no function may
    # reference _tl_rng_state/_tl_rng_counter without its own PhiloxRNG setup.
    for func in _jit_functions(source):
        uses_state = "_tl_rng_state" in func
        defines_state = "tl.PhiloxRNG(" in func
        assert uses_state == defines_state, "PTODSL kernel references RNG state it did not initialize:\n" + func

    # Both kernels run the identical RNG program (same seed/seq/off), so their
    # outputs must match bitwise; this also proves the sibling kernel's state
    # reset did not disturb the stream of the first kernel.
    kernel = tilelang.compile(program, target="pto")
    out_a = torch.empty(N, dtype=torch.float32, device="npu")
    out_b = torch.empty(N, dtype=torch.float32, device="npu")
    kernel(out_a, out_b)
    torch.npu.synchronize()
    a, b = out_a.cpu(), out_b.cpu()
    assert bool(((a >= 0) & (a < 1)).all())
    assert bool(((b >= 0) & (b < 1)).all())
    assert torch.equal(a, b)
    kernel(out_a, out_b)
    torch.npu.synchronize()
    assert torch.equal(a, out_a.cpu())


@pytest.mark.pto
@pytest.mark.parametrize("dist", ["uniform", "normal"])
def test_pto_rng_matches_ascend_backend(dist):
    # Both backends implement the same Philox 4x32-10 stream and Box-Muller
    # construction. Uniform draws must agree bitwise; normal draws may differ
    # by transcendental-function precision on the two codegens.
    results = {}
    for target in ("ascend", "pto"):
        kernel = tilelang.compile(_jit_rng_kernel(dist), target=target)
        out = torch.empty(N, dtype=torch.float32, device="npu")
        kernel(out)
        torch.npu.synchronize()
        results[target] = out.cpu()

    if dist == "uniform":
        assert torch.equal(results["ascend"], results["pto"])
    else:
        assert torch.allclose(results["ascend"], results["pto"], atol=1e-5, rtol=0)
    # Sanity: uniform draws lie in [0, 1).
    if dist == "uniform":
        assert bool(((results["pto"] >= 0) & (results["pto"] < 1)).all())


if __name__ == "__main__":
    tilelang.testing.main()
