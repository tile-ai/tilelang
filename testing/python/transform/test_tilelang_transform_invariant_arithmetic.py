"""Preparation sharing and compile-time stress tests without NVCC timing noise."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


@pytest.mark.parametrize(
    "target, multiply",
    [
        pytest.param("cutedsl", "tl.mul_hi", marks=tilelang.testing.requires_cuda.marks()),
        pytest.param("hip", "__umulhi", marks=tilelang.testing.requires_rocm.marks()),
    ],
)
def test_invariant_arithmetic_device_codegen(target, multiply):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32"), x: T.int32, d: T.int32, mu: T.uint32, shift: T.int32):
        with T.Kernel(1, threads=1):
            B[0] = T.call_intrin("int32", tvm.ir.Op.get("tl.fast_div"), x, d, mu, shift, True, False, False, False)
            B[1] = T.call_intrin("int32", tvm.ir.Op.get("tl.fast_rem"), x, d, mu, shift, True, False, False, False)
            B[2] = T.call_intrin("int32", tvm.ir.Op.get("tl.barrett_reduce"), x, d, mu, True, False, False, False)
            B[3] = T.call_intrin("int32", tvm.ir.Op.get("tl.bounded_rem"), x, d)

    if target == "cutedsl":
        # Source-only CUDA lowering does not require the optional CuTe runtime.
        mod = tilelang.lower(main, target={"kind": "cuda", "arch": "sm_80"}).device_mod
        target = tvm.target.Target({"kind": "cuda", "keys": ["cutedsl", "cuda"], "arch": "sm_80"})
        source = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile")(mod, target).inspect_source()
    else:
        source = tilelang.lower(main, target={"kind": "hip", "mcpu": "gfx942"}).kernel_source
    assert multiply in source
    assert "tl.fast_div" not in source and "tl::fast_div" not in source


def _prepare(divisors, params, copies=1, complex_index=False):
    out = tirx.decl_buffer((len(divisors) * copies,), "uint64", name="out")
    x = tirx.Var("x", "uint64")
    stores = []
    for i in range(copies):
        for j, divisor in enumerate(divisors):
            # Distinct numerators keep the expression forest alive throughout
            # simplification; divisors remain launch-invariant.
            index = x + i
            if complex_index:
                for _ in range(3):
                    low = tirx.bitwise_and(index, tirx.const(63, "uint64"))
                    index = index - low + tirx.bitwise_xor(low, low >> 3)
            value = tirx.floordiv(index, tirx.Cast("uint64", divisor))
            stores.append(tirx.BufferStore(out, value, [i * len(divisors) + j]))
    body = tirx.AttrStmt(tvm.target.Target("cuda"), "target", 0, tirx.SeqStmt(stores))
    func = tirx.PrimFunc([out.data, x, *params], body)
    mod = tilelang.transform.LowerInvariantArithmetic()(tvm.IRModule({"main": func}))
    bindings = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda n: bindings.append(n) if isinstance(n, tirx.Bind) else None)
    return mod, bindings


@tilelang.testing.requires_cuda
def test_product_divisor_preparation_sharing():
    a, b, c = [tirx.Var(name, "uint64") for name in ("a", "b", "c")]
    _, bindings = _prepare([(a * b) * c, c * (b * a), (b * c) * a], [a, b, c])
    assert sum("barrett_reciprocal" in n.var.name for n in bindings) == 1


@tilelang.testing.requires_cuda
def test_product_divisor_preparation_boundaries():
    a, b = [tirx.Var(name, "uint32") for name in ("a", "b")]
    # Same spelling must not merge distinct variables. Casts and repeated
    # factors must not be erased by product matching.
    other_a = tirx.Var("a", "uint32")

    def wide(x):
        return tirx.Cast("uint64", x)

    divisors = [wide(a * b), wide(a) * wide(b), wide(other_a * b), wide(a * a * b), wide(a * b * b)]
    _, bindings = _prepare(divisors, [a, b, other_a])
    assert sum("barrett_reciprocal" in n.var.name for n in bindings) == len(divisors)


def _stress_worker():
    params = [tirx.Var(f"d{i}", "uint64") for i in range(8)]
    divisors = []
    for rotation in range(len(params)):
        factors = params[rotation:] + params[:rotation]
        product = factors[0]
        for factor in factors[1:]:
            product = product * factor
        divisors.append(product)
    # Measure IR construction and preparation, excluding imports, native
    # library loading and NVCC. Keep a modest scaling series in the output.
    measurements = []
    for copies in (1, 8, 32):
        start = time.perf_counter()
        _, bindings = _prepare(divisors, params, copies, complex_index=True)
        elapsed = time.perf_counter() - start
        assert sum("barrett_reciprocal" in n.var.name for n in bindings) == 1
        measurements.append({"sites": len(divisors) * copies, "seconds": elapsed})
    print(json.dumps(measurements))


@tilelang.testing.requires_cuda
def test_large_product_preparation_compile_budget():
    # A process timeout also stops a stuck native prover. Use a generous
    # catastrophe budget, not a flaky sub-second threshold or timing ratio.
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stress-worker"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    measurements = json.loads(result.stdout.splitlines()[-1])
    assert [row["sites"] for row in measurements] == [8, 64, 256]
    print(measurements)


if __name__ == "__main__":
    if "--stress-worker" in sys.argv:
        _stress_worker()
    else:
        tilelang.testing.main()
