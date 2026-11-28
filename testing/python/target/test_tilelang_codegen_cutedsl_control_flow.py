"""CuTeDSL expressions that emit statements in control-flow contexts."""

import ast

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.target import normalize_cutedsl_target
from tilelang.engine.lower import lower


@tilelang.testing.requires_cutedsl(support_required="compile-only")
def test_cutedsl_codegen_stateful_local_var_initializer():
    @T.prim_func
    def main(O: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            value = T.alloc_var("bool")
            # Keep the predicate dynamic without a launch parameter that is
            # only referenced from the allocation annotation.
            T.sblock_attr(
                {"tl.local_var_init": T.if_then_else(T.call_pure_extern("bool", "test_gate"), T.call_extern("bool", "test_flag"), False)}
            )
            O[0] = T.cast(value, "int32")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    source = lower(main, target=target).kernel_source
    # The annotation initializes the scalar in AllocBuffer, where emitting
    # the assignment prefix before the conditional statements breaks syntax.
    assert source.count("test_flag()") == 1
    ast.parse(source)


@tilelang.testing.requires_cutedsl
def test_cutedsl_while_rechecks_stateful_chained_comparison():
    @T.prim_func
    def main(O: T.Tensor((2,), "int64")):
        with T.Kernel(1, threads=1):
            T.rng_init(1234567, 0)
            iterations = T.alloc_var("int32", init=0)
            # The first Philox draw passes this range check and the second
            # fails. The independent limit also bounds a regressed loop.
            while (1 << 31) <= T.cast(T.rng_rand(), "uint64") < (1 << 32) and iterations < 2:
                iterations += 1
            O[0] = T.cast(iterations, "int64")
            O[1] = T.cast(T.rng_rand(), "int64")

    output = torch.empty((2,), dtype=torch.int64, device="cuda")
    tilelang.compile(main, target="cutedsl")(output)
    # One body iteration, followed by the third draw.
    assert output.tolist() == [1, 3457867836]


@tilelang.testing.requires_cutedsl
@pytest.mark.parametrize("direct_store", [False, True], ids=["bind", "store"])
def test_cutedsl_nested_conditional_keeps_rng_in_outer_arm(direct_store):
    def nested_value(outer_gate, inner_gate):
        return T.if_then_else(
            outer_gate > 0,
            T.cast(T.if_then_else(inner_gate > 0, T.rng_rand() > 0, False), "int32"),
            7,
        )

    @T.prim_func
    def main(G: T.Tensor((1,), "int32"), O: T.Tensor((2, 2), "int64")):
        with T.Kernel(2, threads=1) as outer_gate:
            T.rng_init(1234567, 0)
            if direct_store:
                O[outer_gate, 0] = T.cast(nested_value(outer_gate, G[0]), "int64")
            else:
                value = nested_value(outer_gate, G[0])
                O[outer_gate, 0] = T.cast(value, "int64")
            O[outer_gate, 1] = T.cast(T.rng_rand(), "int64")

    inner_gate = torch.ones((1,), dtype=torch.int32, device="cuda")
    output = torch.empty((2, 2), dtype=torch.int64, device="cuda")
    tilelang.compile(main, target="cutedsl")(inner_gate, output)
    # With the inner gate true in both blocks, only the outer true arm
    # may advance the RNG. The final draws are the first and second sample.
    assert output.tolist() == [[7, 3688262095], [1, 1022100329]]
