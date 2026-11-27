"""CuTeDSL expressions that emit statements in control-flow contexts."""

import ast

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.target import normalize_cutedsl_target
from tilelang.engine.lower import lower


@tilelang.testing.requires_cutedsl(support_required="compile-only")
def test_cutedsl_codegen_stateful_local_var_initializer():
    @T.prim_func
    def main(O: T.Tensor((1,), "int32"), gate: T.int32):
        with T.Kernel(1, threads=1):
            value = T.alloc_var("bool")
            T.sblock_attr({"tl.local_var_init": T.if_then_else(gate > 0, T.call_extern("bool", "test_flag"), False)})
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
