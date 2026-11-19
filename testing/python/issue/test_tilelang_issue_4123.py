"""Regression tests for #4123: let-bound warp collectives stay at their binding.

``tl.force_let_inline`` substitutes a let-bound value at every use.  A warp
shuffle, ballot or ``activemask`` is registered as ``TCallEffectKind::kOpaque``
because it is convergent and side-effecting, so ``LetInline`` has to leave the
binding in place: inlining it into a ``T.if_then_else`` arm or a store guard runs
the full-mask ``__shfl_*_sync`` with part of the warp inactive.

These checks run the pass on hand-built TIR, so they do not need a GPU.
"""

import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm import tirx


def _is_shuffle(node):
    return isinstance(node, tirx.Call) and node.op.name.startswith("tl.shfl_")


def _contains_shuffle(value):
    found = []
    tirx.stmt_functor.post_order_visit(
        tirx.Evaluate(value),
        lambda node: found.append(node) if _is_shuffle(node) else None,
    )
    return bool(found)


def _shuffle_stats(func):
    bindings = set()
    calls = []

    def visit(node):
        if _is_shuffle(node):
            calls.append(node)
        elif isinstance(node, tirx.Bind) and _contains_shuffle(node.value):
            bindings.add(node.var.name)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return bindings, len(calls)


def _kernel(variant, n=64):
    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            lane = tx % 32
            for it in T.serial((n + 31) // 32):
                idx = it * 32 + tx
                if variant == "one_arm":
                    x = X[idx]
                    t = T.shfl_up(x, 1, 2)
                    Y[idx] = T.if_then_else(lane % 2 >= 1, x + t, x)
                elif variant == "both_arms":
                    x = X[idx]
                    s = T.shfl_xor(x, 2, 16)
                    t = T.shfl_up(s, 1, 2)
                    Y[idx] = T.if_then_else(lane % 2 >= 1, s + t, s)
                else:
                    x = T.if_then_else(idx < n, X[idx], T.float32(0))
                    s = T.shfl_xor(x, 1, 32)
                    t = T.shfl_xor(s, 1, 2)
                    if idx < n:
                        Y[idx] = t

    return main


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("one_arm", {"t"}),
        ("both_arms", {"s", "t"}),
        ("tail_guard", {"s", "t"}),
    ],
)
def test_warp_collective_binding_survives_let_inline(variant, expected):
    func = _kernel(variant)
    before, before_calls = _shuffle_stats(func)
    assert before == expected
    assert before_calls == len(expected)

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    after = tl.transform.LetInline()(mod)["main"]

    bindings, calls = _shuffle_stats(after)
    assert bindings == expected
    # Every shuffle still executes exactly once, at its binding site, so none
    # was copied into a conditional arm or a store guard.
    assert calls == len(expected)


def test_launch_thread_idx_binding_still_inlines():
    @T.prim_func
    def main(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            A[tx] = T.float32(1.0)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    after = tl.transform.LetInline()(mod)["main"]

    binds = []
    tirx.stmt_functor.post_order_visit(
        after.body,
        lambda node: binds.append(node) if isinstance(node, tirx.Bind) else None,
    )
    assert not any(bind.var.name == "tx" for bind in binds)


@tilelang.testing.requires_cuda
def test_force_let_inline_keeps_warp_shuffle_out_of_branch():
    import torch

    @tl.jit(pass_configs={"tl.force_let_inline": True})
    def kernel():
        @T.prim_func
        def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32")):
            with T.Kernel(1, threads=32):
                tx = T.get_thread_binding(0)
                lane = tx % 32
                for it in T.serial(2):
                    idx = it * 32 + tx
                    x = X[idx]
                    t = T.shfl_up(x, 1, 2)
                    Y[idx] = T.if_then_else(lane % 2 >= 1, x + t, x)

        return main

    x = torch.randint(1, 8, (64,), device="cuda", dtype=torch.int32).float()
    y = torch.zeros_like(x)
    kernel()(x, y)
    torch.cuda.synchronize()

    lanes = torch.arange(32)
    g = x.cpu().view(-1, 32)
    up = g[:, torch.where(lanes % 2 >= 1, lanes - 1, lanes)]
    expected = torch.where(lanes % 2 >= 1, g + up, g).reshape(-1)
    assert torch.equal(y.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
