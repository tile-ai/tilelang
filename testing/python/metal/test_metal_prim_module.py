"""Programmatic multi-kernel programs on the torch Metal backend."""

import pytest
import torch

import tilelang
import tilelang.language as T

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="PyTorch MPS device is required")


@T.macro
def _scale(A, B, n, factor):
    for i in T.Parallel(n):
        B[i] = A[i] * factor


def test_private_schedule_launches_at_every_call_site():
    n = 128
    parameters = (
        ("A", T.Tensor((n,), T.float32)),
        ("scratch", T.Tensor((n,), T.float32)),
        ("B", T.Tensor((n,), T.float32)),
    )

    def schedule_body(source, target):
        with T.Kernel(1, threads=n):
            _scale(source, target, n, 2.0)

    schedule = T.PrimFuncDefinition(
        "double",
        (("source", T.Tensor((n,), T.float32)), ("target", T.Tensor((n,), T.float32))),
        schedule_body,
    )

    def body(private, A, scratch, B):
        private["double"](A, scratch)
        private["double"](scratch, B)

    module = T.build_prim_module("main", parameters, body, (schedule,))
    kernel = tilelang.compile(module, target="metal", execution_backend="torch")

    a = torch.arange(n, dtype=torch.float32, device="mps")
    scratch = torch.empty_like(a)
    b = torch.empty_like(a)
    kernel(a, scratch, b)
    torch.testing.assert_close(b.cpu(), a.cpu() * 4.0)
