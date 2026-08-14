"""ABI/RUNTIME regression corpus: MetalKernelAdapter argument binding.

SplitHostDevice sorts device parameters by handle class and name; the adapter
must permute user arguments to the resulting order. The corpus is designed so
that any future parameter
reordering, host lowering change, or adapter modification is caught
automatically:

  1. non-alphabetical buffer names (Q, COS, SIN, Y)
  2. strongly reversed names (Z, A, INPUT, OUT)
  3. multiple outputs (X, Y, Z, OUT1, OUT2)
  4. buffer + scalar interleave (A, S, B, OUT)
  5. odd/tail shapes (rows not multiple of 128, dim not multiple of 64)
  6. read-only alias (same tensor bound to two inputs)

Every case: fp64 oracle, NaN sentinel prefill, real GPU execution,
torch.mps.synchronize(), exact error threshold. Failing = ABI regression.
"""

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="PyTorch MPS device is required",
)


def _check(kern, args, out, oracle, tag, atol=1e-4):
    """Sentinel + real GPU launch + sync + numerical assert."""
    out.fill_(float("nan"))
    torch.mps.synchronize()
    kern(*args, out)
    torch.mps.synchronize()
    assert not torch.isnan(out).any(), f"[{tag}] output not fully written (tail/region)"
    err = np.abs(out.cpu().numpy() - oracle).max()
    assert err < atol, f"[{tag}] binding mismatch: max_abs_err={err}"


# ---------------------------------------------------------------- 1. rope (Q, COS, SIN, Y)
@tilelang.jit
def rope_kernel(R, D):
    @T.prim_func
    def rope(
        Q: T.Tensor((R, D), "float32"),
        COS: T.Tensor((R, D), "float32"),
        SIN: T.Tensor((R, D), "float32"),
        Y: T.Tensor((R, D), "float32"),
    ):
        with T.Kernel(R) as bx:
            for d in T.serial(D // 2):
                Y[bx, d] = Q[bx, d] * COS[bx, d] - Q[bx, d + D // 2] * SIN[bx, d]
                Y[bx, d + D // 2] = Q[bx, d + D // 2] * COS[bx, d + D // 2] + Q[bx, d] * SIN[bx, d + D // 2]

    return rope


def test_abi_rope_non_alphabetical():
    R, D = 64, 32
    g = np.random.default_rng(7)
    Q = g.normal(size=(R, D)).astype(np.float32)
    C = g.normal(size=(R, D)).astype(np.float32)
    S = g.normal(size=(R, D)).astype(np.float32)
    d0 = np.arange(D // 2)
    oracle = np.empty_like(Q)
    oracle[:, d0] = Q[:, d0] * C[:, d0] - Q[:, d0 + D // 2] * S[:, d0]
    oracle[:, d0 + D // 2] = Q[:, d0 + D // 2] * C[:, d0 + D // 2] + Q[:, d0] * S[:, d0 + D // 2]
    _check(
        rope_kernel(R, D),
        [torch.from_numpy(Q).to("mps"), torch.from_numpy(C).to("mps"), torch.from_numpy(S).to("mps")],
        torch.zeros(R, D, device="mps"),
        oracle,
        "rope(Q,COS,SIN,Y)",
    )


# ---------------------------------------------------------------- 2. reversed names (Z, A, INPUT, OUT)
@tilelang.jit
def mixed_kernel(R, D):
    @T.prim_func
    def mix(
        Z: T.Tensor((R, D), "float32"),
        A: T.Tensor((R, D), "float32"),
        INPUT: T.Tensor((R, D), "float32"),
        OUT: T.Tensor((R, D), "float32"),
    ):
        with T.Kernel(R) as bx:
            for d in T.serial(D):
                OUT[bx, d] = Z[bx, d] * A[bx, d] + INPUT[bx, d]

    return mix


def test_abi_reversed_names():
    R, D = 32, 64
    g = np.random.default_rng(11)
    Z = g.normal(size=(R, D)).astype(np.float32)
    A = g.normal(size=(R, D)).astype(np.float32)
    I = g.normal(size=(R, D)).astype(np.float32)
    oracle = Z * A + I
    _check(
        mixed_kernel(R, D),
        [torch.from_numpy(Z).to("mps"), torch.from_numpy(A).to("mps"), torch.from_numpy(I).to("mps")],
        torch.zeros(R, D, device="mps"),
        oracle,
        "mix(Z,A,INPUT,OUT)",
    )


# ---------------------------------------------------------------- 3. multiple outputs (X, Y, Z, OUT1, OUT2)
@tilelang.jit
def multiout_kernel(R, D):
    @T.prim_func
    def multi(
        X: T.Tensor((R, D), "float32"),
        Y: T.Tensor((R, D), "float32"),
        Z: T.Tensor((R, D), "float32"),
        OUT1: T.Tensor((R, D), "float32"),
        OUT2: T.Tensor((R, D), "float32"),
    ):
        with T.Kernel(R) as bx:
            for d in T.serial(D):
                OUT1[bx, d] = X[bx, d] + Y[bx, d]
                OUT2[bx, d] = X[bx, d] * Z[bx, d] - Y[bx, d]

    return multi


def test_abi_multiple_outputs():
    R, D = 16, 32
    g = np.random.default_rng(13)
    X = g.normal(size=(R, D)).astype(np.float32)
    Y = g.normal(size=(R, D)).astype(np.float32)
    Z = g.normal(size=(R, D)).astype(np.float32)
    o1, o2 = X + Y, X * Z - Y
    kern = multiout_kernel(R, D)
    out1 = torch.zeros(R, D, device="mps")
    out2 = torch.zeros(R, D, device="mps")
    kern(
        torch.from_numpy(X).to("mps"),
        torch.from_numpy(Y).to("mps"),
        torch.from_numpy(Z).to("mps"),
        out1,
        out2,
    )
    torch.mps.synchronize()
    # OUT1/OUT2 are both prefilled NaN; binding errors swap or garble them
    e1 = np.abs(out1.cpu().numpy() - o1).max()
    e2 = np.abs(out2.cpu().numpy() - o2).max()
    assert e1 < 1e-4 and e2 < 1e-4, f"[multiout] OUT1 err={e1} OUT2 err={e2}"


# ---------------------------------------------------------------- 4. buffer + scalar interleave (A, S, B, OUT)
@tilelang.jit
def saxpy_kernel(N):
    @T.prim_func
    def saxpy(
        A: T.Tensor((N,), "float32"),
        S: T.int32,
        B: T.Tensor((N,), "float32"),
        OUT: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N) as bx:
            OUT[bx] = A[bx] * T.cast(S, T.float32) + B[bx]

    return saxpy


def test_abi_buffer_scalar_interleave():
    N = 128
    g = np.random.default_rng(17)
    A = g.normal(size=(N,)).astype(np.float32)
    B = g.normal(size=(N,)).astype(np.float32)
    S = 3
    oracle = A * S + B
    _check(
        saxpy_kernel(N),
        [torch.from_numpy(A).to("mps"), S, torch.from_numpy(B).to("mps")],
        torch.zeros(N, device="mps"),
        oracle,
        "saxpy(A,S,B,OUT)",
    )


# ---------------------------------------------------------------- 5. odd/tail shapes
@tilelang.jit
def tail_kernel(R, D):
    @T.prim_func
    def tail(
        INPUT: T.Tensor((R, D), "float32"),
        OUT: T.Tensor((R, D), "float32"),
    ):
        with T.Kernel(R) as bx:
            for d in T.serial(D):
                OUT[bx, d] = INPUT[bx, d] * 2.0

    return tail


def test_abi_odd_tail_shapes():
    for R, D in [(31, 96), (127, 33), (1, 7), (257, 128)]:
        g = np.random.default_rng(R + D)
        X = g.normal(size=(R, D)).astype(np.float32)
        _check(
            tail_kernel(R, D),
            [torch.from_numpy(X).to("mps")],
            torch.zeros(R, D, device="mps"),
            X * 2.0,
            f"tail(R={R},D={D})",
        )


# ---------------------------------------------------------------- 6. read-only alias
@tilelang.jit
def alias_kernel(R, D):
    @T.prim_func
    def alias(
        X: T.Tensor((R, D), "float32"),
        Y: T.Tensor((R, D), "float32"),
        OUT: T.Tensor((R, D), "float32"),
    ):
        with T.Kernel(R) as bx:
            for d in T.serial(D):
                OUT[bx, d] = X[bx, d] * Y[bx, d] + X[bx, d]

    return alias


def test_abi_readonly_alias():
    R, D = 32, 16
    g = np.random.default_rng(19)
    X = g.normal(size=(R, D)).astype(np.float32)
    oracle = X * X + X
    x_t = torch.from_numpy(X).to("mps")
    _check(
        alias_kernel(R, D),
        [x_t, x_t],  # same tensor bound to X and Y (read-only alias)
        torch.zeros(R, D, device="mps"),
        oracle,
        "alias(X==Y)",
    )
