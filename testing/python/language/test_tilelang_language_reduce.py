import tilelang
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import pytest
import torch

tilelang.testing.set_random_seed()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(M, N, dtype):
    torch_dtype = getattr(torch, dtype)
    if torch_dtype in (torch.int32, torch.int64):
        return torch.randint(-100, 100, (M, N), dtype=torch_dtype).cuda()
    return torch.randn(M, N, dtype=torch_dtype).cuda()


def _ref(A, op):
    if op == "sum":
        return A.sum(dim=1).to(A.dtype)
    if op == "max":
        return A.max(dim=1).values
    if op == "min":
        return A.min(dim=1).values
    if op == "abssum":
        return A.abs().sum(dim=1).to(A.dtype)
    if op == "absmax":
        return A.abs().max(dim=1).values
    raise ValueError(op)


def _reduce_op(T, op, src, dst, dim, batch=1):
    kwargs = {} if batch == 1 else {"batch": batch}
    if op == "sum":
        T.reduce_sum(src, dst, dim=dim, **kwargs)
    elif op == "max":
        T.reduce_max(src, dst, dim=dim, **kwargs)
    elif op == "min":
        T.reduce_min(src, dst, dim=dim, **kwargs)
    elif op == "abssum":
        T.reduce_abssum(src, dst, dim=dim, **kwargs)
    elif op == "absmax":
        T.reduce_absmax(src, dst, dim=dim, **kwargs)


# ---------------------------------------------------------------------------
# test_reduce  (op × dtype × src_scope × dst_scope × threads × batch)
# ---------------------------------------------------------------------------

# int types only support sum (no abssum/absmax for int in tilelang)
REDUCE_CASES = [
    # (op,      dtype,       M,   N,   src_scope,    dst_scope,  threads, batch)
    ("sum", T.float32, 128, 128, "fragment", "fragment", 32, 1),
    ("sum", T.int32, 128, 128, "fragment", "fragment", 32, 1),
    ("sum", T.int64, 192, 64, "fragment", "fragment", 32, 1),
    ("sum", T.float32, 192, 64, "fragment", "fragment", 32, 1),
    ("sum", T.float32, 32, 32, "fragment", "fragment", 16, 1),
    ("sum", T.float32, 16, 16, "fragment", "fragment", 8, 1),
    ("sum", T.float32, 32, 32, "shared", "shared", 32, 1),
    ("sum", T.float32, 32, 32, "fragment", "shared", 32, 1),
    ("max", T.float32, 128, 128, "fragment", "fragment", 32, 1),
    ("max", T.int64, 128, 128, "fragment", "fragment", 32, 1),
    ("max", T.float32, 32, 32, "shared", "shared", 32, 1),
    ("min", T.float32, 128, 128, "fragment", "fragment", 32, 1),
    ("min", T.int64, 128, 128, "fragment", "fragment", 32, 1),
    ("abssum", T.float32, 128, 128, "fragment", "fragment", 32, 1),
    ("abssum", T.int64, 128, 128, "fragment", "fragment", 32, 1),
    ("absmax", T.float32, 128, 128, "fragment", "fragment", 32, 1),
    ("absmax", T.int64, 128, 128, "fragment", "fragment", 32, 1),
    # batch > 1: verify run_batch codegen and correctness together
    ("sum", T.float32, 128, 64, "shared", "fragment", 256, 2),
    ("sum", T.float32, 128, 64, "shared", "fragment", 256, 4),
    ("sum", T.float16, 64, 128, "fragment", "fragment", 256, 4),
    ("max", T.bfloat16, 128, 64, "shared", "fragment", 256, 2),
    ("max", T.float32, 128, 128, "fragment", "fragment", 256, 4),
    ("min", T.float32, 64, 128, "shared", "fragment", 128, 2),
    ("min", T.float16, 128, 128, "fragment", "fragment", 256, 8),
    ("abssum", T.float32, 128, 128, "fragment", "fragment", 256, 4),
    ("absmax", T.float32, 128, 128, "fragment", "fragment", 256, 4),
]


@pytest.mark.parametrize(
    ("op", "dtype", "M", "N", "src_scope", "dst_scope", "threads", "batch"),
    REDUCE_CASES,
    ids=[
        f"{op}-{dtype}-{M}x{N}-{src_scope[0]}2{dst_scope[0]}-t{threads}-b{batch}"
        for op, dtype, M, N, src_scope, dst_scope, threads, batch in REDUCE_CASES
    ],
)
def test_reduce(op, dtype, M, N, src_scope, dst_scope, threads, batch):
    import re

    @tilelang.jit(out_idx=-1)
    def kernel(M, N, dtype, op, src_scope, dst_scope, threads, batch):
        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M,), dtype)):
            with T.Kernel(1, threads=threads):
                if src_scope == "fragment":
                    src = T.alloc_fragment((M, N), dtype)
                else:
                    src = T.alloc_shared((M, N), dtype)
                if dst_scope == "fragment":
                    dst = T.alloc_fragment((M,), dtype)
                else:
                    dst = T.alloc_shared((M,), dtype)
                T.copy(A, src, disable_tma=src_scope == "shared")
                _reduce_op(T, op, src, dst, dim=1, batch=batch)
                T.copy(dst, B)

        return main

    jit_kernel = kernel(M, N, dtype, op, src_scope, dst_scope, threads, batch)

    if batch > 1:
        src = jit_kernel.get_kernel_source()
        m = re.search(r",\s*(\d+)\s*,\s*\d+\s*>::run_batch\(", src)
        assert m is not None, f"Expected run_batch in generated source.\n{src}"
        assert int(m.group(1)) > 1, f"Expected batch_size > 1, got {m.group(1)}.\n{src}"

    A = _make_input(M, N, dtype)
    B = jit_kernel(A)
    # float16/bfloat16 accumulate more rounding error over large reductions
    tol = 1e-1 if dtype in (T.float16, T.bfloat16) else 1e-2
    torch.testing.assert_close(B, _ref(A, op), atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# test_reduce_clear  (op × src_scope × dst_scope, clear=False)
# ---------------------------------------------------------------------------

REDUCE_CLEAR_CASES = [
    # (op,   dtype,       M,   N,  src_scope,   dst_scope,  init,   ref_fn_extra)
    # sum: init=1, ref = A.sum(dim=1) + 1
    ("sum", T.float32, 128, 128, "fragment", "fragment"),
    ("sum", T.float32, 128, 128, "fragment", "shared"),
    ("sum", T.float32, 32, 32, "shared", "shared"),
    # max: init=-inf, ref = A.max(dim=1).values  (max(-inf, x) = x)
    ("max", T.float16, 128, 128, "fragment", "fragment"),
]


@pytest.mark.parametrize(
    ("op", "dtype", "M", "N", "src_scope", "dst_scope"),
    REDUCE_CLEAR_CASES,
    ids=[f"{op}-{dtype}-{M}x{N}-{src_scope[0]}2{dst_scope[0]}" for op, dtype, M, N, src_scope, dst_scope in REDUCE_CLEAR_CASES],
)
def test_reduce_clear(op, dtype, M, N, src_scope, dst_scope):
    @tilelang.jit(out_idx=-1)
    def kernel(M, N, dtype, op, src_scope, dst_scope):
        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M,), dtype)):
            with T.Kernel(1, threads=32):
                if src_scope == "fragment":
                    src = T.alloc_fragment((M, N), dtype)
                else:
                    src = T.alloc_shared((M, N), dtype)
                if dst_scope == "fragment":
                    dst = T.alloc_fragment((M,), dtype)
                else:
                    dst = T.alloc_shared((M,), dtype)
                T.copy(A, src, disable_tma=src_scope == "shared")
                if op == "sum":
                    T.fill(dst, 1)
                    T.reduce_sum(src, dst, dim=1, clear=False)
                elif op == "max":
                    T.fill(dst, -T.infinity(dtype))
                    T.reduce_max(src, dst, dim=1, clear=False)
                T.copy(dst, B)

        return main

    torch_dtype = getattr(torch, dtype)
    A = torch.randn(M, N, dtype=torch_dtype).cuda()
    B = kernel(M, N, dtype, op, src_scope, dst_scope)(A)
    if op == "sum":
        ref = A.sum(dim=1) + 1
    elif op == "max":
        ref = A.max(dim=1).values
    torch.testing.assert_close(B, ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# T.finalize_reducer tests
# ---------------------------------------------------------------------------

_COMPILE_FLAGS = {
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FINALIZE_REDUCER_CASES = [
    # (op,   dtype,      block_M, block_N, batch)
    ("sum", T.float32, 128, 64, 1),
    ("sum", T.float32, 128, 64, 4),
    ("max", T.float16, 64, 128, 1),
    ("max", T.float16, 64, 128, 8),
    ("min", T.float32, 128, 128, 1),
    ("min", T.float32, 128, 128, 16),
]


def _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch):
    @T.prim_func
    def kernel(A: T.Tensor((block_M, block_N), dtype), B: T.Tensor((block_M,), dtype)):
        with T.Kernel(1, threads=256):
            o_reducer = T.alloc_reducer(block_M, dtype, op=op, replication="all")
            T.clear(o_reducer)
            A_smem = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A, A_smem)
            A_frag = T.alloc_fragment((block_M, block_N), dtype)
            T.copy(A_smem, A_frag)
            for i, j in T.Parallel(block_M, block_N):
                if op == "sum":
                    o_reducer[i] += A_frag[i, j]
                elif op == "max":
                    o_reducer[i] = T.max(o_reducer[i], A_frag[i, j])
                else:
                    o_reducer[i] = T.min(o_reducer[i], A_frag[i, j])
            T.finalize_reducer(o_reducer, batch=batch)
            T.copy(o_reducer, B)

    return kernel


@pytest.mark.parametrize(
    ("op", "dtype", "block_M", "block_N", "batch"),
    FINALIZE_REDUCER_CASES,
    ids=[f"{op}-{dtype}-{bM}x{bN}-b{batch}" for op, dtype, bM, bN, batch in FINALIZE_REDUCER_CASES],
)
def test_finalize_reducer_codegen(op, dtype, block_M, block_N, batch):
    """batch=1 → scalar run; batch>1 → run_batch with correct template arg."""
    import re

    src = tl.compile(
        _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch),
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()

    if batch == 1:
        assert "run_batch" not in src, f"batch=1 must not emit run_batch.\n{src}"
    else:
        m = re.search(r",\s*(\d+)\s*,\s*\d+\s*>::run_batch\(", src)
        assert m is not None, f"Expected run_batch in generated source.\n{src}"
        assert int(m.group(1)) == batch, f"Expected batch={batch}, got {m.group(1)}.\n{src}"


@pytest.mark.parametrize(
    ("op", "dtype", "block_M", "block_N", "batch"),
    [c for c in FINALIZE_REDUCER_CASES if c[4] == 1],
    ids=[f"{op}-{dtype}-{bM}x{bN}" for op, dtype, bM, bN, batch in FINALIZE_REDUCER_CASES if batch == 1],
)
def test_finalize_reducer_correctness(op, dtype, block_M, block_N, batch):
    """Numerical correctness (batch=1 scalar path; batch>1 blocked by fragment layout bug)."""
    A = torch.randn(block_M, block_N, dtype=getattr(torch, dtype)).cuda()
    B = tl.compile(
        _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch),
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, _ref(A, op), atol=1e-2, rtol=1e-2)


# (batch, exc_type, match)
FINALIZE_REDUCER_INVALID_CASES = [
    (0, ValueError, "batch must be >= 1"),
    (-1, ValueError, "batch must be >= 1"),
    (128, Exception, "exceeds total output elements"),  # block_M=64, batch=128
    (3, Exception, "must evenly divide"),  # block_M=64, batch=3
]


@pytest.mark.parametrize(
    ("batch", "exc_type", "match"),
    FINALIZE_REDUCER_INVALID_CASES,
    ids=["zero", "negative", "exceeds", "not-divisible"],
)
def test_finalize_reducer_invalid_batch(batch, exc_type, match):
    block_M = 64

    def make_kernel():
        @T.prim_func
        def kernel(A: T.Tensor((block_M, 64), T.float32), B: T.Tensor((block_M,), T.float32)):
            with T.Kernel(1, threads=256):
                o_reducer = T.alloc_reducer(block_M, T.float32, op="sum", replication="all")
                T.clear(o_reducer)
                A_smem = T.alloc_shared((block_M, 64), T.float32)
                T.copy(A, A_smem)
                A_frag = T.alloc_fragment((block_M, 64), T.float32)
                T.copy(A_smem, A_frag)
                for i, j in T.Parallel(block_M, 64):
                    o_reducer[i] += A_frag[i, j]
                T.finalize_reducer(o_reducer, batch=batch)
                T.copy(o_reducer, B)

        return kernel

    with pytest.raises(exc_type, match=match):
        # batch<1 raises at prim_func definition time; others at compile time
        k = make_kernel()
        tl.compile(k, out_idx=-1, pass_configs=_COMPILE_FLAGS)


if __name__ == "__main__":
    tilelang.testing.main()
