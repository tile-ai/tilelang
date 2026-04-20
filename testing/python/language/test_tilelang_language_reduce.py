from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T
import pytest

tilelang.testing.set_random_seed()

REDUCE_SUM_CASES = [
    (T.float32, 128, 128),
    (T.int32, 128, 128),
    (T.int64, 128, 128),
    (T.float32, 192, 64),
    (T.int32, 192, 64),
    (T.int64, 192, 64),
]
REDUCE_OTHER_OP_CASES = [
    ("max", T.float32),
    ("max", T.int64),
    ("min", T.float32),
    ("min", T.int64),
    ("abssum", T.float32),
    ("abssum", T.int64),
    ("absmax", T.float32),
    ("absmax", T.int64),
]


def _make_shared_reduce(M, N, dtype, reduce_cb):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)

            T.copy(A, A_shared)
            reduce_cb(T, A_shared, B_shared)
            T.copy(B_shared, B)

    return main


def _run_program(program, ref_program, atol=1e-2, rtol=1e-2):
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()
    profiler.assert_allclose(ref_program, atol=atol, rtol=rtol)


def reduce_test(M, N, dtype=T.float16, op="sum", threads=32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            if op == "sum":
                T.reduce_sum(A_local, B_local, dim=1)
            elif op == "max":
                T.reduce_max(A_local, B_local, dim=1)
            elif op == "min":
                T.reduce_min(A_local, B_local, dim=1)
            elif op == "abssum":
                T.reduce_abssum(A_local, B_local, dim=1)
            elif op == "absmax":
                T.reduce_absmax(A_local, B_local, dim=1)
            elif op == "bitand":
                T.reduce_bitand(A_local, B_local, dim=1)
            elif op == "bitor":
                T.reduce_bitor(A_local, B_local, dim=1)
            elif op == "bitxor":
                T.reduce_bitxor(A_local, B_local, dim=1)
            T.copy(B_local, B)

    return main


def reduce_sum_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_sum(src, dst, dim=1))


def reduce_max_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_max(src, dst, dim=1))


def reduce_min_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_min(src, dst, dim=1))


def reduce_abssum_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_abssum(src, dst, dim=1))


def reduce_absmax_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_absmax(src, dst, dim=1))


def run_reduce(M, N, dtype=T.float32, op="sum", mode="rr", threads=32):
    if mode == "rr":
        program = reduce_test(M, N, dtype, op, threads)
    elif mode == "ss":
        assert op == "sum", f"shared reduce only supports sum, got {op}"
        program = reduce_sum_ss(M, N, dtype)
    else:
        raise NotImplementedError(f"run_reduce only supports rr and ss, got {mode}")

    import torch

    def ref_fn(A):
        if op == "sum":
            res = A.sum(dim=1)
        elif op == "max":
            res = A.max(dim=1).values
        elif op == "min":
            res = A.min(dim=1).values
        elif op == "abssum":
            res = A.abs().sum(dim=1)
        elif op == "absmax":
            res = A.abs().max(dim=1).values
        if A.dtype in [torch.uint32, torch.int32, torch.int64]:
            return res.to(A.dtype)
        return res

    _run_program(program, ref_fn)


def run_shared_reduce(program_builder, ref_program, M, N, dtype=T.float32):
    program = program_builder(M, N, dtype)
    _run_program(program, ref_program)


def run_reduce_max(M, N, dtype=T.float16):
    program = reduce_test(M, N, dtype, "max")
    _run_program(program, lambda A: A.max(dim=1).values, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("dtype", "M", "N"),
    REDUCE_SUM_CASES,
    ids=[f"{dtype}-{M}x{N}" for dtype, M, N in REDUCE_SUM_CASES],
)
def test_reduce_sum(dtype, M, N):
    run_reduce(M, N, dtype, "sum")


@pytest.mark.parametrize(
    ("op", "dtype"),
    REDUCE_OTHER_OP_CASES,
    ids=[f"{op}-{dtype}" for op, dtype in REDUCE_OTHER_OP_CASES],
)
def test_reduce_other_op(op, dtype):
    run_reduce(128, 128, dtype, op)


def test_reduce_sum_threads():
    run_reduce(32, 32, T.float32, "sum", mode="rr", threads=16)
    run_reduce(16, 16, T.float32, "sum", mode="rr", threads=8)


def test_reduce_sum_shared():
    run_reduce(32, 32, op="sum", mode="ss")


def test_reduce_max():
    run_reduce_max(128, 128, T.float16)
    run_reduce_max(192, 64, T.float32)


def test_reduce_max_shared():
    run_shared_reduce(reduce_max_ss, lambda A: A.max(dim=1).values, 32, 32, T.float32)


def test_reduce_min_shared():
    run_shared_reduce(reduce_min_ss, lambda A: A.min(dim=1).values, 32, 32, T.float32)


def test_reduce_abssum_shared():
    run_shared_reduce(reduce_abssum_ss, lambda A: A.abs().sum(dim=1), 32, 32, T.float32)


def test_reduce_absmax_shared():
    run_shared_reduce(reduce_absmax_ss, lambda A: A.abs().max(dim=1).values, 32, 32, T.float32)


def reduce_sum_test_clear(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.fill(B_local, 1)
            T.reduce_sum(A_local, B_local, dim=1, clear=False)
            T.copy(B_local, B)

    return main


def run_reduce_sum_clear(M, N, dtype=T.float32, tl_func=reduce_sum_test_clear):
    program = tl_func(M, N, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)

    def ref_program(A):
        return A.sum(dim=1) + 1

    import torch

    dummy_A = torch.randn((M, N), dtype=getattr(torch, dtype)).cuda()
    ref_out = ref_program(dummy_A)
    tl_out = jit_kernel(dummy_A)
    torch.testing.assert_close(tl_out, ref_out, atol=1e-2, rtol=1e-2)


def test_reduce_sum_clear():
    run_reduce_sum_clear(128, 128, T.float32)
    run_reduce_sum_clear(192, 64, T.float32)


def reduce_max_test_clear(M, N, dtype=T.float16):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.fill(B_local, -T.infinity(dtype))
            T.reduce_max(A_local, B_local, dim=1, clear=False)
            T.copy(B_local, B)

    return main


def run_reduce_max_clear(M, N, dtype=T.float16):
    program = reduce_max_test_clear(M, N, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)

    def ref_program(A):
        return A.max(dim=1).values

    import torch

    dummy_A = torch.randn((M, N), dtype=getattr(torch, dtype)).cuda()
    ref_out = ref_program(dummy_A)
    tl_out = jit_kernel(dummy_A)
    torch.testing.assert_close(tl_out, ref_out, atol=1e-2, rtol=1e-2)


def test_reduce_max_clear():
    run_reduce_max_clear(128, 128, T.float16)


def reduce_sum_test_clear_B_shared(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)

            T.copy(A, A_local)
            T.fill(B_shared, 1)
            T.reduce_sum(A_local, B_shared, dim=1, clear=False)
            T.copy(B_shared, B)

    return main


def test_reduce_sum_clear_B_shared():
    run_reduce_sum_clear(128, 128, T.float32, reduce_sum_test_clear_B_shared)


def reduce_sum_test_clear_AB_shared(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)

            T.copy(A, A_shared, disable_tma=True)
            T.fill(B_shared, 1)
            T.reduce_sum(A_shared, B_shared, dim=1, clear=False)
            T.copy(B_shared, B)

    return main


def test_reduce_sum_clear_AB_shared():
    run_reduce_sum_clear(32, 32, T.float32, reduce_sum_test_clear_AB_shared)


BATCHED_REDUCE_CASES = [
    ("max", T.bfloat16, 128, 64, 256),
    ("sum", T.float32, 128, 64, 256),
    ("min", T.float32, 64, 128, 128),
]


@pytest.mark.parametrize(
    ("op", "dtype", "M", "N", "threads"),
    BATCHED_REDUCE_CASES,
    ids=[f"{op}-{dtype}-{M}x{N}-t{threads}" for op, dtype, M, N, threads in BATCHED_REDUCE_CASES],
)
def test_batched_allreduce_codegen(op, dtype, M, N, threads):
    """Verify that the batched AllReduce path (run_batch) is emitted when
    the user explicitly passes batch > 1 to the reduce call."""
    import re

    reduce_fn = {
        "sum": T.reduce_sum,
        "max": T.reduce_max,
        "min": T.reduce_min,
    }[op]

    # Use batch=2 — the test cases are chosen so that N_per_thread is even
    # and >= 2, so batch=2 is always valid.
    batch_val = 2

    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared((M, N), dtype)
            frag = T.alloc_fragment((N,), dtype)
            T.copy(A, A_shared)
            reduce_fn(A_shared, frag, dim=0, batch=batch_val)
            T.copy(frag, B)

    mod = tl.compile(
        kernel,
        out_idx=-1,
        pass_configs={
            tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
    )
    src = mod.get_kernel_source()

    # The batched path emits run_batch with batch_size and workspace_stride
    # as the last two template arguments before >::run_batch.
    pattern = r",\s*(\d+)\s*,\s*(\d+)\s*>::run_batch\("
    match = re.search(pattern, src)
    assert match is not None, f"Expected batched AllReduce (run_batch) in generated source, but not found.\nGenerated source:\n{src}"
    batch_size = int(match.group(1))
    assert batch_size > 1, f"Expected batch_size > 1, got {batch_size}.\nGenerated source:\n{src}"


# ---------------------------------------------------------------------------
# Helpers shared by all finalize_reducer tests
# ---------------------------------------------------------------------------

_COMPILE_FLAGS = {
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


def _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch):
    """Return a prim_func that reduces A[block_M, block_N] -> B[block_M]
    using alloc_reducer + finalize_reducer(batch=batch)."""

    @T.prim_func
    def kernel(A: T.Tensor((block_M, block_N), dtype), B: T.Tensor((block_M,), dtype)):
        with T.Kernel(1, threads=256) as _:
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


# ---------------------------------------------------------------------------
# Codegen tests – verify run_batch / run is emitted as expected
# ---------------------------------------------------------------------------

BATCHED_FINALIZE_REDUCER_CASES = [
    # (op,   dtype,        block_M, block_N, threads, batch)
    ("sum", T.float32, 128, 64, 256, 4),
    ("max", T.bfloat16, 64, 128, 256, 8),
    ("min", T.float32, 128, 128, 256, 16),
]


@pytest.mark.parametrize(
    ("op", "dtype", "block_M", "block_N", "threads", "batch"),
    BATCHED_FINALIZE_REDUCER_CASES,
    ids=[f"{op}-{dtype}-{bM}x{bN}-t{threads}-b{batch}" for op, dtype, bM, bN, threads, batch in BATCHED_FINALIZE_REDUCER_CASES],
)
def test_batched_finalize_reducer_codegen(op, dtype, block_M, block_N, threads, batch):
    """Verify that the batched AllReduce path (run_batch) is emitted when
    an explicit batch argument is passed to T.finalize_reducer."""
    import re

    kernel = _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch)
    mod = tl.compile(kernel, out_idx=-1, pass_configs=_COMPILE_FLAGS)
    src = mod.get_kernel_source()

    # The batched path emits run_batch with batch_size and workspace_stride
    # as the last two template arguments before >::run_batch.
    pattern = r",\s*(\d+)\s*,\s*(\d+)\s*>::run_batch\("
    match = re.search(pattern, src)
    assert match is not None, f"Expected batched AllReduce (run_batch) in generated source, but not found.\nGenerated source:\n{src}"
    emitted_batch = int(match.group(1))
    assert emitted_batch == batch, f"Expected batch_size={batch}, got {emitted_batch}.\nGenerated source:\n{src}"


SCALAR_FINALIZE_REDUCER_CASES = [
    # batch=1 (default) must NOT emit run_batch
    ("sum", T.float32, 64, 64, 256, 1),
    ("max", T.float32, 128, 32, 256, 1),
]


@pytest.mark.parametrize(
    ("op", "dtype", "block_M", "block_N", "threads", "batch"),
    SCALAR_FINALIZE_REDUCER_CASES,
    ids=[f"{op}-{dtype}-{bM}x{bN}-t{threads}-b{batch}" for op, dtype, bM, bN, threads, batch in SCALAR_FINALIZE_REDUCER_CASES],
)
def test_scalar_finalize_reducer_codegen(op, dtype, block_M, block_N, threads, batch):
    """Verify that batch=1 (default) does NOT emit run_batch."""

    kernel = _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch)
    mod = tl.compile(kernel, out_idx=-1, pass_configs=_COMPILE_FLAGS)
    src = mod.get_kernel_source()

    assert "run_batch" not in src, f"batch=1 must not emit run_batch.\nGenerated source:\n{src}"


# ---------------------------------------------------------------------------
# Correctness tests – compare against numpy/torch reference
# ---------------------------------------------------------------------------

# batch=1 cases: scalar path, expected to be correct.
# Also includes batch==block_M: when batch equals the full output size, the
# batched AllReduce happens to be equivalent to the full scalar path because
# every element in the fragment buffer participates, so the result is correct.
FINALIZE_REDUCER_CORRECTNESS_SCALAR_CASES = [
    # (op,    dtype,       block_M, block_N, batch)
    ("sum", T.float32, 128, 64, 1),
    ("sum", T.float32, 128, 64, 128),  # batch == block_M: correct by coincidence
    ("max", T.float32, 64, 128, 1),
    ("min", T.float32, 128, 128, 1),
    ("sum", T.float16, 64, 64, 1),
]

# batch>1 but batch<block_M cases: known-buggy batched path – run_batch is
# called with the raw fragment buffer pointer, but the fragment layout is
# non-contiguous per thread, so run_batch reads wrong/zero elements.
# Tracked in work/bug-finalize-reducer-batch.md.
FINALIZE_REDUCER_CORRECTNESS_BATCHED_CASES = [
    # (op,    dtype,       block_M, block_N, batch)
    ("sum", T.float32, 128, 64, 4),
    ("max", T.float32, 64, 128, 8),
    ("min", T.float32, 128, 128, 16),
    ("sum", T.float16, 64, 64, 4),
]


def _run_finalize_reducer_correctness(op, dtype, block_M, block_N, batch):

    kernel = _make_finalize_reducer_kernel(block_M, block_N, dtype, op, batch)
    jit_kernel = tl.compile(kernel, out_idx=-1, pass_configs=_COMPILE_FLAGS)

    def ref_fn(A):
        if op == "sum":
            return A.sum(dim=1)
        elif op == "max":
            return A.max(dim=1).values
        else:
            return A.min(dim=1).values

    profiler = jit_kernel.get_profiler()
    profiler.assert_allclose(ref_fn, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("op", "dtype", "block_M", "block_N", "batch"),
    FINALIZE_REDUCER_CORRECTNESS_SCALAR_CASES,
    ids=[f"{op}-{dtype}-{bM}x{bN}-b{batch}" for op, dtype, bM, bN, batch in FINALIZE_REDUCER_CORRECTNESS_SCALAR_CASES],
)
def test_finalize_reducer_correctness_scalar(op, dtype, block_M, block_N, batch):
    """Correctness of finalize_reducer with batch=1 (scalar AllReduce::run path)."""
    _run_finalize_reducer_correctness(op, dtype, block_M, block_N, batch)


@pytest.mark.xfail(
    reason=(
        "finalize_reducer batched path passes buffer->data directly to run_batch, "
        "but fragment layout is non-contiguous per thread. "
        "See work/bug-finalize-reducer-batch.md for root-cause analysis."
    ),
    strict=True,
)
@pytest.mark.parametrize(
    ("op", "dtype", "block_M", "block_N", "batch"),
    FINALIZE_REDUCER_CORRECTNESS_BATCHED_CASES,
    ids=[f"{op}-{dtype}-{bM}x{bN}-b{batch}" for op, dtype, bM, bN, batch in FINALIZE_REDUCER_CORRECTNESS_BATCHED_CASES],
)
def test_finalize_reducer_correctness_batched(op, dtype, block_M, block_N, batch):
    """Correctness of finalize_reducer with batch>1 – currently xfail due to
    non-contiguous fragment layout bug in the batched AllReduce path."""
    _run_finalize_reducer_correctness(op, dtype, block_M, block_N, batch)


# ---------------------------------------------------------------------------
# Error-input tests
# ---------------------------------------------------------------------------


def test_finalize_reducer_batch_zero_raises():
    """batch=0 must raise ValueError at the Python layer."""
    with pytest.raises(ValueError, match="batch must be >= 1"):

        @T.prim_func
        def kernel(A: T.Tensor((64, 64), T.float32), B: T.Tensor((64,), T.float32)):
            with T.Kernel(1, threads=256) as _:
                o_reducer = T.alloc_reducer(64, T.float32, op="sum", replication="all")
                T.clear(o_reducer)
                T.finalize_reducer(o_reducer, batch=0)
                T.copy(o_reducer, B)


def test_finalize_reducer_batch_negative_raises():
    """Negative batch must raise ValueError at the Python layer."""
    with pytest.raises(ValueError, match="batch must be >= 1"):

        @T.prim_func
        def kernel(A: T.Tensor((64, 64), T.float32), B: T.Tensor((64,), T.float32)):
            with T.Kernel(1, threads=256) as _:
                o_reducer = T.alloc_reducer(64, T.float32, op="sum", replication="all")
                T.clear(o_reducer)
                T.finalize_reducer(o_reducer, batch=-1)
                T.copy(o_reducer, B)


def test_finalize_reducer_batch_exceeds_layout_raises():
    """batch > total output elements must raise at compile/lower time."""
    block_M = 64

    @T.prim_func
    def kernel(A: T.Tensor((block_M, 64), T.float32), B: T.Tensor((block_M,), T.float32)):
        with T.Kernel(1, threads=256) as _:
            o_reducer = T.alloc_reducer(block_M, T.float32, op="sum", replication="all")
            T.clear(o_reducer)
            A_smem = T.alloc_shared((block_M, 64), T.float32)
            T.copy(A, A_smem)
            A_frag = T.alloc_fragment((block_M, 64), T.float32)
            T.copy(A_smem, A_frag)
            for i, j in T.Parallel(block_M, 64):
                o_reducer[i] += A_frag[i, j]
            # batch=block_M*2 exceeds total output elements block_M
            T.finalize_reducer(o_reducer, batch=block_M * 2)
            T.copy(o_reducer, B)

    with pytest.raises(Exception, match="exceeds total output elements"):
        tl.compile(kernel, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_finalize_reducer_batch_not_divisible_raises():
    """batch that does not evenly divide total output elements must raise."""
    block_M = 64  # batch=3 does not divide 64

    @T.prim_func
    def kernel(A: T.Tensor((block_M, 64), T.float32), B: T.Tensor((block_M,), T.float32)):
        with T.Kernel(1, threads=256) as _:
            o_reducer = T.alloc_reducer(block_M, T.float32, op="sum", replication="all")
            T.clear(o_reducer)
            A_smem = T.alloc_shared((block_M, 64), T.float32)
            T.copy(A, A_smem)
            A_frag = T.alloc_fragment((block_M, 64), T.float32)
            T.copy(A_smem, A_frag)
            for i, j in T.Parallel(block_M, 64):
                o_reducer[i] += A_frag[i, j]
            T.finalize_reducer(o_reducer, batch=3)
            T.copy(o_reducer, B)

    with pytest.raises(Exception, match="must evenly divide"):
        tl.compile(kernel, out_idx=-1, pass_configs=_COMPILE_FLAGS)


if __name__ == "__main__":
    tilelang.testing.main()
