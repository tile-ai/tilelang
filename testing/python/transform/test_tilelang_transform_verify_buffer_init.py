import pytest
import torch

import tilelang
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

BC, BK, NB = 16, 64, 16
MESSAGE = "Buffer read before initialization"
GEMM_HINT = "clear_accum=True"


def _gemm_kernel(init_mode):
    """Build the #2936 kernel with one accumulator-initialization variant.

    Parameters
    ----------
    init_mode : str
        One of ``"none"``, ``"clear"``, ``"clear_accum"``, or ``"pipelined"``.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((NB, BC, BK), torch.float32),
        B: T.Tensor((NB, BK, BC), torch.float32),
        C: T.Tensor((NB, BC, BC), torch.float32),
    ):
        with T.Kernel(1, NB, threads=64) as (bx, by):
            bn = bx * NB + by
            a_s = T.alloc_shared((BC, BK), torch.float32)
            b_s = T.alloc_shared((BK, BC), torch.float32)
            T.copy(A[bn, :, :], a_s)
            T.copy(B[bn, :, :], b_s)
            c_f = T.alloc_fragment((BC, BC), torch.float32)

            if init_mode == "clear":
                T.clear(c_f)

            if init_mode == "pipelined":
                # The idiom the check must never flag: clear_accum is an
                # expression, not a literal False.
                for k in T.Pipelined(2, num_stages=1):
                    T.gemm(a_s, b_s, c_f, clear_accum=(k == 0))
            else:
                T.gemm(a_s, b_s, c_f, clear_accum=(init_mode == "clear_accum"))

            T.copy(c_f, C[bn, :, :])

    return kernel


def _verify(func, capfd):
    """Run the pass on ``func`` and return whatever it wrote to stderr."""
    mod = tvm.IRModule.from_expr(func)
    tl.transform.VerifyBufferInit()(mod)
    return capfd.readouterr().err


# --------------------------------------------------------------- gemm cases


def test_warns_when_accumulator_is_uninitialized(capfd):
    """The #2936 reproducer: nothing writes c_f before the gemm reads it."""
    assert MESSAGE in _verify(_gemm_kernel("none"), capfd)


def test_silent_when_cleared_with_t_clear(capfd):
    """T.clear writes the accumulator, so the check stays quiet."""
    assert MESSAGE not in _verify(_gemm_kernel("clear"), capfd)


def test_silent_when_clear_accum_is_true(capfd):
    """A literal clear_accum=True means the gemm establishes C itself."""
    assert MESSAGE not in _verify(_gemm_kernel("clear_accum"), capfd)


def test_silent_for_the_pipelined_clear_accum_idiom(capfd):
    """clear_accum=(k == 0) is not a definite read of C.

    This is the regression guard on GetReadBeforeWriteRegions. Consuming
    GetAccessRegions().reads instead would flag this kernel, because its
    !is_one() predicate keeps C in the read set for a non-literal clear.
    """
    assert MESSAGE not in _verify(_gemm_kernel("pipelined"), capfd)


# ------------------------------------------------------------ general cases


def _extern_filled_kernel():
    """A shared buffer whose contents come from an opaque extern call."""

    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            T.call_extern("handle", "fill_it", T.address_of(s[0]))
            for i in T.Parallel(BC):
                C[bx, i] = s[i]

    return kernel


def test_silent_when_filled_by_an_extern_call(capfd):
    """A buffer handed to an opaque call is conservatively treated as written."""
    assert MESSAGE not in _verify(_extern_filled_kernel(), capfd)


def test_silent_for_global_scope_buffers(capfd):
    """Global buffers are the caller's responsibility, never reported."""

    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            for i in T.Parallel(BC):
                C[bx, i] = A[bx, i]

    assert MESSAGE not in _verify(kernel, capfd)


def test_silent_for_a_scoped_parameter_buffer(capfd):
    """A parameter is the caller's to fill, whatever storage scope it carries.

    Cross-thread scopes ask whether some other *node in this body* writes the
    buffer, and the caller is not one. A shared-scope parameter therefore has
    to be admitted on the strength of being a parameter, or it reports while
    an identical local-scope parameter stays silent.
    """

    @T.prim_func
    def kernel(
        S: T.Tensor((BC,), torch.float32, scope="shared.dyn"),
        C: T.Tensor((BC,), torch.float32),
    ):
        with T.Kernel(1, threads=64) as _:
            for i in T.Parallel(BC):
                C[i] = S[i]

    assert MESSAGE not in _verify(kernel, capfd)


def test_reports_a_shared_buffer_read_before_any_copy(capfd):
    """The general case the gemm-specific check could not see."""

    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            for i in T.Parallel(BC):
                C[bx, i] = s[i]

    assert MESSAGE in _verify(kernel, capfd)


def test_silent_when_shared_buffer_is_copied_first(capfd):
    """The same buffer read after a T.copy into it is fine."""

    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            T.copy(A[bx, :], s)
            for i in T.Parallel(BC):
                C[bx, i] = s[i]

    assert MESSAGE not in _verify(kernel, capfd)


def test_reports_an_uninitialized_local_buffer(capfd):
    """Register-scope buffers are covered too, not just shared memory."""

    @T.prim_func
    def kernel(C: T.Tensor((NB,), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            acc = T.alloc_local((1,), torch.float32)
            C[bx] = acc[0]

    assert MESSAGE in _verify(kernel, capfd)


def test_reports_a_store_that_reads_its_own_destination(capfd):
    """A store evaluates its right-hand side before it establishes anything.

    `idx = T.if_then_else(cond, i, idx)` carries the previous value forward, so
    with nothing written beforehand the first pass reads whatever the slot
    held. examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py writes the
    initializer that its sibling files omit.
    """

    @T.prim_func
    def kernel(A: T.Tensor((NB,), torch.int32), C: T.Tensor((NB,), torch.int32)):
        with T.Kernel(NB, threads=64) as bx:
            idx = T.alloc_local((1,), torch.int32)
            for i in T.serial(NB):
                idx[0] = T.if_then_else(A[i] > 0, i, idx[0])
            C[bx] = idx[0]

    assert MESSAGE in _verify(kernel, capfd)


def test_silent_when_the_destination_is_initialized_first(capfd):
    """The same carry-forward is fine once something establishes a value."""

    @T.prim_func
    def kernel(A: T.Tensor((NB,), torch.int32), C: T.Tensor((NB,), torch.int32)):
        with T.Kernel(NB, threads=64) as bx:
            idx = T.alloc_local((1,), torch.int32)
            idx[0] = 0
            for i in T.serial(NB):
                idx[0] = T.if_then_else(A[i] > 0, i, idx[0])
            C[bx] = idx[0]

    assert MESSAGE not in _verify(kernel, capfd)


def test_silent_when_shared_buffer_is_written_by_a_later_branch(capfd):
    """Warp specialization puts the producer after the consumer in the body.

    Both branches run concurrently and coordinate through barriers, so source
    order says nothing about what has executed. For shared scopes the question
    is whether any other operation writes the buffer at all.
    """

    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=128) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            tx = T.get_thread_binding()
            if tx < 64:
                for i in T.Parallel(BC):
                    C[bx, i] = s[i]
            else:
                T.copy(A[bx, :], s)

    assert MESSAGE not in _verify(kernel, capfd)


def test_reports_a_shared_store_that_reads_its_own_destination(capfd):
    """Ignoring source order must not let a store vouch for itself.

    A shared buffer is satisfied when some *other* node writes it, so the
    reading node's own write has to be discounted here exactly as it is for a
    gemm accumulating into an untouched accumulator. The store below is the
    only writer of `s` and reads the destination it is about to establish.
    """

    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=128) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            for i in T.Parallel(BC):
                s[i] = s[i] + A[bx, i]
            for i in T.Parallel(BC):
                C[bx, i] = s[i]

    assert MESSAGE in _verify(kernel, capfd)


def test_silent_when_another_node_writes_the_shared_destination(capfd):
    """The same accumulation is fine once anything else establishes a value."""

    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=128) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            T.clear(s)
            for i in T.Parallel(BC):
                s[i] = s[i] + A[bx, i]
            for i in T.Parallel(BC):
                C[bx, i] = s[i]

    assert MESSAGE not in _verify(kernel, capfd)


def test_reports_a_fragment_written_only_after_its_read(capfd):
    """Per-thread storage keeps order-sensitive tracking.

    Unlike shared memory, a fragment has no cross-thread producer, so source
    order is this thread's execution order and a write that comes afterwards
    does not excuse the read.
    """

    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            f = T.alloc_fragment((BC,), torch.float32)
            for i in T.Parallel(BC):
                C[bx, i] = f[i]
            T.clear(f)

    assert MESSAGE in _verify(kernel, capfd)


def test_reports_an_unguarded_loop_carried_read(capfd):
    """An unguarded loop-carried read is a real finding, not an artifact.

    The write at the bottom of the body reaches the read only on iterations
    after the first, so iteration zero reads whatever the fragment happened to
    hold. Source-order tracking gets this case right. A read guarded so it
    cannot run on the first iteration is the case it does not.
    """

    @T.prim_func
    def kernel(A: T.Tensor((BC,), torch.float32), C: T.Tensor((BC,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            f = T.alloc_fragment((1,), torch.float32)
            for k in T.serial(BC):
                C[k] = f[0]
                f[0] = A[k]

    assert MESSAGE in _verify(kernel, capfd)


# ---------------------------------------------------------- reporting cases


def _two_uninitialized_kernel():
    """Two distinct buffers, both read before anything writes them."""

    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            t = T.alloc_shared((BC,), torch.float32)
            for i in T.Parallel(BC):
                C[bx, i] = s[i] + t[i]

    return kernel


def test_gemm_accumulator_gets_the_clear_accum_hint(capfd):
    """An uninitialized gemm C operand names the fix that applies to it."""
    err = _verify(_gemm_kernel("none"), capfd)

    assert MESSAGE in err
    assert GEMM_HINT in err


def test_non_gemm_report_omits_the_gemm_hint(capfd):
    """A plain shared-buffer read gets the generic advice, not clear_accum."""
    err = _verify(_two_uninitialized_kernel(), capfd)

    assert MESSAGE in err
    assert GEMM_HINT not in err


def test_aggregates_multiple_uninitialized_buffers(capfd):
    """Two bad buffers produce one aggregated warning, not two."""
    err = _verify(_two_uninitialized_kernel(), capfd)

    assert err.count(MESSAGE) == 1
    assert "2 buffer(s)" in err
    assert "[1]" in err and "[2]" in err


def test_reports_each_buffer_once_however_many_reads(capfd):
    """A buffer read repeatedly still produces a single entry."""

    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            for i in T.Parallel(BC):
                C[bx, i] = s[i] + s[(i + 1) % BC] + s[(i + 2) % BC]

    err = _verify(kernel, capfd)

    assert "1 buffer(s)" in err
    assert "[2]" not in err


# ------------------------------------------------------------------ wiring


@pytest.fixture
def no_kernel_cache():
    """Compile without the kernel cache so the pass runs on every compile.

    A cached kernel is returned without re-running the pass pipeline, which
    would make the warning silently disappear on the second compile.
    """
    was_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    yield
    if was_enabled:
        tilelang.enable_cache()


def _reducer_kernel():
    """A kernel using the legacy T.alloc_reducer / T.finalize_reducer form.

    At the point this pass runs, the tl.tileop.finalize_reducer call carries a
    single argument while the op's builder reads args[1]. Parsing it there
    throws, which must never take a compile down.
    """

    @T.prim_func
    def kernel(A: T.Tensor((BC, BK), torch.float32), B: T.Tensor((BC,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            a_f = T.alloc_fragment((BC, BK), torch.float32)
            r_f = T.alloc_reducer((BC,), torch.float32, op="sum", replication="all")
            T.clear(r_f)
            T.copy(A, a_f)
            for i, j in T.Parallel(BC, BK):
                r_f[i] += a_f[i, j]
            T.finalize_reducer(r_f)
            T.copy(r_f, B)

    return kernel


def test_silent_for_the_reducer_v2_idiom(capfd):
    """The first-class reducer ops inherit the default read-before-write set.

    `T.reducer_init` establishes the accumulator, `T.reducer_update` reads and
    writes it, and `T.finalize_reducer` reads it into a destination it writes.
    None of these ops overrides GetReadBeforeWriteRegions, so this pins that
    the op-agnostic default is already right for them.
    """

    @T.prim_func
    def kernel(A: T.Tensor((BC,), torch.float32), B: T.Tensor((1,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            src = T.alloc_fragment((BC,), torch.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), torch.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(BC):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), torch.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    assert MESSAGE not in _verify(kernel, capfd)


def test_reducer_kernel_does_not_raise(capfd):
    """A tile op the pass cannot parse degrades to opaque, it does not throw."""
    err = _verify(_reducer_kernel(), capfd)

    assert MESSAGE not in err


@tilelang.testing.requires_cuda
def test_reducer_kernel_compiles(no_kernel_cache):
    """End-to-end: the check must not break reducer kernels."""
    tilelang.compile(_reducer_kernel())


def test_check_enabled_by_default(capfd):
    """The check is on unless a pass config turns it off."""
    assert MESSAGE in _verify(_gemm_kernel("none"), capfd)


def test_check_can_be_disabled_via_pass_config(capfd):
    """tl.disable_buffer_init_check silences the pass itself, not just the
    pipeline that schedules it."""
    from tilelang.transform import PassContext

    with PassContext(config={"tl.disable_buffer_init_check": True}):
        assert MESSAGE not in _verify(_gemm_kernel("none"), capfd)


@tilelang.testing.requires_cuda
def test_pipeline_warns_by_default(capfd, no_kernel_cache):
    """The wired pipeline warns on an uninitialized accumulator."""
    tilelang.compile(_gemm_kernel("none"))

    assert MESSAGE in capfd.readouterr().err


@tilelang.testing.requires_cuda
def test_pipeline_silent_when_disabled_via_pass_config(capfd, no_kernel_cache):
    """The pass config suppresses the warning through the whole pipeline."""
    tilelang.compile(
        _gemm_kernel("none"),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK: True},
    )

    assert MESSAGE not in capfd.readouterr().err


if __name__ == "__main__":
    tilelang.testing.main()
