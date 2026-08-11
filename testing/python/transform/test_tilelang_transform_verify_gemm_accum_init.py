import pytest
import torch

import tilelang
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

BC, BK, NB = 16, 64, 16
MESSAGE = "Uninitialized T.gemm accumulator"


def _kernel(init_mode):
    """Build the #2936 kernel with one accumulator-initialization variant.

    Parameters
    ----------
    init_mode : str
        One of ``"none"`` (no initialization at all), ``"clear"``, ``"fill"``,
        ``"copy"``, ``"parallel_store"``, ``"clear_accum"``, ``"prior_gemm"``,
        or ``"pipelined"``.

    Returns
    -------
    tvm.tirx.PrimFunc
        The traced kernel.
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
            elif init_mode == "fill":
                T.fill(c_f, 0)
            elif init_mode == "copy":
                T.copy(C[bn, :, :], c_f)
            elif init_mode == "parallel_store":
                for i, j in T.Parallel(BC, BC):
                    c_f[i, j] = 0.0

            if init_mode == "prior_gemm":
                # The first gemm is exempt via clear_accum and writes c_f, so
                # the second gemm has a definitely-initialized accumulator.
                T.gemm(a_s, b_s, c_f, clear_accum=True)
                T.gemm(a_s, b_s, c_f)
            elif init_mode == "pipelined":
                # The idiom the check must never flag: clear_accum is an
                # expression, not a literal False.
                for k in T.Pipelined(2, num_stages=1):
                    T.gemm(a_s, b_s, c_f, clear_accum=(k == 0))
            else:
                T.gemm(a_s, b_s, c_f, clear_accum=(init_mode == "clear_accum"))

            T.copy(c_f, C[bn, :, :])

    return kernel


def _kernel_two_uninitialized():
    """Build a kernel with two distinct, both-uninitialized accumulators."""

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
            d_f = T.alloc_fragment((BC, BC), torch.float32)
            T.gemm(a_s, b_s, c_f)
            T.gemm(a_s, b_s, d_f)
            T.copy(c_f, C[bn, :, :])
            T.copy(d_f, C[bn, :, :])

    return kernel


def _make(init_mode):
    """Wrap :func:`_kernel` in an IRModule ready to run the pass on."""
    return tvm.IRModule.from_expr(_kernel(init_mode))


def _verify(mod, capfd):
    """Run the pass on ``mod`` and return whatever it wrote to stderr."""
    tl.transform.VerifyGemmAccumInit()(mod)
    return capfd.readouterr().err


@pytest.fixture
def no_kernel_cache():
    """Compile without the kernel cache so the pass runs on every compile.

    A cached kernel is returned without re-running the pass pipeline, which
    would make a warning silently disappear on the second compile.
    """
    was_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    yield
    if was_enabled:
        tilelang.enable_cache()


def test_warns_when_accumulator_is_uninitialized(capfd):
    """The #2936 reproducer: no initialization anywhere, so it must warn."""
    assert MESSAGE in _verify(_make("none"), capfd)


def test_aggregates_multiple_uninitialized_accumulators(capfd):
    """Two bad accumulators produce one aggregated warning, not two."""
    mod = tvm.IRModule.from_expr(_kernel_two_uninitialized())
    err = _verify(mod, capfd)

    assert err.count(MESSAGE) == 1
    assert "2 accumulator(s)" in err
    assert "[1]" in err and "[2]" in err


def test_silent_when_cleared_with_t_clear(capfd):
    """T.clear writes the accumulator, so the check stays quiet."""
    assert MESSAGE not in _verify(_make("clear"), capfd)


def test_silent_when_filled_with_t_fill(capfd):
    """T.fill marks its region as written, like T.clear."""
    assert MESSAGE not in _verify(_make("fill"), capfd)


def test_silent_when_initialized_by_a_copy(capfd):
    """A T.copy whose destination is the accumulator counts as a write."""
    assert MESSAGE not in _verify(_make("copy"), capfd)


def test_silent_when_initialized_by_a_parallel_store(capfd):
    """A direct BufferStore in a T.Parallel loop counts as a write."""
    assert MESSAGE not in _verify(_make("parallel_store"), capfd)


def test_silent_when_clear_accum_is_true(capfd):
    """A literal clear_accum=True zeroes the accumulator in the gemm itself."""
    assert MESSAGE not in _verify(_make("clear_accum"), capfd)


def test_silent_when_a_prior_gemm_initialized_the_accumulator(capfd):
    """A preceding gemm on the same buffer leaves it written."""
    assert MESSAGE not in _verify(_make("prior_gemm"), capfd)


def test_silent_for_the_pipelined_clear_accum_idiom(capfd):
    """clear_accum=(k == 0) is not a literal False, so it is never reported."""
    assert MESSAGE not in _verify(_make("pipelined"), capfd)


def test_check_enabled_by_default():
    """The check is on unless a pass config turns it off."""
    from tilelang.backend.pass_pipeline import pipeline_utils

    assert pipeline_utils.should_enable_gemm_accum_init_check() is True


def test_check_can_be_disabled_via_pass_config():
    """tl.disable_gemm_accum_init_check silences the check."""
    from tilelang.backend.pass_pipeline import pipeline_utils
    from tilelang.transform import PassContext

    with PassContext(config={"tl.disable_gemm_accum_init_check": True}):
        assert pipeline_utils.should_enable_gemm_accum_init_check() is False


@tilelang.testing.requires_cuda
def test_pipeline_warns_by_default(capfd, no_kernel_cache):
    """The wired pipeline warns on an uninitialized accumulator."""
    tilelang.compile(_kernel("none"))

    assert MESSAGE in capfd.readouterr().err


@tilelang.testing.requires_cuda
def test_pipeline_silent_when_disabled_via_pass_config(capfd, no_kernel_cache):
    """The pass config suppresses the warning through the whole pipeline."""
    tilelang.compile(
        _kernel("none"),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_GEMM_ACCUM_INIT_CHECK: True},
    )

    assert MESSAGE not in capfd.readouterr().err


if __name__ == "__main__":
    tilelang.testing.main()
