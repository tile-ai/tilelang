"""Metal fragment-reduce single-simdgroup plan enforcement.

Every XOR-butterfly step must remain inside a 32-lane simdgroup. Metal has
no named barriers, so an in-place cross-simdgroup combine cannot rely on SIMD
lockstep ordering. For each step, ``nt = extent * scale`` must describe a
power-of-two closure no wider than 32 lanes, and the full threadgroup must be
an integer number of complete ``nt`` blocks. Otherwise a partner read can
escape the participating range, the threadgroup, or the initialized scratch.

The enforced criterion is:

    reject iff nt = extent*scale is not a power of two
                OR nt > 32
                OR N % nt != 0
    (allow iff nt is a power of two AND nt <= 32 AND N % nt == 0)

Power-of-two replication plans and complete multi-block closures remain
valid, including the 32-lane reduction shape used by Qwen and DeepSeek-style
normalization and routing kernels.

These tests are compile/lower-only (no GPU launch): the Metal backend
must reject unsafe plans at lowering time with the single-simdgroup
diagnostic (naming the step's scale, the offending nt, and the
threadgroup extent N), and must accept every plan whose nt is a power
of two <= 32 with N % nt == 0, with all XOR offsets < 32 AND the raw
[0, N) execution prefix closed under every mask found in the MSL.

Additional coverage in this file:
  - tl.infinity lowering/offline-MSL legality for fp32/fp16/bf16;
  - bf16/fp16 reduce lowering (fp32-accumulate path) and offline MSL
    legality, including the bf16 max INFINITY identity;
  - clear=False duplicate-buffer update lowering (fp32 + bf16);
  - multi-block closure (N = k*nt) runtime numerics on MPS;
  - Reducer v2 (FinalizeReducerOp) fail-loud boundary on Metal.
All payloads are tiny synthetic tensors; no model weights or downloads.
"""

import os
import re
import shutil
import subprocess
import tempfile

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm


def _metal_toolchain_available() -> bool:
    """True only when the offline Metal compiler CLI is actually usable.

    ``shutil.which("xcrun")`` alone is not enough: macOS hosts with only
    CommandLineTools have xcrun but no ``metal`` utility (that ships with
    Xcode). The runtime MPS tests below are the fallback MSL-validity check
    on such hosts.
    """
    if shutil.which("xcrun") is None:
        return False
    try:
        proc = subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, text=True
        )
    except OSError:
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


_HAS_METAL_TOOLCHAIN = _metal_toolchain_available()


def _make_allreduce_dim0_scale_kernel(
    reduce_fn, logical_width, scale, threads=None, dtype="float32", clear=True
):
    """Copy of the upstream public constructor
    (testing/python/language/test_tilelang_language_reduce.py), used as
    the public construction path for the Metal backend.

    Threads default to logical_width * scale (N == nt). Supplying an
    explicit value decouples the threadgroup extent from nt so misaligned
    threadgroups are reachable through the same public constructor.

    ``dtype`` covers the fp32/fp16/bf16 paths; ``clear=False`` exercises
    the duplicate-buffer update path (Phase 3) that accumulate-into-dst
    reductions take on Metal.
    """
    if threads is None:
        threads = logical_width * scale

    @T.prim_func
    def kernel(
        A: T.Tensor((logical_width, scale), dtype),
        B: T.Tensor((scale,), dtype),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((logical_width, scale), dtype)
            dst = T.alloc_fragment((scale,), dtype)
            T.copy(A, src)
            reduce_fn(src, dst, dim=0, clear=clear)
            T.copy(dst, B)

    return kernel


def _make_infinity_fill_kernel(dtype):
    """Tiny fill kernel whose only payload is ``tl.infinity(dtype)``.

    Used to verify that the Metal ``tl.infinity`` lowering folds to a
    constant ``FloatImm`` and that the codegen emits an MSL-legal
    ``INFINITY`` literal for fp32, fp16 and bf16 destinations.
    """

    @T.prim_func
    def kernel(A: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            T.fill(A, T.infinity(dtype))

    return kernel


def _make_finalize_reducer_v2_kernel():
    """Minimal reducer-v2 epoch (``T.alloc_reducer`` + ``finalize_reducer``).

    Metal registers no ``FinalizeReducerOp`` implementation upstream, so
    lowering this kernel must fail loudly instead of silently producing an
    invalid kernel. This documents the v2 boundary of the PR: only legacy
    ``T.reduce`` is covered on Metal.
    """

    @T.prim_func
    def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((8,), T.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(8):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return kernel


def _xor_closed(offsets, n):
    """The raw execution prefix [0, n) must be closed under every XOR
    mask used in the MSL.

    XOR with a mask is a permutation of the integers, so the image of
    [0, n) is [0, n) iff the maximum partner stays inside the range.
    For the participating range [0, nt), this holds iff nt is a
    power of two (a non-power-of-two nt e.g. 24 with masks 12/6/3 lets
    some partner escape even though every mask is < 32). The raw
    butterfly runs on ALL N threads without a tid < nt guard, so the
    domain that must actually be closed is the whole prefix [0, N);
    with nt a power of two this holds iff N % nt == 0 (every nt-block
    complete).
    """
    assert n > 0
    return all(max(tid ^ mask for tid in range(n)) < n for mask in offsets)


def _lower_metal(prim_func):
    target = tvm.target.Target("metal", tvm.target.Target("llvm"))
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            target_host="llvm",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact.kernel_source or ""


def _compile_metal(prim_func):
    """Compile for real MPS execution through the supported Metal path.

    TileLang's Metal execution backend is ``torch`` (``torch.mps.compile_shader``,
    see tilelang/metal/execution_backend.py); the ``tvm_ffi`` backend rejects
    torch MPS tensors with a device_type mismatch, so runtime tests must use
    ``execution_backend="torch"``. Compilation of the MSL happens lazily on
    the first call, which doubles as the device-side MSL legality check
    (including the bf16 ``INFINITY`` literal).
    """
    return tilelang.compile(
        prim_func,
        out_idx=-1,
        target="metal",
        execution_backend="torch",
    )


def _compile_msl_with_metal_toolchain(src, *, label=""):
    """Validate generated MSL with the offline Metal compiler (CPU-only).

    This is the GPU-free legality check: it runs ``xcrun metal -c`` on the
    lowered source and fails the test with the compiler diagnostics if the
    shader does not parse. It is what catches MSL-invalid literals such as
    a bf16 ``INFINITY`` that ``torch.mps.compile_shader`` would only reject
    later at launch time.
    """
    assert _HAS_METAL_TOOLCHAIN, "xcrun is required for MSL legality checks"
    with tempfile.TemporaryDirectory() as tmp:
        msl_path = os.path.join(tmp, "kernel.metal")
        air_path = os.path.join(tmp, "kernel.air")
        with open(msl_path, "w") as f:
            f.write(src)
        last_err = ""
        for std in (None, "osx-metal3.0", "metal3.0"):
            cmd = ["xcrun", "-sdk", "macosx", "metal", "-c", msl_path, "-o", air_path]
            if std is not None:
                cmd.insert(4, f"-std={std}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return
            last_err = proc.stderr
        pytest.fail(
            f"MSL compile failed{(' for ' + label) if label else ''}:\n{last_err}\n"
            f"--- source ---\n{src}"
        )


@pytest.mark.parametrize(
    ("logical_width", "scale", "nt"),
    [
        (16, 3, 48),  # 32^24 = 56 -> OOB on a 48-thread group
        # group; nt is neither a power of two nor <= 32
        (8, 5, 40),  # same window: offset 20, partner 32^20 = 52 -> OOB
    ],
)
def test_rejects_nt_gt_32_non_pow2_scale(logical_width, scale, nt):
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*\b32\b",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale))


@pytest.mark.parametrize(
    ("logical_width", "scale", "nt"),
    [
        # nt <= 32 yet not a power of two.
        # Masks 12/6/3: partner 16^12 = 28 >= 24 escapes [0, 24) (OOB
        # scratch / never-written slot) even though every mask < 32.
        (8, 3, 24),
        # Masks 6/3: partner 8^6 = 14 >= 12 escapes [0, 12); slots
        # 12..15 exist only in codegen padding and are never written.
        (4, 3, 12),
    ],
)
def test_rejects_nt_le_32_non_pow2_scale(logical_width, scale, nt):
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*\b32\b",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale))


@pytest.mark.parametrize(
    ("logical_width", "scale", "nt"),
    [
        (16, 4, 64),  # offsets 32..4: first offset crosses the boundary
        (32, 2, 64),  # offsets 32..2: first offset crosses the boundary
        (64, 1, 64),  # cross-group extent
        (128, 1, 128),  # cross-group extent
    ],
)
def test_rejects_nt_gt_32_pow2_scale(logical_width, scale, nt):
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*\b32\b",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale))


@pytest.mark.parametrize(
    ("logical_width", "scale", "threads", "nt"),
    [
        # nt is a power of two <= 32, but the
        # threadgroup extent is NOT an integer multiple of nt, so the
        # last incomplete nt-block reads partners >= N (OOB: scratch is
        # sized N) or never-written codegen-padding slots.  All these
        # shapes reach the plan-level gate and were previously allowed.
        # (8,2,24): nt=16, N%16=8; partners 24..31 escape the 24-slot
        #           scratch (maximum partners
        #           tid 16..23 ^ 8 = 24..31).
        (8, 2, 24, 16),
        # (2,4,12): nt=8, N%8=4; tid 8..11 ^ 4 = 12..15 read slots that
        #           exist only in codegen padding and are never written.
        (2, 4, 12, 8),
        # (16,2,40): nt=32, N%32=8; task-specified (nt=32, N=40) case.
        (16, 2, 40, 32),
        (8, 4, 40, 32),  # same N=40 misalignment, nt=32
        (4, 4, 20, 16),  # nt=16, N%16=4
        (2, 2, 6, 4),  # nt=4, N%4=2
        (8, 2, 56, 16),  # N=3*nt+8 boundary: tail block incomplete
    ],
)
def test_rejects_misaligned_threadgroup(logical_width, scale, threads, nt):
    """N % nt != 0 must produce a tail-block alignment diagnostic."""
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*N = {threads}.*"
        rf"not an integer multiple of nt = {nt}",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale, threads))


@pytest.mark.parametrize(
    ("logical_width", "scale", "threads", "nt"),
    [
        # N == nt single-block closures.
        (32, 1, 32, 32),  # v2 boundary: full 32-lane closure, offsets 16..1
        (16, 2, 32, 32),  # power-of-two replication: offsets 16..2
        (8, 4, 32, 32),  # power-of-two replication: offsets 16..4
        (16, 1, 16, 16),  # sub-simdgroup closure: offsets 8..1
        (8, 2, 16, 16),  # power-of-two nt, offsets 8..2
        (2, 4, 8, 8),  # power-of-two nt, offset 4
        (4, 8, 32, 32),  # replicated small fragment: offsets 16..8
        # Multi-block closures (N = k*nt complete blocks; every
        # block computes its own copy of the same group total, owners
        # store): the [0, N) execution prefix must be closed under every
        # mask — this is the property N % nt == 0 guarantees.
        (8, 2, 32, 16),  # N = 2*nt: two complete 16-lane closures
        (2, 4, 16, 8),  # N = 2*nt
        (2, 4, 32, 8),  # N = 4*nt
        (8, 2, 64, 16),  # N = 4*nt
        (4, 4, 32, 16),  # N = 2*nt
        (16, 2, 96, 32),  # N = 3*nt
        (8, 2, 128, 16),  # N = 8*nt (v2-scale threadgroup extent)
    ],
)
def test_allows_pow2_nt_le_32_closure(logical_width, scale, threads, nt):
    src = _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale, threads))
    # every XOR butterfly offset must be < 32 (closure inside one
    # simdgroup; offsets are nt/2 halving down to scale), the max offset
    # must be nt/2, the threadgroup extent must be an integer multiple
    # of nt (complete nt-blocks in the [0, N) execution prefix),
    # and the raw [0, N) prefix must be closed under every mask in the
    # MSL on the actual execution domain.
    offsets = [int(v) for v in re.findall(r"\^ ?(\d+)", src)]
    assert offsets, "no XOR butterfly offsets found in MSL"
    assert all(o < 32 for o in offsets)
    assert max(offsets) == nt // 2
    assert threads % nt == 0, (
        f"threadgroup extent {threads} is not a multiple of nt={nt}: incomplete tail block would escape the execution prefix"
    )
    assert _xor_closed(offsets, threads), f"XOR masks {sorted(set(offsets))} not closed on [0, {threads})"


@tilelang.testing.requires_metal
@pytest.mark.parametrize("reduce_fn", [T.reduce_sum, T.reduce_max], ids=["sum", "max"])
def test_runtime_float32_reduction(reduce_fn):
    """Execute a complete 32-lane closure on MPS and check its value."""
    logical_width, scale = 16, 2
    kernel = _compile_metal(_make_allreduce_dim0_scale_kernel(reduce_fn, logical_width, scale))
    values = torch.linspace(-1.5, 2.0, logical_width * scale, dtype=torch.float32).reshape(logical_width, scale).to("mps")
    result = torch.empty(scale, dtype=torch.float32, device="mps")
    kernel(values, result)
    torch.mps.synchronize()

    reference = values.cpu().sum(dim=0) if reduce_fn is T.reduce_sum else values.cpu().max(dim=0).values
    torch.testing.assert_close(result.cpu(), reference, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# tl.infinity lowering (fp32 / fp16 / bf16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", ["float32", "float16", "bfloat16"], ids=["fp32", "fp16", "bf16"]
)
def test_infinity_lowers_to_msl_infinity_literal(dtype):
    """tl.infinity must fold to a constant FloatImm that the Metal codegen
    prints as the MSL ``INFINITY`` literal (never an unsupported extern
    call or a per-dtype hex pattern). bf16 additionally requires an
    explicit ``(bfloat)`` cast: MSL has no implicit float/half -> bfloat
    conversion, so the bare macro is rejected by the Metal compiler."""
    src = _lower_metal(_make_infinity_fill_kernel(dtype))
    assert "INFINITY" in src
    if dtype == "bfloat16":
        assert "(bfloat)(INFINITY)" in src
        assert "(half)(INFINITY)" not in src
    elif dtype == "float16":
        assert "(half)(INFINITY)" in src
        assert "(bfloat)(INFINITY)" not in src
    else:
        assert "(bfloat)(INFINITY)" not in src
        assert "(half)(INFINITY)" not in src


@pytest.mark.skipif(
    not _HAS_METAL_TOOLCHAIN, reason="Metal toolchain (xcrun metal) not available"
)
@pytest.mark.parametrize(
    "dtype", ["float32", "float16", "bfloat16"], ids=["fp32", "fp16", "bf16"]
)
def test_infinity_msl_compiles(dtype):
    """The emitted ``INFINITY`` literal must be accepted by the offline Metal
    compiler. This is the legality check for the bf16 case in particular:
    ``bfloat`` has no dedicated MSL infinity literal, so the codegen's bare
    ``INFINITY`` relies on MSL's implicit float -> bfloat conversion."""
    _compile_msl_with_metal_toolchain(
        _lower_metal(_make_infinity_fill_kernel(dtype)), label=f"infinity-{dtype}"
    )


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "dtype", ["float32", "float16", "bfloat16"], ids=["fp32", "fp16", "bf16"]
)
def test_runtime_infinity_fill(dtype):
    """Real MPS execution: a fill with tl.infinity must produce +inf."""
    kernel = _compile_metal(_make_infinity_fill_kernel(dtype))
    out = torch.empty(32, dtype=getattr(torch, dtype), device="mps")
    kernel(out)
    torch.mps.synchronize()
    assert torch.all(out.cpu() == torch.inf), f"fill with tl.infinity({dtype}) != inf"


# ---------------------------------------------------------------------------
# bf16 / fp16 reduce lowering (fp32-accumulate path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce_fn", [T.reduce_sum, T.reduce_max], ids=["sum", "max"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"], ids=["bf16", "fp16"])
def test_reduce_bf16_fp16_lowers(reduce_fn, dtype):
    """bf16/fp16 fragment reduces must lower without crashing and keep the
    same single-simdgroup XOR-butterfly closure as the fp32 path. bf16
    accumulates in fp32, so the butterfly offsets must still be present in
    the MSL."""
    logical_width, scale, threads = 16, 2, 32
    src = _lower_metal(
        _make_allreduce_dim0_scale_kernel(reduce_fn, logical_width, scale, threads, dtype=dtype)
    )
    if dtype == "bfloat16":
        assert "bfloat" in src
    else:
        assert "half" in src
    offsets = [int(v) for v in re.findall(r"\^ ?(\d+)", src)]
    assert offsets, "no XOR butterfly offsets found in MSL"
    assert all(o < 32 for o in offsets)
    assert max(offsets) == 16  # nt = extent*scale = 32 -> offsets 16..2
    assert _xor_closed(offsets, threads)
    if reduce_fn is T.reduce_max:
        # max identity is +inf; for bf16 it is materialized in fp32 and
        # must still print as a legal MSL INFINITY literal.
        assert "INFINITY" in src


@pytest.mark.skipif(
    not _HAS_METAL_TOOLCHAIN, reason="Metal toolchain (xcrun metal) not available"
)
@pytest.mark.parametrize("reduce_fn", [T.reduce_sum, T.reduce_max], ids=["sum", "max"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"], ids=["bf16", "fp16"])
def test_reduce_bf16_fp16_msl_compiles(reduce_fn, dtype):
    """Offline MSL legality for the bf16/fp16 reduce path (including the
    bf16 max INFINITY identity)."""
    src = _lower_metal(
        _make_allreduce_dim0_scale_kernel(reduce_fn, 16, 2, 32, dtype=dtype)
    )
    _compile_msl_with_metal_toolchain(src, label=f"reduce-{dtype}-{reduce_fn.__name__}")


@tilelang.testing.requires_metal
@pytest.mark.parametrize("reduce_fn", [T.reduce_sum, T.reduce_max], ids=["sum", "max"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"], ids=["bf16", "fp16"])
def test_runtime_bf16_fp16_reduction(reduce_fn, dtype):
    """Real MPS execution for bf16/fp16. Small integer payloads keep sums
    exact in both formats so the comparison is not tolerance-dominated."""
    logical_width, scale = 16, 2
    kernel = _compile_metal(
        _make_allreduce_dim0_scale_kernel(reduce_fn, logical_width, scale, dtype=dtype)
    )
    values = torch.arange(1, logical_width * scale + 1, dtype=torch.float32)
    values = values.reshape(logical_width, scale).to(getattr(torch, dtype)).to("mps")
    result = torch.empty(scale, dtype=getattr(torch, dtype), device="mps")
    kernel(values, result)
    torch.mps.synchronize()

    if reduce_fn is T.reduce_sum:
        reference = values.cpu().to(torch.float32).sum(dim=0)
    else:
        reference = values.cpu().to(torch.float32).max(dim=0).values
    torch.testing.assert_close(
        result.cpu().to(torch.float32), reference, rtol=1e-2, atol=1e-2
    )


# ---------------------------------------------------------------------------
# clear=False (duplicate-buffer update path) and multi-block closures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"], ids=["fp32", "bf16"])
def test_reduce_clear_false_lowers(dtype):
    """clear=False forces the duplicate-buffer Phase-3 update path (and, for
    bf16, an fp32 accumulator with a final cast back to dst). It must lower
    without crashing and keep the butterfly closure."""
    logical_width, scale, threads = 16, 2, 32
    src = _lower_metal(
        _make_allreduce_dim0_scale_kernel(
            T.reduce_sum, logical_width, scale, threads, dtype=dtype, clear=False
        )
    )
    offsets = [int(v) for v in re.findall(r"\^ ?(\d+)", src)]
    assert offsets, "no XOR butterfly offsets found in MSL"
    assert all(o < 32 for o in offsets)
    assert max(offsets) == 16
    assert _xor_closed(offsets, threads)


@pytest.mark.skipif(
    not _HAS_METAL_TOOLCHAIN, reason="Metal toolchain (xcrun metal) not available"
)
def test_reduce_clear_false_bf16_msl_compiles():
    """Offline MSL legality for the bf16 clear=False update path (fp32
    accumulator scratch + final bf16 cast)."""
    src = _lower_metal(
        _make_allreduce_dim0_scale_kernel(
            T.reduce_sum, 16, 2, 32, dtype="bfloat16", clear=False
        )
    )
    _compile_msl_with_metal_toolchain(src, label="reduce-clear-false-bf16")


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    ("logical_width", "scale", "threads"),
    [
        (8, 2, 32),  # nt=16, N = 2*nt: two complete 16-lane closures
        (2, 4, 32),  # nt=8, N = 4*nt
        (16, 2, 96),  # nt=32, N = 3*nt
        (8, 2, 128),  # nt=16, N = 8*nt
    ],
    ids=["2x16", "4x8", "3x32", "8x16"],
)
@pytest.mark.parametrize("reduce_fn", [T.reduce_sum, T.reduce_max], ids=["sum", "max"])
def test_runtime_multi_block_closure(reduce_fn, logical_width, scale, threads):
    """Multi-block closures (N = k*nt complete nt-blocks) must compute the
    same group totals as a single block: every complete block computes its
    own copy and the layout owners store once."""
    kernel = _compile_metal(
        _make_allreduce_dim0_scale_kernel(reduce_fn, logical_width, scale, threads)
    )
    values = torch.linspace(-1.5, 2.0, logical_width * scale, dtype=torch.float32)
    values = values.reshape(logical_width, scale).to("mps")
    result = torch.empty(scale, dtype=torch.float32, device="mps")
    kernel(values, result)
    torch.mps.synchronize()

    reference = (
        values.cpu().sum(dim=0)
        if reduce_fn is T.reduce_sum
        else values.cpu().max(dim=0).values
    )
    torch.testing.assert_close(result.cpu(), reference, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Reducer v2 (FinalizeReducerOp) boundary: unsupported on Metal
# ---------------------------------------------------------------------------


def test_finalize_reducer_v2_rejected_on_metal():
    """Reducer v2 (``tl.finalize_reducer`` / ``FinalizeReducerOp``) has no
    Metal implementation upstream, so lowering must fail loudly with the
    registered-implementation diagnostic instead of emitting an invalid
    kernel. This PR only covers the legacy ``T.reduce`` path."""
    with pytest.raises(
        Exception, match=r"no finalize_reducer implementation is registered"
    ):
        _lower_metal(_make_finalize_reducer_v2_kernel())


if __name__ == "__main__":
    tilelang.testing.main()
