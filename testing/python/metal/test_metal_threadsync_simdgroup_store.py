"""Test ThreadSync barriers for simdgroup_store / simdgroup_load tile accesses.

Regression for the ThreadSyncPlanner gap fixed in
src/transform/thread_storage_sync.cc: the pointer argument of
simdgroup_store/simdgroup_load is an address_of(BufferLoad) which the
planner used to record as a *single-element kRead* — hiding the write side
of simdgroup_store and the full 8x8 tile footprint. Multi-simdgroup staged
epilogues (fragment -> shared via simdgroup_store, then cross-simdgroup
reads) therefore got 0 barriers between the staging stores and the reads
(the 256-thread staged kernel failed 12/12 rounds, maxerr=5.969).

The same gap exists for the second official pointer form, the Metal
macro / tensor-intrin T.access_ptr (tvm_access_ptr) used by
metal_macro_generator.py and tensor_intrin/metal.py: those accesses were
recorded as plain single-element pointer accesses and FindConflict could
prove them disjoint from other accesses via PointerAccessIsDisjoint —
again skipping the barrier between overlapping cross-simdgroup tiles. The regression below
covers both forms: the staged address_of path and the explicit
T.access_ptr simdgroup_store/load path (normal + transpose + padded
stride, cross-warp RAW and WAR).

Acceptance criteria:
  1. per-round max abs err vs fp32 CPU reference < 1e-2
     (fp16 noise floor is far below 1e-2, which separates races like 5.969
     from rounding noise),
  2. run-to-run determinism: N rounds bit-identical (int32 bitview),
  3. fresh output buffer allocated per round, all rounds must pass,
  4. static MSL: a threadgroup_barrier must exist *after* the last
     simdgroup_store of the staged epilogue (bar_after_store=True),
  5. static MSL: for the T.access_ptr forms a threadgroup_barrier must
     exist between the producer simdgroup op (store for RAW / load for
     WAR) and the consumer simdgroup op of the other warp (pre-fix: no
     such barrier, 8/8 configs),
  6. the direct fragment-C -> global path (dense_gemm_frag
     t256, zero-communication) must keep barriers==2 and stay bit-identical.

The staged kernel below represents a DeepSeek V4 Flash-style gated-MoE
epilogue (t256 = 8 simdgroups, block_M=8 -> each warp owns one row; the
epilogue reads Cs_sh[i, 0], which is warp 0's column).
"""

import tilelang
import tilelang.metal.language as T
import tilelang.testing
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="PyTorch MPS device is required",
)

# fp16 noise floor measured on M2 (maxerr of the correct t32 staged path).
# 1e-2 separates the observed race failures (O(1), maxerr=5.969)
# rounding noise is ~2.4e-04.
MAX_ERR = 1e-2
N_ROUNDS = 12
STAGED_SHAPE = "64_512_2048"  # T_ x Is x H


@tilelang.jit(execution_backend="tvm_ffi")
def _staged_msg(x, Wg, Wu, Ws_g, mid, T_, Is, H, block_M=8, block_N=128, block_K=32, threads=256):
    """DeepSeek V4 Flash-style t256 epilogue: fragment-to-shared
    T.copy + cross-group epilogue read (Cs_sh[i,0] is warp-0's column)."""
    x: T.Tensor((T_, H), "float16")
    Wg: T.Tensor((Is, H), "float16")
    Wu: T.Tensor((Is, H), "float16")
    Ws_g: T.Tensor((1, H), "float16")
    mid: T.Tensor((T_, Is), "float16")
    with T.Kernel(T.ceildiv(T_, block_M), T.ceildiv(Is, block_N), threads=threads) as (bm, bn):
        A_s = T.alloc_shared((block_M, block_K), "float16")
        Bg_s = T.alloc_shared((block_N, block_K), "float16")
        Bu_s = T.alloc_shared((block_N, block_K), "float16")
        Bs_s = T.alloc_shared((block_N, block_K), "float16")
        Cg = T.alloc_fragment((block_M, block_N), "float32")
        Cu = T.alloc_fragment((block_M, block_N), "float32")
        Cs = T.alloc_fragment((block_M, block_N), "float32")
        Cg_sh = T.alloc_shared((block_M, block_N), "float32")
        Cu_sh = T.alloc_shared((block_M, block_N), "float32")
        Cs_sh = T.alloc_shared((block_M, block_N), "float32")
        T.clear(Cg)
        T.clear(Cu)
        T.clear(Cs)
        for kk in T.serial(T.ceildiv(H, block_K)):
            for i, j in T.Parallel(block_M, block_K):
                A_s[i, j] = T.if_then_else(
                    (bm * block_M + i < T_) and (kk * block_K + j < H), x[bm * block_M + i, kk * block_K + j], T.cast(0, "float16")
                )
            for i, j in T.Parallel(block_N, block_K):
                Bg_s[i, j] = T.if_then_else(
                    (bn * block_N + i < Is) and (kk * block_K + j < H), Wg[bn * block_N + i, kk * block_K + j], T.cast(0, "float16")
                )
            for i, j in T.Parallel(block_N, block_K):
                Bu_s[i, j] = T.if_then_else(
                    (bn * block_N + i < Is) and (kk * block_K + j < H), Wu[bn * block_N + i, kk * block_K + j], T.cast(0, "float16")
                )
            for i, j in T.Parallel(block_N, block_K):
                Bs_s[i, j] = T.if_then_else((i == 0) and (kk * block_K + j < H), Ws_g[0, kk * block_K + j], T.cast(0, "float16"))
            T.gemm(A_s, Bg_s, Cg, transpose_B=True)
            T.gemm(A_s, Bu_s, Cu, transpose_B=True)
            T.gemm(A_s, Bs_s, Cs, transpose_B=True)
        T.copy(Cg, Cg_sh)
        T.copy(Cu, Cu_sh)
        T.copy(Cs, Cs_sh)
        for i, j in T.Parallel(block_M, block_N):
            if bm * block_M + i < T_ and bn * block_N + j < Is:
                gg = Cg_sh[i, j]
                sig = 1.0 / (1.0 + T.exp(-Cs_sh[i, 0]))
                mid[bm * block_M + i, bn * block_N + j] = T.cast((gg / (1 + T.exp(-gg))) * Cu_sh[i, j] * sig, "float16")


@tilelang.jit(execution_backend="tvm_ffi")
def _dense_gemm_frag(A, B, C, M, N, K, block_M, block_N, block_K, threads):
    """Fragment-C direct simdgroup_store to global (zero
    cross-warp communication; must stay unchanged by the fix)."""
    A: T.Tensor((M, K), "float16")
    B: T.Tensor((N, K), "float16")
    C: T.Tensor((M, N), "float32")
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_s = T.alloc_shared((block_M, block_K), "float16")
        B_s = T.alloc_shared((block_N, block_K), "float16")
        C_l = T.alloc_fragment((block_M, block_N), "float32")
        T.clear(C_l)
        for kk in T.serial(T.ceildiv(K, block_K)):
            for i, j in T.Parallel(block_M, block_K):
                A_s[i, j] = T.if_then_else(
                    (by * block_M + i < M) and (kk * block_K + j < K), A[by * block_M + i, kk * block_K + j], T.cast(0, "float16")
                )
            for i, j in T.Parallel(block_N, block_K):
                B_s[i, j] = T.if_then_else(
                    (bx * block_N + i < N) and (kk * block_K + j < K), B[bx * block_N + i, kk * block_K + j], T.cast(0, "float16")
                )
            T.gemm(A_s, B_s, C_l, transpose_B=True)
        T.copy(C_l, C[by * block_M, bx * block_N])


def _msl_facts(kernel):
    src = kernel.get_kernel_source()
    lines = src.splitlines()
    last_bar = max((i for i, line in enumerate(lines) if "threadgroup_barrier" in line), default=-1)
    last_store = max((i for i, line in enumerate(lines) if "simdgroup_store" in line), default=-1)
    return dict(
        src=src,
        barriers=src.count("threadgroup_barrier"),
        last_store=last_store,
        last_bar=last_bar,
        bar_after_store=(last_bar > last_store),
    )


def _staged_inputs(seed=11):
    dev = "mps"
    T_, Is, H = [int(v) for v in STAGED_SHAPE.split("_")]
    torch.manual_seed(seed)
    x = (torch.randn(T_, H) * 0.5).to(torch.float16).to(dev)
    Wg = (torch.randn(Is, H) * 0.05).to(torch.float16).to(dev)
    Wu = (torch.randn(Is, H) * 0.05).to(torch.float16).to(dev)
    Ws = (torch.randn(1, H) * 0.05).to(torch.float16).to(dev)
    xf = x.float().cpu()
    ref = (torch.nn.functional.silu(xf @ Wg.float().cpu().T) * (xf @ Wu.float().cpu().T) * torch.sigmoid(xf @ Ws.float().cpu().T)).to(
        torch.float16
    )
    return x, Wg, Wu, Ws, ref, T_, Is, H


def _check_staged_threads(threads, n_rounds=N_ROUNDS, seed=11):
    """Run the staged case with `threads` simdgroup threads and check the
    full criterion (err per round, determinism, MSL barrier position)."""
    x, Wg, Wu, Ws, ref, T_, Is, H = _staged_inputs(seed)
    dev = "mps"
    bn = 128 if threads > 32 else 64
    k = _staged_msg.compile(x, Wg, Wu, Ws, torch.zeros(T_, Is, dtype=torch.float16, device=dev), T_, Is, H, 8, bn, 32, threads)
    facts = _msl_facts(k)
    fails = 0
    maxerr = 0.0
    det_ok = True
    prev = None
    for _ in range(n_rounds):
        # Use a fresh output buffer every round.
        mid = torch.empty(T_, Is, dtype=torch.float16, device=dev)
        k(x, Wg, Wu, Ws, mid)
        torch.mps.synchronize()
        err = (mid.cpu() - ref).abs().max().item()
        maxerr = max(maxerr, err)
        if prev is not None:
            det_ok = det_ok and torch.equal(mid.cpu(), prev)
        prev = mid.cpu()
        if err > MAX_ERR:
            fails += 1
    assert fails == 0, f"staged t{threads}: {fails}/{n_rounds} rounds wrong, maxerr={maxerr:.3e} (threshold {MAX_ERR})"
    assert maxerr < MAX_ERR, f"staged t{threads}: maxerr={maxerr:.3e}"
    assert det_ok, f"staged t{threads}: run-to-run divergence detected"
    assert facts["bar_after_store"], (
        f"staged t{threads}: no threadgroup_barrier after the last "
        f"simdgroup_store (last_store={facts['last_store']}, "
        f"last_bar={facts['last_bar']})"
    )
    return facts


def test_codegen_staged_t256_barrier_after_store():
    """Static MSL evidence (no GPU needed): t256 staged epilogue must have a
    barrier after the last simdgroup_store (pre-fix: 2 barriers, none after
    the stores at lines ~132-137; post-fix: 4, barrier at line 139)."""
    x, Wg, Wu, Ws, _, T_, Is, H = _staged_inputs(seed=11)
    k = _staged_msg.compile(x, Wg, Wu, Ws, torch.zeros(T_, Is, dtype=torch.float16, device="mps"), T_, Is, H, 8, 128, 32, 256)
    facts = _msl_facts(k)
    assert facts["bar_after_store"], (
        f"bar_after_store={facts['bar_after_store']} (last_store={facts['last_store']}, last_bar={facts['last_bar']})"
    )
    assert facts["barriers"] >= 3, f"barriers={facts['barriers']}"


def test_codegen_frag_t256_unchanged_barriers():
    """The direct fragment-C -> global path must keep
    the zero-communication barrier structure (2 barriers, all in the kk
    loop, none added around the final simdgroup_store)."""
    dev = "mps"
    M, N, K = 64, 2048, 512
    A = torch.zeros(M, K, dtype=torch.float16, device=dev)
    B = torch.zeros(N, K, dtype=torch.float16, device=dev)
    C = torch.zeros(M, N, dtype=torch.float32, device=dev)
    k = _dense_gemm_frag.compile(A, B, C, M, N, K, 8, 128, 32, 256)
    facts = _msl_facts(k)
    assert facts["barriers"] == 2, f"frag t256 barriers={facts['barriers']}"


@tilelang.testing.requires_metal
def test_correctness_staged_t256():
    """Dynamic: t256 staged epilogue, 12 rounds fresh buffer, all rounds must
    be < 1e-2 vs fp32 ref and run-to-run bit-identical. Pre-fix: 12/12 wrong
    (maxerr=5.969)."""
    facts = _check_staged_threads(256)
    assert facts["barriers"] >= 3


@tilelang.testing.requires_metal
def test_correctness_staged_t32():
    """Dynamic: t32 (single simdgroup) staged epilogue must stay correct
    (pre-fix baseline remains below the race-detection threshold)."""
    _check_staged_threads(32)


@tilelang.testing.requires_metal
def test_correctness_frag_t256_bit_identical():
    """dense_gemm_frag t256 direct-to-global must
    stay bit-identical to the fp32 reference and keep barriers==2."""
    dev = "mps"
    M, N, K = 64, 2048, 512
    torch.manual_seed(7)
    A = (torch.randn(M, K) * 0.5).to(torch.float16).to(dev)
    B = (torch.randn(N, K) * 0.05).to(torch.float16).to(dev)
    C = torch.empty(M, N, dtype=torch.float32, device=dev)
    k = _dense_gemm_frag.compile(A, B, C, M, N, K, 8, 128, 32, 256)
    k(A, B, C)
    torch.mps.synchronize()
    ref = (A.float() @ B.float().T).cpu()
    assert torch.equal(C.cpu(), ref), f"frag t256 diverged from fp32 reference (maxerr={(C.cpu() - ref).abs().max().item():.3e})"
    assert _msl_facts(k)["barriers"] == 2


# ---------------------------------------------------------------------------
# T.access_ptr (tvm_access_ptr) pointer-form coverage
# ---------------------------------------------------------------------------
# The Metal macro / tensor-intrin path builds simdgroup_store/load pointer
# arguments as T.access_ptr(...) (lowered to tvm_access_ptr with a
# single-element default extent). Previously these were recorded as
# plain single-element pointer accesses and PointerAccessIsDisjoint could
# prove two *base elements* disjoint while the 8x8 tile footprints overlap
# (here: warp 0 stores the tile at sh[1, 0] covering rows 1..8; warp 1
# loads the tile at sh[8, 0] covering rows 8..15; the tiles overlap at
# row 8 while the base offsets 1*stride and 8*stride are disjoint).
#
# Kernel layout (threads=64, 2 simdgroups): each warp owns one 8x8
# simdgroup matrix in a per-warp metal.simdgroup local buffer. The shared
# buffer is seeded with zeros, warp 0 produces/consumes the sh[1, 0] tile
# and warp 1 consumes/produces the sh[8, 0] tile; results are read back to
# global with a per-warp base (out[0, warp*8]).
#
# - RAW kernel: warp 0 simdgroup_store (access_ptr "w") then warp 1
#   simdgroup_load (access_ptr "r") -> barrier must separate them, and
#   out[0, 8:16] must equal x[7, 0:8] (the overlap row of the stored tile).
# - WAR kernel: warp 0 simdgroup_load (access_ptr "r") then warp 1
#   simdgroup_store (access_ptr "w") -> barrier must separate them, and
#   out[:, 0:8] must stay all-zero (the load sees the seed).
#
# Variants: transpose in {False, True} x shared stride in {8, 16}
# (padded stride: row stride 16 with an 8-column tile footprint).

AP_CONFIGS = ((0, 8), (0, 16), (1, 8), (1, 16))  # (transpose, stride)


@tilelang.jit(execution_backend="tvm_ffi")
def _access_ptr_raw(x, out, transpose, stride, threads=64):
    """Cross-warp RAW through the T.access_ptr pointer form: warp 0 stores
    the tile at sh[1, 0] (rows 1..8), warp 1 loads the tile at sh[8, 0]
    (rows 8..15). The footprints overlap at row 8 while the base elements
    are disjoint, so the pre-fix single-element disjointness proof skipped
    the RAW barrier."""
    x: T.Tensor((8, 16), "float32")
    out: T.Tensor((8, 16), "float32")
    with T.Kernel(1, threads=threads) as (_bx,):
        sh = T.alloc_shared((16, stride), "float32")
        C_l = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        C2 = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        for i, j in T.Parallel(16, stride):
            sh[i, j] = 0.0
        T.simdgroup_load(C_l.data, 0, T.access_ptr(x[0, T.get_thread_binding() // 32 * 8], "r"), 16, 8, 8, T.bool(transpose))
        T.make_filled_simdgroup_matrix(C2.data, 0, T.cast(0.0, "float32"))
        if T.get_thread_binding() < 32:
            T.simdgroup_store(C_l.data, 0, T.access_ptr(sh[1, 0], "w"), stride, 8, 8, T.bool(transpose))
        if T.get_thread_binding() >= 32:
            T.simdgroup_load(C2.data, 0, T.access_ptr(sh[8, 0], "r"), stride, 8, 8, T.bool(transpose))
        T.simdgroup_store(C2.data, 0, T.access_ptr(out[0, T.get_thread_binding() // 32 * 8], "w"), 16, 8, 8, T.bool(transpose))


@tilelang.jit(execution_backend="tvm_ffi")
def _access_ptr_war(x, out, transpose, stride, threads=64):
    """Cross-warp WAR through the T.access_ptr pointer form: warp 0 loads
    the tile at sh[1, 0] (rows 1..8), warp 1 stores the tile at sh[8, 0]
    (rows 8..15). The footprints overlap at row 8; a barrier must separate
    the load from the store."""
    x: T.Tensor((8, 16), "float32")
    out: T.Tensor((8, 16), "float32")
    with T.Kernel(1, threads=threads) as (bx,):
        sh = T.alloc_shared((16, stride), "float32")
        C_l = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        C2 = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        for i, j in T.Parallel(16, stride):
            sh[i, j] = 0.0
        T.simdgroup_load(C_l.data, 0, T.access_ptr(x[0, T.get_thread_binding() // 32 * 8], "r"), 16, 8, 8, T.bool(transpose))
        T.make_filled_simdgroup_matrix(C2.data, 0, T.cast(0.0, "float32"))
        if T.get_thread_binding() < 32:
            T.simdgroup_load(C2.data, 0, T.access_ptr(sh[1, 0], "r"), stride, 8, 8, T.bool(transpose))
        if T.get_thread_binding() >= 32:
            T.simdgroup_store(C_l.data, 0, T.access_ptr(sh[8, 0], "w"), stride, 8, 8, T.bool(transpose))
        T.simdgroup_store(C2.data, 0, T.access_ptr(out[0, T.get_thread_binding() // 32 * 8], "w"), 16, 8, 8, T.bool(transpose))


@tilelang.jit(execution_backend="tvm_ffi")
def _access_ptr_waw(out, stride, threads=64):
    """Two sequential simdgroups store overlapping 8x8 shared tiles.

    The first tile covers rows 1..8 and the second covers rows 8..15, so
    row 8 requires a barrier between the stores. The final load keeps the
    shared-memory writes observable in generated code.
    """
    out: T.Tensor((8, 16), "float32")
    with T.Kernel(1, threads=threads) as (bx,):
        sh = T.alloc_shared((16, stride), "float32")
        first = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        second = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        loaded = T.alloc_local((64,), "float32", scope="metal.simdgroup")
        T.make_filled_simdgroup_matrix(first.data, 0, T.cast(1.0, "float32"))
        T.make_filled_simdgroup_matrix(second.data, 0, T.cast(2.0, "float32"))
        if T.get_thread_binding() < 32:
            T.simdgroup_store(first.data, 0, T.access_ptr(sh[1, 0], "w"), stride, 8, 8, T.bool(False))
        if T.get_thread_binding() >= 32:
            T.simdgroup_store(second.data, 0, T.access_ptr(sh[8, 0], "w"), stride, 8, 8, T.bool(False))
        T.simdgroup_load(loaded.data, 0, T.access_ptr(sh[8, 0], "r"), stride, 8, 8, T.bool(False))
        T.simdgroup_store(loaded.data, 0, T.access_ptr(out[0, T.get_thread_binding() // 32 * 8], "w"), 16, 8, 8, T.bool(False))


def _msl_barrier_between(src, first_marker, second_marker):
    """True if a threadgroup_barrier line exists between the last
    first_marker line that precedes some second_marker line and the first
    second_marker line after it."""
    lines = src.splitlines()
    bars = [i for i, line in enumerate(lines) if "threadgroup_barrier" in line]
    first = [i for i, line in enumerate(lines) if first_marker in line]
    second = [i for i, line in enumerate(lines) if second_marker in line]
    first_before = [i for i in first if any(j > i for j in second)]
    if not first_before:
        return False
    last_first = max(first_before)
    second_after = [i for i in second if i > last_first]
    if not second_after:
        return False
    return any(last_first < b < min(second_after) for b in bars)


def _msl_barrier_between_first_two(src, marker):
    lines = src.splitlines()
    positions = [i for i, line in enumerate(lines) if marker in line]
    assert len(positions) >= 2, f"expected at least two {marker} occurrences"
    return any("threadgroup_barrier" in line for line in lines[positions[0] + 1 : positions[1]])


def _access_ptr_inputs(seed=3):
    dev = "mps"
    torch.manual_seed(seed)
    x = (torch.randn(8, 16) * 0.5).to(torch.float32).to(dev)
    exp_raw = torch.zeros(8, 16, dtype=torch.float32, device=dev)
    # overlap row: warp 1's loaded tile row 0 = sh[8, 0:8] = the stored
    # tile's last row = x[7, 0:8] (identical for both transpose layouts:
    # the transpose chain maps the overlap row to out[0, 8:16] the same
    # way). warp 0's half is the initialized zero matrix.
    exp_raw[0, 8:16] = x[7, 0:8]
    exp_war = torch.zeros(8, 16, dtype=torch.float32, device=dev)
    return x, exp_raw, exp_war


def test_codegen_access_ptr_raw_barrier():
    """Static: a threadgroup_barrier must separate the access_ptr-form
    simdgroup_store (warp 0) from the overlapping simdgroup_load (warp 1)
    for normal/transpose x stride 8/16. The single-element disjointness
    proof must not hide the full tile footprint."""
    dev = "mps"
    x = torch.randn(8, 16, dtype=torch.float32, device=dev)
    out = torch.empty(8, 16, dtype=torch.float32, device=dev)
    for transpose, stride in AP_CONFIGS:
        k = _access_ptr_raw.compile(x, out, transpose, stride, 64)
        assert _msl_barrier_between(k.get_kernel_source(), "simdgroup_store", "simdgroup_load"), (
            f"RAW access_ptr t{transpose} s{stride}: no barrier between simdgroup_store and simdgroup_load"
        )


def test_codegen_access_ptr_war_barrier():
    """Static: a threadgroup_barrier must separate the access_ptr-form
    simdgroup_load (warp 0) from the overlapping simdgroup_store (warp 1)
    for normal/transpose x stride 8/16."""
    dev = "mps"
    x = torch.randn(8, 16, dtype=torch.float32, device=dev)
    out = torch.empty(8, 16, dtype=torch.float32, device=dev)
    for transpose, stride in AP_CONFIGS:
        k = _access_ptr_war.compile(x, out, transpose, stride, 64)
        assert _msl_barrier_between(k.get_kernel_source(), "simdgroup_load", "simdgroup_store"), (
            f"WAR access_ptr t{transpose} s{stride}: no barrier between simdgroup_load and simdgroup_store"
        )


@pytest.mark.parametrize("stride", (8, 16))
def test_codegen_access_ptr_overlapping_waw_barrier(stride):
    """Overlapping tile stores must be ordered for compact and padded rows."""
    out = torch.empty(8, 16, dtype=torch.float32, device="mps")
    kernel = _access_ptr_waw.compile(out, stride, 64)
    assert _msl_barrier_between_first_two(kernel.get_kernel_source(), "simdgroup_store"), (
        f"overlapping WAW access_ptr stride={stride}: no barrier between tile stores"
    )


@tilelang.testing.requires_metal
def test_correctness_access_ptr_raw_war():
    """Dynamic: all 8 access_ptr configs x N_ROUNDS, fresh output buffers
    per round, bit-identical vs the exact fp32 reference and run-to-run
    deterministic. RAW: out[0, 8:16] == x[7, 0:8]; WAR: all zeros."""
    x, exp_raw, exp_war = _access_ptr_inputs()
    for transpose, stride in AP_CONFIGS:
        for kind, kern, exp in (("raw", _access_ptr_raw, exp_raw), ("war", _access_ptr_war, exp_war)):
            initial_out = torch.empty(8, 16, dtype=torch.float32, device="mps")
            compiled = kern.compile(x, initial_out, transpose, stride, 64)
            prev = None
            for _ in range(N_ROUNDS):
                out = torch.empty(8, 16, dtype=torch.float32, device="mps")
                compiled(x, out)
                torch.mps.synchronize()
                got = out.cpu()
                ref = exp.cpu()
                assert torch.equal(got, ref), (
                    f"access_ptr {kind} t{transpose} s{stride} diverged (maxerr={(got - ref).abs().max().item():.3e})"
                )
                if prev is not None:
                    assert torch.equal(got, prev), f"access_ptr {kind} t{transpose} s{stride}: run-to-run divergence"
                prev = got


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
