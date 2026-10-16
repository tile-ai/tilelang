import re

import tilelang
import tilelang.language as T
import torch
import pytest
import tilelang.testing


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42, generator="curandStatePhilox4_32_10_t"):
    num_per_thread = 128
    threads = 1
    blk_M = num_per_thread * threads

    @T.prim_func
    def rand_kernel(
        A: T.Tensor((M,), "uint32"),
        B: T.Tensor((M,), "float32"),
        C: T.Tensor((M,), "float64"),
        D: T.Tensor((M,), "float32"),
        E: T.Tensor((M,), "float64"),
    ):
        with T.Kernel(T.ceildiv(M, threads * num_per_thread), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx * blk_M + tx * num_per_thread, generator=generator)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    A[idx] = T.rng_rand()
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    B[idx] = T.rng_rand_float()
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    C[idx] = T.rng_rand_float(bit=64)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    D[idx] = T.rng_rand_float(dist="normal")
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    E[idx] = T.rng_rand_float(bit=64, dist="normal")

    return rand_kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, seed, generator", [(1024, 42, "curandStateMRG32k3a_t"), (512, 123, "curandStatePhilox4_32_10_t"), (128, 0, "curandStateXORWOW_t")]
)
def test_rand_1d(M, seed, generator):
    kernel = tilelang_rand_1d(M, seed, generator)
    A = torch.empty(M, dtype=torch.uint32, device="cuda")
    B = torch.empty(M, dtype=torch.float32, device="cuda")
    C = torch.empty(M, dtype=torch.float64, device="cuda")
    D = torch.empty(M, dtype=torch.float32, device="cuda")
    E = torch.empty(M, dtype=torch.float64, device="cuda")
    kernel(A, B, C, D, E)


@tilelang.jit
def tilelang_rand_blockwise(M=64, seed=42, generator="curandStatePhilox4_32_10_t"):
    threads = 32

    @T.prim_func
    def rand_kernel(A: T.Tensor((M,), "uint32")):
        with T.Kernel(M, threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx, generator=generator)
            if tx == 0:
                A[bx] = T.rng_rand()

    return rand_kernel


@tilelang.jit
def tilelang_rand_guarded_cumsum(M=64, seed=42, generator="curandStatePhilox4_32_10_t"):
    threads = 32

    @T.prim_func
    def rand_kernel(
        A: T.Tensor((M,), "uint32"),
        n: T.int32,
    ):
        with T.Kernel(M, threads=threads) as bx:
            tx = T.get_thread_binding()
            s = T.alloc_shared((threads,), "int32")
            # rng_init inside a runtime guard, with a shared-memory cumsum
            # between init and use: sync legalization hoists __syncthreads()
            # out of the guard and splits it into sibling blocks, so the
            # curand state must be declared at function scope to stay visible.
            if bx < n:
                T.rng_init(seed, 0, bx, generator=generator)
                s[tx] = 1
                T.cumsum(s, dim=0)
                if tx == 0:
                    A[bx] = T.rng_rand() + T.cast(s[threads - 1], "uint32") * 0

    return rand_kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("generator", ["curandStateMRG32k3a_t", "curandStatePhilox4_32_10_t", "curandStateXORWOW_t"])
def test_rand_init_in_split_guard(generator):
    M, seed, n = 64, 42, 37
    guarded = tilelang_rand_guarded_cumsum(M, seed, generator)
    baseline = tilelang_rand_blockwise(M, seed, generator)

    sentinel = 0xDEADBEEF
    A = torch.full((M,), sentinel, dtype=torch.uint32, device="cuda")
    guarded(A, n)
    A_ref = torch.empty(M, dtype=torch.uint32, device="cuda")
    baseline(A_ref)

    assert torch.equal(A[:n], A_ref[:n]), "guarded rng output differs from unguarded baseline"
    assert (A[n:] == sentinel).all(), "rows outside the guard must stay untouched"



# --- RNG hardening regressions (#3034) ---

_NO_INIT_MATCH = re.escape("requires a preceding `T.rng_init(...)`")


def _no_init_rng_rand():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "uint32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            Out[tx] = T.rng_rand()

    return rand_kernel


def _init_rng_rand():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "uint32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            T.rng_init(1234)
            Out[tx] = T.rng_rand()

    return rand_kernel


def _no_init_rng_rand_float():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            Out[tx] = T.rng_rand_float()

    return rand_kernel


def _init_rng_rand_float():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            T.rng_init(1234)
            Out[tx] = T.rng_rand_float()

    return rand_kernel


def _no_init_rng_rand_float_bit64():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float64")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            Out[tx] = T.rng_rand_float(bit=64)

    return rand_kernel


def _init_rng_rand_float_bit64():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float64")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            T.rng_init(1234)
            Out[tx] = T.rng_rand_float(bit=64)

    return rand_kernel


def _no_init_rng_rand_float_normal():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            Out[tx] = T.rng_rand_float(dist="normal")

    return rand_kernel


def _init_rng_rand_float_normal():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            T.rng_init(1234)
            Out[tx] = T.rng_rand_float(dist="normal")

    return rand_kernel


def _no_init_rng_rand_float_bit64_normal():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float64")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            Out[tx] = T.rng_rand_float(bit=64, dist="normal")

    return rand_kernel


def _init_rng_rand_float_bit64_normal():
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "float64")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            T.rng_init(1234)
            Out[tx] = T.rng_rand_float(bit=64, dist="normal")

    return rand_kernel


_PRODUCERS = [
    ("rng_rand", _no_init_rng_rand, _init_rng_rand, "uint32"),
    ("rng_rand_float", _no_init_rng_rand_float, _init_rng_rand_float, "float32"),
    ("rng_rand_float_bit64", _no_init_rng_rand_float_bit64, _init_rng_rand_float_bit64, "float64"),
    ("rng_rand_float_normal", _no_init_rng_rand_float_normal, _init_rng_rand_float_normal, "float32"),
    (
        "rng_rand_float_bit64_normal",
        _no_init_rng_rand_float_bit64_normal,
        _init_rng_rand_float_bit64_normal,
        "float64",
    ),
]
_IDS = [p[0] for p in _PRODUCERS]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("name,no_init,initialized,dtype", _PRODUCERS, ids=_IDS)
def test_rand_producer_without_init_is_rejected(name, no_init, initialized, dtype):
    """#3034: every producer form diagnoses a missing rng_init before nvcc."""
    with pytest.raises(Exception, match=_NO_INIT_MATCH):
        tilelang.compile(no_init())


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("name,no_init,initialized,dtype", _PRODUCERS, ids=_IDS)
def test_rand_producer_after_init_compiles(name, no_init, initialized, dtype):
    """Positive control: every producer form works once rng_init has run."""
    kernel = tilelang.compile(initialized())
    out = torch.zeros(8, dtype=getattr(torch, dtype), device="cuda")
    kernel(out)
    if out.is_floating_point():
        assert torch.isfinite(out).all()

if __name__ == "__main__":
    tilelang.testing.main()
