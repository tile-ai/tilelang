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


# --- RNG hardening regressions (#2625 / #3029 / #3034) ---

_NO_INIT_MATCH = re.escape("requires a preceding `T.rng_init(...)`")
# Binding the void `rng_init` result is rejected either by the eager frontend
# ("value-less expression") or, on paths that bypass it, by the CUDA codegen
# ("cannot be bound to variable"). Either diagnostic is acceptable; the contract
# is that malformed CUDA never reaches nvcc.
_VOID_BINDING_MATCH = "value-less expression|cannot be bound to variable"


def _curand_init_line(source: str) -> str:
    """The generated `curand_init(...)` call, which carries the effective seq."""
    for line in source.splitlines():
        if "curand_init(" in line:
            return line
    raise AssertionError(f"no curand_init in generated source:\n{source}")


def _default_seq_2d_threads(seed=1234):
    @T.prim_func
    def rand_kernel(Out: T.Tensor((2, 2), "int32")):
        with T.Kernel(1, threads=(2, 2)):
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            T.rng_init(seed)
            Out[tx, ty] = T.reinterpret(T.rng_rand(), dtype="int32")

    return rand_kernel


def _default_seq_2d_grid(seed=1234):
    @T.prim_func
    def rand_kernel(Out: T.Tensor((2, 2), "int32")):
        with T.Kernel(2, 2, threads=1):
            bx = T.get_block_binding(0)
            by = T.get_block_binding(1)
            T.rng_init(seed)
            Out[bx, by] = T.reinterpret(T.rng_rand(), dtype="int32")

    return rand_kernel


def _default_seq_1d(M=64, threads=32, seed=1234):
    @T.prim_func
    def rand_kernel(Out: T.Tensor((M,), "int32")):
        with T.Kernel(T.ceildiv(M, threads), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed)
            idx = bx * threads + tx
            if idx < M:
                Out[idx] = T.reinterpret(T.rng_rand(), dtype="int32")

    return rand_kernel


def _explicit_seq_1d(M=64, threads=32, seed=1234):
    @T.prim_func
    def rand_kernel(Out: T.Tensor((M,), "int32")):
        with T.Kernel(T.ceildiv(M, threads), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, seq=tx + bx * threads)
            idx = bx * threads + tx
            if idx < M:
                Out[idx] = T.reinterpret(T.rng_rand(), dtype="int32")

    return rand_kernel


def _explicit_seq_2d_threads(seed=1234):
    @T.prim_func
    def rand_kernel(Out: T.Tensor((2, 2), "int32")):
        with T.Kernel(1, threads=(2, 2)):
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            T.rng_init(seed, seq=tx * 2 + ty)
            Out[tx, ty] = T.reinterpret(T.rng_rand(), dtype="int32")

    return rand_kernel


@tilelang.testing.requires_cuda
def test_rand_default_seq_covers_thread_y():
    """#2625: the default seq must fold in threadIdx.y, not only threadIdx.x."""
    kernel = tilelang.compile(_default_seq_2d_threads())
    assert "threadIdx.y" in _curand_init_line(kernel.get_kernel_source())

    out = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    kernel(out)
    values = out.cpu().flatten().tolist()
    assert len(set(values)) == 4, f"threadIdx.y shares one curand stream: {values}"


@tilelang.testing.requires_cuda
def test_rand_default_seq_covers_block_y():
    """#2625: the default seq must fold in blockIdx.y, not only blockIdx.x."""
    kernel = tilelang.compile(_default_seq_2d_grid())
    assert "blockIdx.y" in _curand_init_line(kernel.get_kernel_source())

    out = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    kernel(out)
    values = out.cpu().flatten().tolist()
    assert len(set(values)) == 4, f"blockIdx.y shares one curand stream: {values}"


@tilelang.testing.requires_cuda
def test_rand_default_seq_keeps_1d_layout():
    """A 1-D launch keeps the historical `threadIdx.x + blockIdx.x * blockDim.x`."""
    default = tilelang.compile(_default_seq_1d())
    explicit = tilelang.compile(_explicit_seq_1d())

    a = torch.zeros(64, dtype=torch.int32, device="cuda")
    b = torch.zeros(64, dtype=torch.int32, device="cuda")
    default(a)
    explicit(b)
    assert torch.equal(a, b), "the default seq changed the 1-D stream layout"

    line = _curand_init_line(default.get_kernel_source())
    assert "threadIdx.y" not in line and "blockIdx.y" not in line


@tilelang.testing.requires_cuda
def test_rand_explicit_seq_is_used_verbatim():
    """An explicit seq is forwarded unchanged, with no implicit flattening."""
    kernel = tilelang.compile(_explicit_seq_2d_threads())
    line = _curand_init_line(kernel.get_kernel_source())
    assert "threadIdx.y" in line and "blockIdx" not in line

    out = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    kernel(out)
    assert len(set(out.cpu().flatten().tolist())) == 4


# Each builder inlines its draw: the parser only resolves module globals, so a
# `draw` callback captured in an enclosing closure would raise NameError first.
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


def _bind_init_result_kernel(seed=1234):
    @T.prim_func
    def rand_kernel(Out: T.Tensor((8,), "uint32")):
        with T.Kernel(1, threads=8):
            tx = T.get_thread_binding()
            _state = T.rng_init(seed)
            Out[tx] = T.rng_rand()

    return rand_kernel


@tilelang.testing.requires_cuda
def test_rand_init_result_cannot_be_bound():
    """#3029: binding the void rng_init result is rejected before nvcc."""
    with pytest.raises(Exception, match=_VOID_BINDING_MATCH):
        tilelang.compile(_bind_init_result_kernel())


if __name__ == "__main__":
    tilelang.testing.main()
