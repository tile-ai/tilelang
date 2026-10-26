"""CuTeDSL Philox4x32-10 RNG, checked against the CUDA backend."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def rand_kernel(M=1024, seed=42, generator="curandStatePhilox4_32_10_t"):
    num_per_thread = 128
    threads = 1

    @T.prim_func
    def kernel(
        A: T.Tensor((M,), "uint32"),
        B: T.Tensor((M,), "float32"),
        C: T.Tensor((M,), "float64"),
        D: T.Tensor((M,), "float32"),
        E: T.Tensor((M,), "float64"),
    ):
        with T.Kernel(T.ceildiv(M, threads * num_per_thread), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx * num_per_thread + tx * num_per_thread, generator=generator)
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                if idx < M:
                    A[idx] = T.rng_rand()
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                if idx < M:
                    B[idx] = T.rng_rand_float()
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                if idx < M:
                    C[idx] = T.rng_rand_float(bit=64)
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                if idx < M:
                    D[idx] = T.rng_rand_float(dist="normal")
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                if idx < M:
                    E[idx] = T.rng_rand_float(bit=64, dist="normal")

    return kernel


def _run(target, M, seed, generator="curandStatePhilox4_32_10_t"):
    mod = tilelang.compile(rand_kernel(M, seed, generator), target=target)
    out = [
        torch.empty(M, dtype=torch.uint32, device="cuda"),
        torch.empty(M, dtype=torch.float32, device="cuda"),
        torch.empty(M, dtype=torch.float64, device="cuda"),
        torch.empty(M, dtype=torch.float32, device="cuda"),
        torch.empty(M, dtype=torch.float64, device="cuda"),
    ]
    mod(*out)
    return out


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("M, seed", [(1024, 42), (512, 123)])
def test_rand_matches_cuda(M, seed):
    _require_cutedsl()
    ref = _run("cuda", M, seed)
    got = _run("cutedsl", M, seed)

    # Integers, uniform floats and the double uniform are bit-identical.
    assert torch.equal(got[0], ref[0]), "Philox integers differ from the CUDA backend"
    assert torch.equal(got[1].view(torch.int32), ref[1].view(torch.int32)), "uniform float32 differs"
    assert torch.equal(got[2].view(torch.int64), ref[2].view(torch.int64)), "uniform float64 differs"

    # Normals run the same Box-Muller transform over the same samples.
    assert torch.equal(got[3].view(torch.int32), ref[3].view(torch.int32)), "normal float32 differs"
    # curand's double pair uses libdevice's sincospi, which CuTeDSL does not
    # expose; sin(pi * v) differs from it by a few ulps on some architectures.
    torch.testing.assert_close(got[4], ref[4], rtol=1e-13, atol=1e-13)


@tilelang.testing.requires_cuda
def test_double_normal_keeps_the_stream():
    _require_cutedsl()
    M, seed = 64, 42
    assert torch.equal(
        _tail_after_double_normal("cutedsl", M, seed),
        _tail_after_double_normal("cuda", M, seed),
    ), "integer draws after double normals diverged from the CUDA backend"


def tail_kernel(M, seed=42):
    """Draw double normals, then integers: the integers pin the generator position."""

    @T.prim_func
    def kernel(
        F: T.Tensor((M,), "float64"),
        A: T.Tensor((M,), "uint32"),
    ):
        with T.Kernel(1, threads=1):
            T.rng_init(seed, 0, 0)
            for i in T.Parallel(M):
                F[i] = T.rng_rand_float(bit=64, dist="normal")
            for i in T.Parallel(M):
                A[i] = T.rng_rand()

    return kernel


def _tail_after_double_normal(target, M, seed):
    mod = tilelang.compile(tail_kernel(M, seed), target=target)
    F = torch.zeros(M, dtype=torch.float64, device="cuda")
    A = torch.zeros(M, dtype=torch.uint32, device="cuda")
    mod(F, A)
    return A


@tilelang.testing.requires_cuda
def test_rand_unsupported_generator():
    _require_cutedsl()
    # The device kernel is only traced when the module is launched.
    with pytest.raises(NotImplementedError, match="Philox4x32-10"):
        _run("cutedsl", 128, 0, generator="curandStateXORWOW_t")


if __name__ == "__main__":
    tilelang.testing.main()
