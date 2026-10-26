"""``T.sync_threads(barrier_id)`` without a thread count synchronizes the whole
CTA on the CuTeDSL target, as ``tl::__sync_thread_partial(id)`` does on CUDA."""

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


def _rotate_through_shared(threads, barrier):
    @T.prim_func
    def main(Out: T.Tensor((4, threads), T.int32), bid: T.int32):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            S = T.alloc_shared((threads,), T.int32)
            for r in T.serial(4):
                S[tx] = tx * 10 + r
                if barrier == "static":
                    T.sync_threads(1)
                else:
                    T.sync_threads(bid)
                Out[r, tx] = S[(tx + 32 * (r + 1)) % threads]
                T.sync_threads(2, threads)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("barrier", ["static", "dynamic"])
def test_sync_threads_barrier_id_without_count(barrier):
    _require_cutedsl()
    threads = 256
    kernel = tilelang.compile(_rotate_through_shared(threads, barrier), target="cutedsl")
    out = torch.zeros(4, threads, device="cuda", dtype=torch.int32)
    kernel(out, 3)
    tx = torch.arange(threads, device="cuda", dtype=torch.int32)
    expected = torch.stack([((tx + 32 * (r + 1)) % threads) * 10 + r for r in range(4)])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
