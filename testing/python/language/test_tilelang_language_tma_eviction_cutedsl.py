"""CuTeDSL TMA cache-eviction hints on plain tile loads and stores."""

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


def _roundtrip_kernel(N: int, load_policy: str, store_policy: str):

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((N,), "float32")
            T.copy(A, s, eviction_policy=load_policy)
            T.copy(s, B, eviction_policy=store_policy)

    return kernel


@pytest.mark.parametrize(
    "load_policy, store_policy",
    [("evict_first", "evict_last"), ("evict_last", "evict_first")],
)
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_eviction_cutedsl(load_policy, store_policy):
    _require_cutedsl()
    N = 1024
    mod = tilelang.compile(_roundtrip_kernel(N, load_policy, store_policy), out_idx=[1], target="cutedsl")
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = mod(A)
    torch.testing.assert_close(B, A)


if __name__ == "__main__":
    tilelang.testing.main()
