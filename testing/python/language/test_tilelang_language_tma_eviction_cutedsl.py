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
    """Explicit TMA copies, since a plain T.copy of a small tile may not use TMA."""

    @T.prim_func
    def kernel(
        A: T.Tensor((N, N), "float32"),
        B: T.Tensor((N, N), "float32"),
    ):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((N, N), "float32")
            mbar = T.alloc_barrier(128)
            T.tma_copy(A, s, barrier=mbar, eviction_policy=load_policy)
            T.barrier_arrive(mbar)
            T.barrier_wait(mbar, 0)
            T.tma_copy(s, B, eviction_policy=store_policy)
            T.tma_store_wait(0)

    return kernel


@pytest.mark.parametrize(
    "load_policy, store_policy",
    [("evict_first", "evict_last"), ("evict_last", "evict_first")],
)
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_eviction_cutedsl(load_policy, store_policy):
    _require_cutedsl()
    N = 32
    mod = tilelang.compile(_roundtrip_kernel(N, load_policy, store_policy), out_idx=[1], target="cutedsl")

    # Correctness alone would hold if the hint were dropped, so also check that
    # each policy reaches its own TMA call.
    source = mod.get_kernel_source()
    load_call = next(line for line in source.splitlines() if "tl.tma_load(" in line)
    store_call = next(line for line in source.splitlines() if "tl.tma_store(" in line)
    load_kind = load_policy.removeprefix("evict_").upper()
    store_kind = store_policy.removeprefix("evict_").upper()
    assert f'eviction_kind="EVICT_{load_kind}"' in load_call, load_call
    assert f'eviction_kind="EVICT_{store_kind}"' in store_call, store_call

    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = mod(A)
    torch.testing.assert_close(B, A)


if __name__ == "__main__":
    tilelang.testing.main()
