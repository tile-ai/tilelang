import pytest
import torch

import tilelang
from tilelang import tvm
import tilelang.language as T
import tilelang.testing


CUDA_SOURCE_TARGET = {"kind": "cuda", "arch": "sm_100a"}
LOAD_CACHE_POLICIES = ("ca", "cg", "cs", "lu", "cv")
STORE_CACHE_POLICIES = ("wb", "cg", "cs", "wt")


def make_cache_policy_copy(
    size=130,
    *,
    load_cache_policy=None,
    store_cache_policy=None,
    prefer_instruction="sync",
):
    @T.prim_func
    def main(A: T.Tensor((size,), T.float32), B: T.Tensor((size,), T.float32)):
        with T.Kernel(1, threads=32):
            T.copy(
                A,
                B,
                coalesced_width=2 if size > 1 else None,
                load_cache_policy=load_cache_policy,
                store_cache_policy=store_cache_policy,
                prefer_instruction=prefer_instruction,
            )

    return main


def make_all_cache_policy_copies(size=130):
    @T.prim_func
    def main(A: T.Tensor((5, size), T.float32), B: T.Tensor((5, size), T.float32)):
        with T.Kernel(1, threads=32):
            T.copy(
                A[0, :],
                B[0, :],
                coalesced_width=2,
                load_cache_policy="ca",
                store_cache_policy="wb",
                prefer_instruction="sync",
            )
            T.copy(
                A[1, :],
                B[1, :],
                coalesced_width=2,
                load_cache_policy="cg",
                store_cache_policy="cg",
                prefer_instruction="sync",
            )
            T.copy(
                A[2, :],
                B[2, :],
                coalesced_width=2,
                load_cache_policy="cs",
                store_cache_policy="cs",
                prefer_instruction="sync",
            )
            T.copy(
                A[3, :],
                B[3, :],
                coalesced_width=2,
                load_cache_policy="lu",
                store_cache_policy="wt",
                prefer_instruction="sync",
            )
            T.copy(
                A[4, :],
                B[4, :],
                coalesced_width=2,
                load_cache_policy="cv",
                prefer_instruction="sync",
            )

    return main


def lower_cuda_source(program):
    with tvm.target.Target(CUDA_SOURCE_TARGET):
        artifact = tilelang.lower(program, target=CUDA_SOURCE_TARGET)
    return artifact.kernel_source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("policy", LOAD_CACHE_POLICIES)
def test_copy_load_cache_policy_codegen(policy):
    source = lower_cuda_source(make_cache_policy_copy(load_cache_policy=policy))
    assert f"tl::load_global_cache<tl::LoadCachePolicy::k{policy.upper()}>" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("policy", STORE_CACHE_POLICIES)
def test_copy_store_cache_policy_codegen(policy):
    source = lower_cuda_source(make_cache_policy_copy(store_cache_policy=policy))
    assert f"tl::store_global_cache<tl::StoreCachePolicy::k{policy.upper()}>" in source


@tilelang.testing.requires_cuda
def test_copy_cache_policy_scalar_codegen():
    @T.prim_func
    def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=1):
            T.copy(
                A[0],
                B[0],
                load_cache_policy="cg",
                store_cache_policy="wt",
                prefer_instruction="sync",
            )

    source = lower_cuda_source(main)
    assert "tl::load_global_cache<tl::LoadCachePolicy::kCG>" in source
    assert "tl::store_global_cache<tl::StoreCachePolicy::kWT>" in source


@tilelang.testing.requires_cuda
def test_copy_cache_policy_none_preserves_default_codegen():
    source = lower_cuda_source(make_cache_policy_copy(prefer_instruction=None))
    assert "tl::load_global_cache" not in source
    assert "tl::store_global_cache" not in source


@tilelang.testing.requires_cuda
def test_copy_cache_policy_annotations_take_precedence():
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=32):
            T.copy(
                A,
                B,
                load_cache_policy="ca",
                prefer_instruction="sync",
                annotations={"load_cache_policy": "cg"},
            )

    source = lower_cuda_source(main)
    assert "tl::load_global_cache<tl::LoadCachePolicy::kCG>" in source
    assert "tl::load_global_cache<tl::LoadCachePolicy::kCA>" not in source


@tilelang.testing.requires_cuda
def test_copy_cache_policy_vectorized_tail_correctness():
    size = 130
    kernel = tilelang.compile(
        make_all_cache_policy_copies(size),
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    source = kernel.get_kernel_source()
    for policy in LOAD_CACHE_POLICIES:
        assert f"tl::load_global_cache<tl::LoadCachePolicy::k{policy.upper()}>" in source
    for policy in STORE_CACHE_POLICIES:
        assert f"tl::store_global_cache<tl::StoreCachePolicy::k{policy.upper()}>" in source

    data = torch.randn(5, size, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel(data), data)


@pytest.mark.parametrize(
    ("keyword", "policy", "message"),
    (
        ("load_cache_policy", "wt", "load_cache_policy"),
        ("store_cache_policy", "cv", "store_cache_policy"),
    ),
)
def test_copy_rejects_invalid_cache_policy(keyword, policy, message):
    kwargs = {keyword: policy}
    with pytest.raises(ValueError, match=message):
        make_cache_policy_copy(**kwargs)


def test_copy_cache_policy_rejects_non_cuda_target():
    program = make_cache_policy_copy(load_cache_policy="cg")
    with tvm.target.Target("c"), pytest.raises(Exception, match="only supported by the CUDA normal-copy backend"):
        tilelang.lower(program, target="c")


@tilelang.testing.requires_cuda
def test_copy_cache_policy_requires_sync_instruction():
    program = make_cache_policy_copy(load_cache_policy="cg", prefer_instruction=None)
    with pytest.raises(Exception, match='currently require prefer_instruction="sync"'):
        lower_cuda_source(program)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("policy_kind", ("load", "store"))
def test_copy_cache_policy_requires_global_access(policy_kind):
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((128,), T.float32)
            if policy_kind == "load":
                T.copy(A, shared, prefer_instruction="sync")
                T.copy(shared, B, load_cache_policy="cg", prefer_instruction="sync")
            else:
                T.copy(
                    A,
                    shared,
                    store_cache_policy="wt",
                    prefer_instruction="sync",
                )

    expected = "global-memory source" if policy_kind == "load" else "global-memory destination"
    with pytest.raises(Exception, match=expected):
        lower_cuda_source(main)


if __name__ == "__main__":
    tilelang.testing.main()
