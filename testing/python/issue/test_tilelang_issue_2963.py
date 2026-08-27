import pytest
import torch

import tilelang
import tilelang.testing
import tvm
from tilelang import language as T
from tilelang.backend import create_backend_context
from tilelang.engine.lower import lower_to_host_device_ir


@pytest.fixture(autouse=True)
def _disable_tilelang_cache():
    tilelang.disable_cache()
    try:
        yield
    finally:
        tilelang.enable_cache()


def _packed_input(dtype, n):
    values = [(((byte // 8) % 8) << 4) | (byte % 8) for byte in range(n // 2)]
    packed = torch.tensor(values, dtype=torch.uint8, device="cuda")
    return packed if dtype == "uint4" else packed.view(torch.int8)


def _copy_kernel(dtype, n, threads, *, through_shared=False, coalesced_width=None):
    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=threads):
            if through_shared:
                shared = T.alloc_shared((n,), dtype)
                T.copy(A, shared, coalesced_width=coalesced_width)
                T.copy(shared, B, coalesced_width=coalesced_width)
            else:
                T.copy(A, B, coalesced_width=coalesced_width)

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4", "float4_e2m1fn"])
@pytest.mark.parametrize("through_shared", [False, True])
def test_packed_subbyte_copy_keeps_byte_on_one_thread(dtype, through_shared):
    n = 128
    kernel = _copy_kernel(dtype, n, threads=128, through_shared=through_shared)
    compiled = tilelang.compile(kernel, out_idx=[1])

    source = _packed_input(dtype, n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


@tilelang.testing.requires_cuda
def test_scalar_coalesced_width_rejected_for_contiguous_packed_store():
    kernel = _copy_kernel("int4", 128, threads=128, coalesced_width=1)

    with pytest.raises(Exception, match=r"split a writable byte across threads|no available layout found"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_unaligned_contiguous_packed_store_rejected():
    n = 130

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i + 1] = A[i + 1]

    with pytest.raises(Exception, match=r"Cannot safely lower a packed four-bit store|no available layout found"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_single_nibble_per_byte_store_remains_legal():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(n // 2):
                B[2 * i + 1] = A[2 * i + 1]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input("int4", n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8) & 0xF0, source.view(torch.uint8) & 0xF0)


@tilelang.testing.requires_cuda
def test_disjoint_packed_store_sites_remain_legal():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n // 4):
                B[2 * i] = A[2 * i]
                B[2 * (i + n // 4) + 1] = A[2 * (i + n // 4) + 1]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input("int4", n)
    result_bytes = compiled(source).view(torch.uint8)
    source_bytes = source.view(torch.uint8)
    assert torch.equal(result_bytes[: n // 4] & 0x0F, source_bytes[: n // 4] & 0x0F)
    assert torch.equal(result_bytes[n // 4 :] & 0xF0, source_bytes[n // 4 :] & 0xF0)


def test_packed_byte_ownership_does_not_vectorize_hip_layout():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n):
                B[i] = A[i]

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx950"})
    with target:
        context = create_backend_context(target, None, "auto")
        _, device_mod, _, _, _ = lower_to_host_device_ir(kernel, context)
    lowered = str(device_mod)
    assert "B_1[tx] = A_1[tx]" in lowered
    assert "B_1[tx * 2:tx * 2 + 2]" not in lowered


@tilelang.testing.requires_cuda
def test_packed_fragment_pipeline_preserves_byte_ownership():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((n,), "int4")
            B_local = T.alloc_fragment((n,), "int4")
            T.copy(A, A_local)
            for i in T.Parallel(n):
                B_local[i] = A_local[i]
            T.copy(B_local, B)

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input("int4", n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


@tilelang.testing.requires_cuda
def test_fragment_candidate_fallback_preserves_byte_ownership():
    n = 128

    def permuted_replicated_byte_owner(i, replica):
        byte = i // 2
        return byte % 2 * (n // 4) + byte // 2 + replica * (n // 2), i % 2

    fragment_layout = T.Fragment((n,), forward_fn=permuted_replicated_byte_owner, replicate=2)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "uint4"), B: T.Tensor((n,), "uint4")):
        with T.Kernel(1, threads=128):
            local = T.alloc_fragment((n,), "uint4")
            T.annotate_layout({local: fragment_layout})
            T.copy(A, local)
            for i in T.Parallel(n):
                B[i] = local[i]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = torch.arange(n // 2, dtype=torch.uint8, device="cuda")
    result = compiled(source)

    assert torch.equal(result.view(torch.uint8), source)


@tilelang.testing.requires_cuda
def test_missing_fragment_and_buffer_candidates_report_layout_conflict():
    n = 128

    def single_thread_owner(i):
        return 0, i

    fragment_layout = T.Fragment((1,), forward_fn=single_thread_owner)

    @T.prim_func
    def kernel(A: T.Tensor((1,), "uint4"), B: T.Tensor((n,), "uint4")):
        with T.Kernel(1, threads=128):
            local = T.alloc_fragment((1,), "uint4")
            T.annotate_layout({local: fragment_layout})
            local[0] = A[0]
            for i in T.Parallel(n):
                B[i] = local[0]

    with pytest.raises(Exception, match=r"no available layout found|No compatible loop layout"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_explicit_byte_safe_width_preserves_values():
    n = 128
    kernel = _copy_kernel("int4", n, threads=128, coalesced_width=2)
    compiled = tilelang.compile(kernel, out_idx=[1])

    source = _packed_input("int4", n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


@tilelang.testing.requires_cuda
def test_multiple_packed_store_sites_are_rejected_when_ownership_disagrees():
    n = 132

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(64):
                B[2 * i] = A[2 * i]
                shifted = 2 * i + 3
                B[shifted] = A[shifted]

    with pytest.raises(Exception, match=r"no available layout found|no legal vector width"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_replicated_packed_physical_store_requires_single_replica_guard():
    n = 64

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            local = T.alloc_fragment((n,), "int4")
            for i in T.Parallel(n):
                local[i] = A[i]
                B[i] = A[i]

    with pytest.raises(
        Exception,
        match=r"single-replica guard|no legal vector width|no available layout found",
    ):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_auto_width_search_tries_all_packed_safe_candidates():
    n = 128

    # Width four gives i and i + 64 different byte owners in each store site;
    # width two keeps them on one thread, while the branches make runtime
    # writes unique across the two output buffers.
    @T.prim_func
    def kernel(
        A0: T.Tensor((n // 2,), "uint4"),
        A1: T.Tensor((n // 2,), "uint4"),
        B0: T.Tensor((n // 2,), "uint4"),
        B1: T.Tensor((n // 2,), "uint4"),
    ):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                j = i % (n // 2)
                if i < n // 2:
                    B0[j] = A0[j]
                else:
                    B1[j] = A1[j]

    compiled = tilelang.compile(kernel, out_idx=[2, 3])
    source0 = torch.arange(n // 4, dtype=torch.uint8, device="cuda")
    source1 = torch.arange(n // 4 - 1, -1, -1, dtype=torch.uint8, device="cuda")
    result0, result1 = compiled(source0, source1)

    assert torch.equal(result0.view(torch.uint8), source0)
    assert torch.equal(result1.view(torch.uint8), source1)


@tilelang.testing.requires_cuda
def test_annotated_shared_layout_revalidates_physical_byte_ownership():
    n = 128

    def deinterleaved_storage(i):
        return i // 2 + (i % 2) * (n // 2)

    def logical_pair_owner(i):
        return i // 2, i % 2

    pair_layout = T.Fragment((n,), forward_fn=logical_pair_owner)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "uint4"), B: T.Tensor((n,), "uint4")):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((n,), "uint4")
            T.annotate_layout({shared: T.Layout((n,), deinterleaved_storage)})
            for i in T.Parallel(n, loop_layout=pair_layout):
                shared[i] = A[i]
            T.sync_threads()
            for i in T.Parallel(n, loop_layout=pair_layout):
                B[i] = shared[i]

    with pytest.raises(Exception, match="Cannot safely lower a packed four-bit store"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_byte_safe_annotated_shared_layout_remains_legal():
    n = 128

    def reversed_byte_storage(i):
        return 2 * (n // 2 - 1 - i // 2) + i % 2

    def logical_pair_owner(i):
        return i // 2, i % 2

    pair_layout = T.Fragment((n,), forward_fn=logical_pair_owner)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "uint4"), B: T.Tensor((n,), "uint4")):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((n,), "uint4")
            T.annotate_layout({shared: T.Layout((n,), reversed_byte_storage)})
            for i in T.Parallel(n, loop_layout=pair_layout):
                shared[i] = A[i]
            T.sync_threads()
            for i in T.Parallel(n, loop_layout=pair_layout):
                B[i] = shared[i]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = torch.arange(n // 2, dtype=torch.uint8, device="cuda")
    result = compiled(source)

    assert torch.equal(result.view(torch.uint8), source)


@tilelang.testing.requires_cuda
def test_replicated_byte_safe_layout_uses_single_replica_guard():
    n = 128

    def replicated_pair_owner(i, replica):
        return (i // 2) % 64 + replica * 64, i % 2

    replicated_pair = T.Fragment((n,), forward_fn=replicated_pair_owner, replicate=2)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "uint4"), B: T.Tensor((n,), "uint4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n, loop_layout=replicated_pair):
                B[i] = A[i]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = torch.arange(n // 2, dtype=torch.uint8, device="cuda")
    result = compiled(source)

    assert torch.equal(result.view(torch.uint8), source)


@tilelang.testing.requires_cuda
def test_annotated_one_nibble_per_thread_layout_is_rejected():
    n = 128

    def one_nibble_layout_fn(i):
        return i, 0

    one_nibble_per_thread = T.Fragment((n,), forward_fn=one_nibble_layout_fn)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n, loop_layout=one_nibble_per_thread):
                B[i] = A[i]

    with pytest.raises(Exception, match="logical elements that share a writable byte"):
        tilelang.compile(kernel, out_idx=[1])
