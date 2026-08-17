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

    with pytest.raises(Exception, match="split a writable byte across threads"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_unaligned_contiguous_packed_store_rejected():
    n = 130

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i + 1] = A[i + 1]

    with pytest.raises(Exception, match="Cannot safely lower a packed four-bit store"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_scalar_strided_packed_store_remains_legal():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(n // 2):
                B[2 * i] = A[2 * i]

    source = tilelang.compile(kernel, out_idx=[1]).get_kernel_source()
    assert "tl_int4_packed_store" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4", "float4_e2m1fn"])
def test_high_nibble_only_store_remains_legal(dtype):
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(n // 2):
                B[2 * i + 1] = A[2 * i + 1]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input(dtype, n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8) & 0xF0, source.view(torch.uint8) & 0xF0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4", "float4_e2m1fn"])
def test_disjoint_packed_store_sites_remain_legal(dtype):
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n // 4):
                B[2 * i] = A[2 * i]
                B[2 * (i + n // 4) + 1] = A[2 * (i + n // 4) + 1]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input(dtype, n)
    result_bytes = compiled(source).view(torch.uint8)
    source_bytes = source.view(torch.uint8)
    assert torch.equal(result_bytes[: n // 4] & 0x0F, source_bytes[: n // 4] & 0x0F)
    assert torch.equal(result_bytes[n // 4 :] & 0xF0, source_bytes[n // 4 :] & 0xF0)


@tilelang.testing.requires_cuda
def test_unverified_cyclic_packed_owner_inverse_is_rejected():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(n // 2):
                byte = (i + n // 4) % (n // 2)
                B[2 * byte + 1] = A[2 * byte + 1]

    with pytest.raises(Exception, match="no legal vector width"):
        tilelang.compile(kernel, out_idx=[1])


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_byte_ownership_does_not_vectorize_hip_layout(dtype):
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
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
@pytest.mark.parametrize("dtype", ["int4", "uint4", "float4_e2m1fn"])
def test_packed_fragment_pipeline_preserves_byte_ownership(dtype):
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((n,), dtype)
            B_local = T.alloc_fragment((n,), dtype)
            T.copy(A, A_local)
            for i in T.Parallel(n):
                B_local[i] = A_local[i]
            T.copy(B_local, B)

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input(dtype, n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4", "float4_e2m1fn"])
@pytest.mark.parametrize("coalesced_width", [2, 4])
def test_explicit_byte_safe_widths_preserve_values(dtype, coalesced_width):
    n = 128
    kernel = _copy_kernel(dtype, n, threads=128, coalesced_width=coalesced_width)
    compiled = tilelang.compile(kernel, out_idx=[1])

    source = _packed_input(dtype, n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


@tilelang.testing.requires_cuda
def test_nested_serial_packed_store_is_rejected_when_ownership_is_unproven():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n):
                for j in T.serial(1):
                    B[i + j] = A[i + j]

    with pytest.raises(Exception, match="no legal vector width keeps each"):
        tilelang.compile(kernel, out_idx=[1])


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

    with pytest.raises(Exception, match="no available layout found|no legal vector width"):
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

    with pytest.raises(Exception, match="single-replica guard|no legal vector width"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_annotated_pair_layout_preserves_byte_ownership():
    n = 128

    def pair_layout_fn(i):
        return i // 2, i % 2

    pair_layout = T.Fragment((n,), forward_fn=pair_layout_fn)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n, loop_layout=pair_layout):
                B[i] = A[i]

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = _packed_input("int4", n)
    result = compiled(source)
    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))


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


@tilelang.testing.requires_cuda
def test_deinterleaved_parallel_packed_store_is_rejected():
    n = 128

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=128):
            for parity, byte in T.Parallel(2, n // 2):
                index = 2 * byte + parity
                B[index] = A[index]

    with pytest.raises(Exception, match="no legal vector width keeps each"):
        tilelang.compile(kernel, out_idx=[1])


@tilelang.testing.requires_cuda
def test_packed_sink_does_not_constrain_unrelated_float_fanout():
    n = 128

    @T.prim_func
    def kernel(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "int4"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1, threads=64):
            x = T.alloc_fragment((n,), "float32")
            q = T.alloc_fragment((n,), "int4")
            y = T.alloc_fragment((n,), "float32")
            T.copy(A, x)
            for i in T.Parallel(n):
                q[i] = T.cast(x[i], "int4")
            for t, k in T.Parallel(64, 2):
                y[t + 64 * k] = x[2 * t + k]
            T.copy(q, B)
            T.copy(y, C)

    compiled = tilelang.compile(kernel, out_idx=[1, 2])
    source = (torch.arange(n, device="cuda") % 8).float()
    packed, deinterleaved = compiled(source)

    values = source.to(torch.uint8)
    expected_packed = values[0::2] | (values[1::2] << 4)
    expected_deinterleaved = torch.cat([source[0::2], source[1::2]])
    assert torch.equal(packed.view(torch.uint8), expected_packed)
    assert torch.equal(deinterleaved, expected_deinterleaved)


@tilelang.testing.requires_cuda
def test_fragment_only_packed_values_allow_one_nibble_per_thread():
    n = 128

    def one_nibble_layout_fn(i):
        return i, 0

    one_nibble_per_thread = T.Fragment((n,), forward_fn=one_nibble_layout_fn)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int4"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=128):
            packed_local = T.alloc_fragment((n,), "int4")
            for i in T.Parallel(n, loop_layout=one_nibble_per_thread):
                packed_local[i] = A[i]
            for i in T.Parallel(n):
                C[i] = T.cast(packed_local[i], "float32")

    compiled = tilelang.compile(kernel, out_idx=[1])
    byte_values = torch.tensor(
        [(((byte // 8) % 8) << 4) | (byte % 8) for byte in range(n // 2)],
        dtype=torch.uint8,
        device="cuda",
    )
    result = compiled(byte_values.view(torch.int8))
    expected = torch.empty(n, dtype=torch.float32, device="cuda")
    expected[0::2] = (byte_values & 0xF).float()
    expected[1::2] = (byte_values >> 4).float()
    assert torch.equal(result, expected)


@tilelang.testing.requires_cuda
def test_packed_sink_does_not_constrain_unrelated_reads_in_same_parallel_op():
    n = 128

    def deinterleaved_layout_fn(i):
        return i % 64, i // 64

    deinterleaved = T.Fragment((n,), forward_fn=deinterleaved_layout_fn)

    @T.prim_func
    def kernel(
        A: T.Tensor((n,), "float32"),
        Q: T.Tensor((n,), "int4"),
        B: T.Tensor((n,), "int4"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1, threads=64):
            x = T.alloc_fragment((n,), "float32")
            q = T.alloc_fragment((n,), "int4")
            y = T.alloc_fragment((n,), "float32")
            for i in T.Parallel(n, loop_layout=deinterleaved):
                x[i] = A[i]
            T.copy(Q, q)
            for t, k in T.Parallel(64, 2):
                B[2 * t + k] = q[2 * t + k]
                y[t + 64 * k] = x[t + 64 * k]
            T.copy(y, C)

    compiled = tilelang.compile(kernel, out_idx=[2, 3])
    float_source = torch.arange(n, device="cuda").float()
    packed_source = torch.tensor(
        [(((byte // 8) % 8) << 4) | (byte % 8) for byte in range(n // 2)],
        dtype=torch.uint8,
        device="cuda",
    )
    packed_result, float_result = compiled(float_source, packed_source.view(torch.int8))
    assert torch.equal(packed_result.view(torch.uint8), packed_source)
    assert torch.equal(float_result, float_source)


@tilelang.testing.requires_cuda
def test_shared_memory_relayout_stops_packed_pair_propagation():
    n = 128

    def deinterleaved_layout_fn(i):
        return i % 64, i // 64

    deinterleaved = T.Fragment((n,), forward_fn=deinterleaved_layout_fn)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "int4")):
        with T.Kernel(1, threads=64):
            x = T.alloc_fragment((n,), "float32")
            shared = T.alloc_shared((n,), "float32")
            q = T.alloc_fragment((n,), "int4")
            for i in T.Parallel(n, loop_layout=deinterleaved):
                x[i] = A[i]
            for t, k in T.Parallel(64, 2):
                shared[t + 64 * k] = x[t + 64 * k]
            T.sync_threads()
            for i in T.Parallel(n):
                q[i] = T.cast(shared[i], "int4")
            T.copy(q, B)

    compiled = tilelang.compile(kernel, out_idx=[1])
    source = (torch.arange(n, device="cuda") % 8).float()
    result = compiled(source)
    values = source.to(torch.uint8)
    expected = values[0::2] | (values[1::2] << 4)
    assert torch.equal(result.view(torch.uint8), expected)
