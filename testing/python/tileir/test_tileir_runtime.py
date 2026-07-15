"""End-to-end TileIR runtime execution tests."""

from __future__ import annotations


import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal
Target = tvm.target.Target


from tileir_test_utils import (
    _load_mla_paged_example,
    _load_mla_decode_example,
    _load_mla_kv_fp8_example,
    _load_mla_persistent_example,
    _skip_if_tileir_toolchain_unavailable,
    _enable_tileir_runtime,
)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("num_split", [1, 2])
def test_tileir_runs_deepseek_mla_decode_runtime(monkeypatch, num_split):
    _skip_if_tileir_toolchain_unavailable()
    _enable_tileir_runtime(monkeypatch)

    example = _load_mla_decode_example()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = 1, 16, 1, 64 * num_split, 64, 64
    block_n, block_h = 64, 16
    softmax_scale = (dim + pe_dim) ** -0.5
    kernel = example.flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split, softmax_scale)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    profiler.assert_allclose(example.ref_program, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("num_split", [1, 2])
def test_tileir_runs_deepseek_mla_paged_runtime(monkeypatch, num_split):
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)

    example = _load_mla_paged_example()
    batch, heads, kv_heads = 1, 16, 1
    cache_seqlen = 64 * num_split
    dim, value_dim = 128, 64
    pe_dim = dim - value_dim
    max_seqlen_pad = 256
    block_size, block_n, block_h = 64, 64, 16
    dtype = torch.float16

    q = torch.randn(batch, 1, heads, dim, dtype=dtype, device="cuda")
    cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32, device="cuda")
    block_table = torch.arange(
        batch * max_seqlen_pad // block_size,
        dtype=torch.int32,
        device="cuda",
    ).view(batch, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, kv_heads, dim, dtype=dtype, device="cuda")

    q_nope, q_pe = q[..., :value_dim].contiguous(), q[..., value_dim:].contiguous()
    blocked_k_nope = blocked_k[..., :value_dim].contiguous()
    blocked_k_pe = blocked_k[..., value_dim:].contiguous()
    glse = torch.empty(batch, heads, num_split, dtype=dtype, device="cuda")
    output_partial = torch.empty(batch, heads, num_split, value_dim, dtype=dtype, device="cuda")

    kernel = example.mla_decode_tilelang(
        batch,
        heads,
        kv_heads,
        max_seqlen_pad,
        value_dim,
        pe_dim,
        block_n,
        block_h,
        num_split,
        block_size,
        None,
    )
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    out = profiler.func(
        q_nope.view(-1, heads, value_dim),
        q_pe.view(-1, heads, pe_dim),
        blocked_k_nope.view(-1, kv_heads, value_dim),
        blocked_k_pe.view(-1, kv_heads, pe_dim),
        block_table,
        cache_seqlens,
        glse,
        output_partial,
    ).view(batch, 1, heads, value_dim)

    ref = example.run_torch_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        batch,
        1,
        cache_seqlens,
        heads,
        kv_heads,
        dim,
        value_dim,
        True,
        dtype,
    )
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_runs_deepseek_mla_persistent_runtime(monkeypatch):
    _skip_if_tileir_toolchain_unavailable()
    _enable_tileir_runtime(monkeypatch)

    example = _load_mla_persistent_example()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = 1, 16, 1, 128, 64, 64
    block_n, block_h, num_split = 64, 16, 2
    kernel = example.flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    profiler.assert_allclose(example.ref_program, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_runs_deepseek_mla_kv_fp8_runtime(monkeypatch):
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)

    example = _load_mla_kv_fp8_example()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = 1, 16, 1, 64, 64, 64
    block_n, block_h = 64, 16
    kernel = example.flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    q = torch.randn(batch, heads, dim, dtype=torch.float16, device="cuda")
    q_pe = torch.randn(batch, heads, pe_dim, dtype=torch.float16, device="cuda")
    kv = torch.randn(batch, kv_ctx, kv_heads, dim, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    k_pe = torch.randn(batch, kv_ctx, kv_heads, pe_dim, dtype=torch.float16, device="cuda")

    out = profiler.func(q, q_pe, kv, k_pe)
    ref = example.ref_program(q, q_pe, kv.to(torch.float16), k_pe)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("latency, disable_tma", [(None, False), (8, False), (None, True), (4, True)])
def test_tileir_runs_copy_with_load_store_hints(monkeypatch, latency, disable_tma):
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)
    major, minor = torch.cuda.get_device_capability()
    target = f"tileir -arch=sm_{major}{minor}"
    M = N = 128

    @tilelang.jit(out_idx=[-1], target=target, execution_backend="tileir")
    def copy_kernel(M: int, N: int, latency: int | None = None, disable_tma: bool = False):
        @T.prim_func
        def main(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                A_shared = T.alloc_shared((M, N), "float32")
                T.copy(A[0:M, 0:N], A_shared, latency=latency, disable_tma=disable_tma)
                T.copy(A_shared, C[0:M, 0:N])

        return main

    kernel = copy_kernel(M, N, latency=latency, disable_tma=disable_tma)
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel(a), a, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_serial_accum_from_2d_fragment_not_silently_zero(monkeypatch):
    """A serial row accumulation must compute correctly or fail explicitly."""
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)
    major, minor = torch.cuda.get_device_capability()
    target = f"tileir -arch=sm_{major}{minor}"
    N = 4

    @tilelang.jit(out_idx=[-1], target=target, execution_backend="tileir")
    def accum_kernel():
        @T.prim_func
        def main(Fin: T.Tensor((N, 128), "float32"), Out: T.Tensor((1, 128), "float32")):
            with T.Kernel(1, threads=128) as bx:
                frag = T.alloc_fragment([N, 128], "float32")
                acc = T.alloc_fragment([128], "float32")
                T.copy(Fin, frag)  # clean full-tile fill (no partial-index global load)
                T.clear(acc)
                for k in T.serial(N):
                    for j in T.Parallel(128):
                        acc[j] += frag[k, j]
                for j in T.Parallel(128):
                    Out[bx, j] = acc[j]

        return main

    fin = torch.ones(N, 128, device="cuda", dtype=torch.float32)
    try:
        kernel = accum_kernel()
        out = kernel(fin)
    except NotImplementedError:
        return  # loud failure is acceptable; a silent zero is not
    val = float(out[0, 0].item())
    assert abs(val - N) < 1e-4, f"silent miscompile: serial accumulation gave {val}, expected {N}"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_register_fragment_row_slice_accumulates_correctly(monkeypatch):
    """Reading a row of a 2D register fragment indexed by a serial-loop var
    (``acc[j] += frag[k, j]``) must compute the correct sum, not over-broaden the
    accumulator. Requires register/shared-fragment row-slice lowering (the same
    pattern as flash_decode's split-K combine). Strict: must NOT raise.
    """
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)
    major, minor = torch.cuda.get_device_capability()
    target = f"tileir -arch=sm_{major}{minor}"
    N = 4

    @tilelang.jit(out_idx=[-1], target=target, execution_backend="tileir")
    def accum_kernel():
        @T.prim_func
        def main(Fin: T.Tensor((N, 128), "float32"), Out: T.Tensor((1, 128), "float32")):
            with T.Kernel(1, threads=128) as bx:
                frag = T.alloc_fragment([N, 128], "float32")
                acc = T.alloc_fragment([128], "float32")
                T.copy(Fin, frag)  # clean full-tile fill (no partial-index global load)
                T.clear(acc)
                for k in T.serial(N):
                    for j in T.Parallel(128):
                        acc[j] += frag[k, j]
                for j in T.Parallel(128):
                    Out[bx, j] = acc[j]

        return main

    fin = torch.arange(1, N + 1, device="cuda", dtype=torch.float32).reshape(N, 1).expand(N, 128).contiguous()  # row k = k+1
    out = accum_kernel()(fin)
    expected = float(N * (N + 1) / 2)  # 1+2+3+4 = 10
    val = float(out[0, 0].item())
    assert abs(val - expected) < 1e-4, f"row-slice accumulation gave {val}, expected {expected}"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_parallel_alloc_var_if_else_masked_store_is_correct(monkeypatch):
    """An ``alloc_var`` written in BOTH arms of an if/else inside T.Parallel must
    keep each lane's branch value, not clobber the whole tile with one branch.

    The masked REGISTER store lowers ``if c: v = A else: v = B`` to two masked
    selects; the scalar ``v`` ([1]) must be promoted to the parallel tile shape.
    The result is 0 at [0,0] (where i+j==0) and 1 everywhere else.
    """
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)
    major, minor = torch.cuda.get_device_capability()
    target = f"tileir -arch=sm_{major}{minor}"

    @tilelang.jit(out_idx=[-1], target=target, execution_backend="tileir")
    def alloc_var_kernel():
        @T.prim_func
        def main(Out: T.Tensor((16, 8), "float32")):
            with T.Kernel(1):
                frag = T.alloc_fragment((16, 8), "float32")
                for i, j in T.Parallel(16, 8):
                    value = T.alloc_var("float32")
                    if i + j > 0:
                        value = 1.0
                    else:
                        value = 0.0
                    frag[i, j] = value
                T.copy(frag, Out)

        return main

    out = alloc_var_kernel()()
    expected = torch.ones(16, 8, device="cuda", dtype=torch.float32)
    expected[0, 0] = 0.0
    assert out[0, 0].item() == 0.0, f"[0,0] should be 0 (i+j==0), got {out[0, 0].item()}"
    assert out[1, 0].item() == 1.0, f"[1,0] should be 1 (i+j>0), got {out[1, 0].item()}"
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tileir_reinterpret_is_bitcast_not_numeric_convert(monkeypatch):
    """T.reinterpret must preserve the bit pattern (a bitcast), not numerically
    convert. reinterpret(int32, 1.0f) must be 0x3f800000 (1065353216), not 1.

    The differing-type reinterpret was lowered to a numeric Cast (ftoi), silently
    miscompiling bit-reinterprets used by quantization/packing kernels.
    """
    _skip_if_tileir_toolchain_unavailable()
    torch = _enable_tileir_runtime(monkeypatch)
    major, minor = torch.cuda.get_device_capability()
    target = f"tileir -arch=sm_{major}{minor}"
    N = 128

    @tilelang.jit(out_idx=[-1], target=target, execution_backend="tileir")
    def reinterpret_kernel():
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "int32")):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(N):
                    B[i] = T.reinterpret("int32", A[i])

        return main

    a = torch.linspace(-3.0, 5.0, N, device="cuda", dtype=torch.float32)
    out = reinterpret_kernel()(a)
    expected = a.view(torch.int32)
    assert torch.equal(out, expected), f"reinterpret must bit-preserve: got {out[:3].tolist()}, expected {expected[:3].tolist()}"
