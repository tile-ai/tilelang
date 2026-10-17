import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tilelang.layout import SwizzleMode
from tilelang.layout.swizzle import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import compute_gmma_descriptor


@pytest.mark.parametrize(
    "maker,mode,wgmma_field,sbo",
    [
        (make_full_bank_swizzled_layout, SwizzleMode.SWIZZLE_128B, 1, 64),
        (make_half_bank_swizzled_layout, SwizzleMode.SWIZZLE_64B, 2, 32),
        (make_quarter_bank_swizzled_layout, SwizzleMode.SWIZZLE_32B, 3, 16),
    ],
)
@pytest.mark.parametrize("n_dim,k_dim", [(64, 64), (64, 256), (128, 64), (64, 512)])
def test_compute_gmma_descriptor_k_major(maker, mode, wgmma_field, sbo, n_dim, k_dim):
    buf = tirx.decl_buffer((n_dim, k_dim), "float16", name="B", scope="shared")
    p = compute_gmma_descriptor(maker(buf), buf, transposed=False)
    assert p.swizzle_mode == mode
    assert p.swizzle_mode.wgmma_layout_type() == wgmma_field
    assert p.leading_byte_offset == 1
    assert p.stride_byte_offset == sbo


@pytest.mark.parametrize("k_dim,n_dim", [(64, 128), (64, 256), (64, 64)])
def test_compute_gmma_descriptor_mn_major_128b(k_dim, n_dim):
    buf = tirx.decl_buffer((k_dim, n_dim), "float16", name="B", scope="shared")
    p = compute_gmma_descriptor(make_full_bank_swizzled_layout(buf), buf, transposed=True)
    assert p.swizzle_mode == SwizzleMode.SWIZZLE_128B
    assert p.stride_byte_offset == 64
    # LBO is 0 for a single MN atom (n_dim == atom), else the multi-atom step.
    assert p.leading_byte_offset in (0, 512)


def test_compute_gmma_descriptor_same_layout_both_majors():
    # FlashMLA KV_shared [block_N, h_dim] (128B swizzle) must analyze as BOTH a
    # K-major operand (QK) and an MN-major operand (PV) -- the same physical bytes.
    buf = tirx.decl_buffer((64, 256), "float16", name="KV", scope="shared")
    lay = make_full_bank_swizzled_layout(buf)
    pk = compute_gmma_descriptor(lay, buf, transposed=False)
    pmn = compute_gmma_descriptor(lay, buf, transposed=True)
    assert pk.leading_byte_offset == 1 and pk.stride_byte_offset == 64
    assert pmn.swizzle_mode == SwizzleMode.SWIZZLE_128B


def _make_wgmma_kernel(gemm_op):
    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            T.copy(A[0:64, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            gemm_op(A_shared, B_shared, C_local)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:64, 0:64])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize(
    "gemm_api",
    [T.wgmma_gemm],
)
def test_wgmma_gemm_has_no_implicit_wait(gemm_api):
    kernel = tilelang.compile(_make_wgmma_kernel(lambda A, B, C: gemm_api(A, B, C, clear_accum=True)), target="cuda")
    src = kernel.get_kernel_source()

    assert src.count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_dispatch_has_no_implicit_wait():
    kernel = tilelang.compile(
        _make_wgmma_kernel(lambda A, B, C: T.wgmma_gemm(A, B, C, clear_accum=True)),
        target="cuda",
    )

    assert kernel.get_kernel_source().count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_rejects_mma_fallback():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((32, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float16)

            T.copy(A[0:32, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:32, 0:64])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.wgmma_gemm\(\) requires Hopper WGMMA lowering",
    ):
        tilelang.compile(main, target="cuda")


def _make_sliced_wgmma_kernel(M, N, K, num_k_tiles):
    k_tile = K // num_k_tiles

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        D: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.float16)
            B_shared = T.alloc_shared((N, K), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            for j in T.serial(num_k_tiles):
                T.wgmma_gemm(
                    A_shared[:, j * k_tile : (j + 1) * k_tile],
                    B_shared[:, j * k_tile : (j + 1) * k_tile],
                    C_local,
                    transpose_B=True,
                )
            T.wait_wgmma(0)
            T.copy(C_local, D)

    return main


def _make_static_sliced_wgmma_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((64, 32), T.float16),
        B: T.Tensor((64, 32), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 32), T.float16)
            B_shared = T.alloc_shared((64, 32), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.wgmma_gemm(
                A_shared[:, 16:],
                B_shared[:, 16:],
                C_local,
                transpose_B=True,
            )
            T.wait_wgmma(0)
            T.copy(C_local, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_static_slice_initializes_from_slice_pointer():
    import torch

    kernel = tilelang.compile(_make_static_sliced_wgmma_kernel(), target="cuda")
    source = kernel.get_kernel_source()

    assert source.count("initialize_wgmma_descriptor") == 2
    assert "increase_descriptor_offset" not in source
    assert source.count("+ 2048") == 2

    a = torch.randn((64, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 32), device="cuda", dtype=torch.float16)
    d = torch.empty((64, 64), device="cuda", dtype=torch.float16)
    kernel(a, b, d)
    expected = (a[:, 16:].float() @ b[:, 16:].float().T).half()
    torch.testing.assert_close(d, expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_sliced_operand_emits_offset():
    # A sliced (non-zero-origin) operand must build the descriptor from the buffer
    # base and advance it with increase_descriptor_offset.
    kernel = tilelang.compile(_make_sliced_wgmma_kernel(64, 64, 64, 4), target="cuda")
    src = kernel.get_kernel_source()
    assert "increase_descriptor_offset" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize(
    "M,N,K,num_k_tiles",
    [
        (64, 64, 64, 4),
        (64, 64, 128, 4),
        (64, 64, 256, 8),
        (128, 64, 64, 2),
        (64, 128, 64, 2),
    ],
)
def test_wgmma_gemm_sliced_operand_correctness(M, N, K, num_k_tiles):
    import torch

    kernel = tilelang.compile(_make_sliced_wgmma_kernel(M, N, K, num_k_tiles), target="cuda")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    d = torch.empty(M, N, device="cuda", dtype=torch.float16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().t()).half()
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)


def _make_async_wgmma_reuse_kernel(num_ctas=256, m=64, d=64, conditional_first_wait=False):
    # B0 is last used syntactically by the first wgmma_gemm (via its descriptor), but the
    # MMA reads it until wait_wgmma. B1 is written in between and must not share B0's storage.
    @T.prim_func
    def main(
        Q: T.Tensor((num_ctas, m, d), T.bfloat16),
        K: T.Tensor((num_ctas, 2 * m, d), T.bfloat16),
        O0: T.Tensor((num_ctas, m, m), T.float32),
        O1: T.Tensor((num_ctas, m, m), T.float32),
    ):
        with T.Kernel(num_ctas, threads=128) as b:
            C0 = T.alloc_fragment((m, m), T.float32)
            C1 = T.alloc_fragment((m, m), T.float32)
            A = T.alloc_shared((m, d), T.bfloat16)
            B0 = T.alloc_shared((m, d), T.bfloat16)
            B1 = T.alloc_shared((m, d), T.bfloat16)
            T.copy(Q[b, 0:m, 0:d], A, disable_tma=True)
            T.copy(K[b, 0:m, 0:d], B0, disable_tma=True)
            T.wgmma_gemm(A, B0, C0, transpose_B=True, clear_accum=True)
            if conditional_first_wait:  # noqa: SIM102 - the prim_func parser needs the nested form
                # Only some CTAs retire the first wgmma here, so B0 must stay live until the final wait.
                if b % 2 == 0:
                    T.wait_wgmma(0)
            T.copy(K[b, m : 2 * m, 0:d], B1, disable_tma=True)
            T.wgmma_gemm(A, B1, C1, transpose_B=True, clear_accum=True)
            T.wait_wgmma(0)
            T.warpgroup_fence_operand(C0, num_regs=32)
            T.warpgroup_fence_operand(C1, num_regs=32)
            T.copy(C0, O0[b, 0:m, 0:m])
            T.copy(C1, O1[b, 0:m, 0:m])

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("conditional_first_wait", [False, True])
def test_async_wgmma_operand_not_merged_before_wait(conditional_first_wait):
    # Compile-only: B0 and B1 must get distinct shared-memory offsets (#3525), also when an
    # earlier wait_wgmma sits in a branch that other paths skip.
    import re

    kernel = tilelang.compile(
        _make_async_wgmma_reuse_kernel(conditional_first_wait=conditional_first_wait),
        target={"kind": "cuda", "arch": "sm_90a"},
    )
    src = kernel.get_kernel_source()
    offsets = {}
    for name in ("B0", "B1"):
        match = re.search(rf"\b{name}\s*=\s*\(\(void\*\)\(\(char\*\)buf_dyn_shmem\s*\+\s*(\d+)\)\)", src)
        assert match is not None, f"{name} not found in generated source"
        offsets[name] = int(match.group(1))
    assert offsets["B0"] != offsets["B1"]


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_async_wgmma_operand_not_merged_before_wait_correctness():
    import torch

    num_ctas, m, d = 256, 64, 64
    kernel = tilelang.compile(_make_async_wgmma_reuse_kernel(num_ctas, m, d), target="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(num_ctas, m, d, device="cuda", generator=generator).to(torch.bfloat16)
    k = torch.randn(num_ctas, 2 * m, d, device="cuda", generator=generator).to(torch.bfloat16)
    ref0 = torch.einsum("bmd,bnd->bmn", q.double(), k[:, :m].double()).float()
    ref1 = torch.einsum("bmd,bnd->bmn", q.double(), k[:, m:].double()).float()
    for _ in range(5):
        o0 = torch.empty(num_ctas, m, m, device="cuda")
        o1 = torch.empty(num_ctas, m, m, device="cuda")
        kernel(q, k, o0, o1)
        torch.testing.assert_close(o0, ref0, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(o1, ref1, rtol=1e-2, atol=1e-2)


def _merged_shared_dyn_sizes(func):
    import re

    mod = tvm.IRModule.from_expr(func)
    cuda_target = tvm.target.Target("cuda", host="llvm")
    mod = tvm.tirx.transform.Apply(lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target}))(mod)
    mod = tvm.tirx.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations()(mod)
    return [int(size) for size in re.findall(r'alloc_buffer\(\((\d+),\), "uint8", scope="shared.dyn"\)', mod.script())]


def test_wgmma_descriptor_reused_after_full_wait_keeps_operand_live():
    # One descriptor initialization feeds two WGMMA batches with a full wait in between. The
    # second batch reads B0 again, so a buffer written while it is in flight must not be merged
    # onto B0's storage.
    from tvm.script import tirx as T

    op = tvm.ir.Op.get

    @T.prim_func(private=True)
    def func():
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        B0 = T.alloc_buffer((4096,), "uint8", scope="shared.dyn")
        Tmp = T.alloc_buffer((4096,), "uint8", scope="shared.dyn")
        desc = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
        T.evaluate(
            T.call_intrin(
                "handle",
                op("tl.initialize_wgmma_descriptor"),
                desc[0],
                T.tvm_access_ptr(T.type_annotation("uint8"), B0.data, 0, 4096, 1),
                1,
                1,
                64,
            )
        )
        T.evaluate(T.call_intrin("handle", op("tl.ptx_wgmma_ss"), desc.data))
        T.evaluate(T.call_intrin("handle", op("tl.wait_wgmma"), 0))
        T.evaluate(T.call_intrin("handle", op("tl.ptx_wgmma_ss"), desc.data))
        Tmp[tx] = T.uint8(1)
        T.evaluate(T.call_intrin("handle", op("tl.wait_wgmma"), 0))

    # Disjoint storage is the sum of both buffers; merged storage would be 4096.
    assert _merged_shared_dyn_sizes(func) == [8192]


def test_wgmma_wait_in_sibling_branch_does_not_retire_other_branch_operand():
    # The wait in the else branch runs only when the then branch did not, so B0 (issued in the
    # then branch) is still in flight when Tmp is written after the if.
    from tvm.script import tirx as T

    op = tvm.ir.Op.get

    @T.prim_func(private=True)
    def func():
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        B0 = T.alloc_buffer((4096,), "uint8", scope="shared.dyn")
        Tmp = T.alloc_buffer((4096,), "uint8", scope="shared.dyn")
        desc = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
        if tx < 64:
            T.evaluate(
                T.call_intrin(
                    "handle",
                    op("tl.initialize_wgmma_descriptor"),
                    desc[0],
                    T.tvm_access_ptr(T.type_annotation("uint8"), B0.data, 0, 4096, 1),
                    1,
                    1,
                    64,
                )
            )
            T.evaluate(T.call_intrin("handle", op("tl.ptx_wgmma_ss"), desc.data))
        else:
            T.evaluate(T.call_intrin("handle", op("tl.wait_wgmma"), 0))
        Tmp[tx] = T.uint8(1)
        T.evaluate(T.call_intrin("handle", op("tl.wait_wgmma"), 0))

    assert _merged_shared_dyn_sizes(func) == [8192]
