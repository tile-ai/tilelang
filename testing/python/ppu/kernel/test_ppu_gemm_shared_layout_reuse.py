"""MMA reuses existing shared layouts, without hidden conversions or barriers."""

from contextlib import nullcontext

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import (
    make_linear_layout,
    make_half_bank_swizzled_layout,
    make_full_bank_swizzled_layout,
    make_swizzled_layout,
    make_ppu_swizzled_layout,
)
from tilelang.tools.compile_only import cuda_codegen_available


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize(
    "copy_producer,transposed,threads,disable_sync",
    [(False, False, 32, False), (False, False, 128, True), (True, False, 128, False), (True, True, 128, True)],
)
def test_gemm_reuses_current_shared_operand(copy_producer, transposed, threads, disable_sync):
    blocks = 32
    rows, columns = (64, 16) if transposed else (16, 64)

    @T.prim_func
    def main(
        A: T.Tensor((16, 16), T.bfloat16),
        B: T.Tensor((blocks, rows, columns), T.bfloat16),
        C: T.Tensor((blocks, 2, 16, 64), T.float32),
        Copied: T.Tensor((blocks, rows, columns), T.bfloat16),
    ):
        with T.Kernel(blocks, threads=threads) as block:
            a = T.alloc_fragment((16, 16), T.bfloat16)
            b = T.alloc_shared((rows, columns), T.bfloat16)
            c = T.alloc_fragment((16, 64), T.float32)
            T.annotate_layout({b: make_linear_layout(b)})
            T.copy(A, a)
            if copy_producer:
                T.copy(B[block, :, :], b)
            else:
                for i, j in T.Parallel(rows, columns):
                    b[i, j] = B[block, i, j]
            for step in T.serial(2):
                # Finish reads from the previous GEMM before overwriting B.
                T.sync_threads()
                if step == 1:
                    for i, j in T.Parallel(rows, columns):
                        b[i, j] = b[i, j] + T.bfloat16(1)
                T.sync_threads()
                T.gemm(a, b, c, transpose_B=transposed, clear_accum=True)
                T.copy(c, C[block, step, :, :])
            T.copy(b, Copied[block, :, :])

    kernel = tilelang.compile(main, out_idx=[2, 3], pass_configs={"tl.disable_thread_storage_sync": disable_sync})
    source = kernel.get_kernel_source()
    assert "extern __shared__ __align__(1024) bfloat16_t b[];" in source
    assert "tix_ldmatrix_swzl" not in source
    torch.manual_seed(0)
    a = torch.randint(-2, 3, (16, 16), device="cuda").bfloat16()
    b = torch.randint(-2, 3, (blocks, rows, columns), device="cuda").bfloat16()
    actual, copied = kernel(a, b)
    a_cpu, b_cpu = a.cpu().float(), b.cpu().float()
    expected = torch.stack(
        [a_cpu @ ((b_cpu + step).transpose(-1, -2) if transposed else b_cpu + step) for step in range(2)],
        dim=1,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(copied.cpu().float(), b_cpu + 1, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("thread_start,barrier_id", [(0, 3), (128, 7)])
def test_gemm_reuses_shared_with_explicit_partial_sync(thread_start, barrier_id):
    @T.prim_func
    def main(A: T.Tensor((16, 16), T.bfloat16), B: T.Tensor((16, 64), T.bfloat16), C: T.Tensor((16, 64), T.float32)):
        with T.Kernel(1, threads=256):
            a = T.alloc_fragment((16, 16), T.bfloat16)
            b = T.alloc_shared((16, 64), T.bfloat16)
            c = T.alloc_fragment((16, 64), T.float32)
            T.annotate_layout({b: make_linear_layout(b)})
            tx = T.get_thread_binding()
            if thread_start <= tx and tx < thread_start + 128:
                T.copy(A, a)
                T.copy(B, b)
                T.sync_threads(barrier_id, 128)
                T.gemm(a, b, c, clear_accum=True)
                T.copy(c, C)

    kernel = tilelang.compile(main, out_idx=[2], pass_configs={"tl.disable_thread_storage_sync": True})
    source = kernel.get_kernel_source()
    assert source.count("tl::__sync_thread_partial(") == 1
    assert f"tl::__sync_thread_partial({barrier_id}, 128);" in source
    assert "extern __shared__ __align__(1024) bfloat16_t b[];" in source
    a = torch.eye(16, device="cuda", dtype=torch.bfloat16)
    b = torch.arange(1024, device="cuda").reshape(16, 64).bfloat16()
    torch.testing.assert_close(kernel(a, b), b.float(), rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("path", ["SR", "RS"])
@pytest.mark.parametrize("layout_fn", [make_linear_layout, make_full_bank_swizzled_layout])
def test_gemm_preserves_shared_layout(path, layout_fn):
    @T.prim_func
    def main(A: T.Tensor((64, 64), T.bfloat16), B: T.Tensor((64, 64), T.bfloat16), C: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, 64), T.bfloat16)
            b = T.alloc_shared((64, 64), T.bfloat16)
            operand = T.alloc_fragment((64, 64), T.bfloat16)
            c = T.alloc_fragment((64, 64), T.float32)
            T.annotate_layout({a: layout_fn(a), b: layout_fn(b)})
            T.copy(A, a)
            T.copy(B, b)
            if path == "RS":
                T.copy(a, operand)
                T.gemm(operand, b, c, transpose_B=True, clear_accum=True)
            else:
                T.copy(b, operand)
                T.gemm(a, operand, c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    kernel = tilelang.compile(main, out_idx=[2])
    source = kernel.get_kernel_source()
    assert "a_trans" not in source and "b_trans" not in source
    assert ("tl::tix_ldmatrix_swzl" in source) == (layout_fn is make_full_bank_swizzled_layout)
    a = torch.randint(-2, 3, (64, 64), device="cuda").bfloat16()
    b = torch.randint(-2, 3, (64, 64), device="cuda").bfloat16()
    torch.testing.assert_close(kernel(a, b).cpu(), a.cpu().float() @ b.cpu().float().T, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize(
    "width,layout_fn,annotate_scales",
    [
        (128, make_linear_layout, False),
        (128, make_half_bank_swizzled_layout, True),
        (256, make_linear_layout, True),
        (256, make_half_bank_swizzled_layout, False),
        (256, make_full_bank_swizzled_layout, False),
    ],
)
def test_mxfp4_preserves_shared_layout(width, layout_fn, annotate_scales):
    @T.prim_func
    def main(
        A: T.Tensor((64, width), T.float4_e2m1fn),
        B: T.Tensor((64, width), T.float4_e2m1fn),
        SA: T.Tensor((width // 64, 64), T.uint16),
        SB: T.Tensor((width // 64, 64), T.uint16),
        C: T.Tensor((64, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, width), T.float4_e2m1fn)
            b = T.alloc_shared((64, width), T.float4_e2m1fn)
            sa = T.alloc_shared((width // 64, 64), T.uint16)
            sb = T.alloc_shared((width // 64, 64), T.uint16)
            c = T.alloc_fragment((64, 64), T.float32)
            T.annotate_layout({a: layout_fn(a), b: layout_fn(b)})
            if annotate_scales:
                T.annotate_layout({sa: make_linear_layout(sa), sb: make_linear_layout(sb)})
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa)
            T.copy(SB, sb)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True, scale_A=sa, scale_B=sb)
            T.copy(c, C)

    kernel = tilelang.compile(main, out_idx=[4])
    source = kernel.get_kernel_source()
    assert "a_trans" not in source and "b_trans" not in source
    assert ("tl::tix_ldmatrix_swzl" in source) == (layout_fn is not make_linear_layout)
    assert source.count("tl::tix_ldmatrix_x1_trans(") == (0 if annotate_scales else 2)
    generator = torch.Generator().manual_seed(17)
    packed, scales, values = [], [], []
    for _ in range(2):
        codes = torch.randint(0, 16, (64, width), generator=generator, dtype=torch.uint8)
        packed.append((codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.int8).cuda())
        exponents = torch.randint(125, 130, (64, width // 32), generator=generator)
        scales.append((exponents[:, ::2] | (exponents[:, 1::2] << 8)).T.contiguous().to(torch.uint16).cuda())
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes.long() & 7]
        decoded = magnitude * torch.where(codes & 8 != 0, -1.0, 1.0)
        values.append(decoded * torch.pow(2.0, exponents - 127).repeat_interleave(32, dim=1))
    actual = kernel(*packed, *scales)
    # Match the existing MXFP4 GEMM tolerance for native scaled MMA accumulation.
    torch.testing.assert_close(actual.cpu(), values[0] @ values[1].T, rtol=2e-2, atol=5e-1)


@pytest.mark.parametrize(
    "arch,dtype,layout_name,transposed,error",
    [
        (10, "float16", "linear", True, True),
        (10, "bfloat16", "linear", True, True),
        (10, "float16", "swizzled", True, True),
        (10, "float16", "chained", True, False),
        (10, "bfloat16", "default", True, False),
        (10, "float16", "linear", False, False),
        (15, "float16", "linear", True, False),
    ],
)
def test_chained_gemm_layout_contract(arch, dtype, layout_name, transposed, error):
    @T.prim_func
    def main(A: T.Tensor((16, 16), dtype), B: T.Tensor((16, 16), dtype), C: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((16, 16), dtype)
            b = T.alloc_shared((16, 16), dtype)
            c0 = T.alloc_fragment((16, 16), T.float32)
            a1 = T.alloc_fragment((16, 16), dtype)
            c1 = T.alloc_fragment((16, 16), T.float32)
            if layout_name == "linear":
                T.annotate_layout({b: make_linear_layout(b)})
            elif layout_name == "swizzled":
                T.annotate_layout({b: make_swizzled_layout(b)})
            elif layout_name == "chained":
                T.annotate_layout({b: make_ppu_swizzled_layout(b, k_major=True, is_gemm_rs=True)})
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, a, c0, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(16, 16):
                a1[i, j] = T.cast(c0[i, j], dtype)
            T.gemm(a1, b, c1, transpose_B=transposed, clear_accum=True)
            T.copy(c1, C)

    target = tvm.target.Target({"kind": "ppu", "arch": f"ppu_{arch}"})
    if error:
        expected = pytest.raises(ValueError, match="PPU0010 chained RS.*make_ppu_swizzled_layout")
    else:
        expected = nullcontext()
    with target, tvm.transform.PassContext(config={"tl.disable_aiu_lower": True}), expected:
        tilelang.lower(main, target=target, enable_device_compile=False)


@pytest.mark.parametrize("threads", [32, 128])
@pytest.mark.parametrize("disable_sync", [False, True])
def test_reused_shared_codegen_matches_cuda(threads, disable_sync):
    @T.prim_func
    def main(A: T.Tensor((16, 16), T.bfloat16), B: T.Tensor((16, 64), T.bfloat16), C: T.Tensor((2, 16, 64), T.float32)):
        with T.Kernel(1, threads=threads):
            a = T.alloc_fragment((16, 16), T.bfloat16)
            b = T.alloc_shared((16, 64), T.bfloat16)
            c = T.alloc_fragment((16, 64), T.float32)
            T.annotate_layout({b: make_linear_layout(b)})
            T.copy(A, a)
            for step in T.serial(2):
                for i, j in T.Parallel(16, 64):
                    b[i, j] = B[i, j] + T.bfloat16(step)
                T.sync_threads()
                T.gemm(a, b, c, clear_accum=True)
                T.copy(c, C[step, :, :])

    # The CUDA reference leg only runs on wheels that embed CUDA codegen
    # (PPU-only builds lack AnnotateDeviceBoundTmaCopies).
    targets = [("cuda", "sm_80")] if cuda_codegen_available() else []
    targets += [("ppu", "ppu_10"), ("ppu", "ppu_15")]
    for kind, arch in targets:
        target = tvm.target.Target({"kind": kind, "arch": arch})
        with target, tvm.transform.PassContext(config={"tl.disable_thread_storage_sync": disable_sync}):
            artifact = tilelang.lower(main, target=target, enable_device_compile=False)
        # No duplicate B allocation. Auto-sync protects the loop's WAR hazard;
        # disabling it leaves only the caller's explicit barrier, on both arches.
        functions = list(artifact.device_mod.functions.values())
        assert len(functions) == 1
        assert int(functions[0].attrs["dyn_shared_memory_buf"]) == 16 * 64 * 2
        assert artifact.kernel_source.count("__syncthreads();") == (1 if disable_sync else 2)
        assert "ldmatrix" in artifact.kernel_source
        assert "ldmatrix_swzl" not in artifact.kernel_source
