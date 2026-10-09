"""Mutable SIMD reads and immutable snapshots observe the correct version."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
import tilelang.testing
import torch
from tvm import tirx
from testing.ascend._ir import calls, nodes


@tilelang.jit(
    target="ascend",
    pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
)
def vselr_prefix_sum_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), T.float32), dst: T.Tensor((64,), T.float32)) -> None:
        with T.Kernel(1):
            src_ub = T.alloc_shared((64,), T.float32)
            dst_ub = T.alloc_shared((64,), T.float32)
            T.copy(src, src_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                zero = T.simd.vdup(0.0, "float32", full)
                lane = T.simd.vci(0, "int32", "INC_ORDER")
                x = T.simd.alloc_local((1,), "float32")
                x[0] = T.simd.vld(src_ub[0])
                shifted_1 = T.simd.vsel(
                    T.simd.vselr(x[0], T.simd.vadds(lane, -1, full)),
                    zero,
                    T.simd.vcmps(lane, 1, full, "ge"),
                )
                x[0] = T.simd.vadd(x[0], shifted_1, full)
                shifted_2 = T.simd.vsel(
                    T.simd.vselr(x[0], T.simd.vadds(lane, -2, full)),
                    zero,
                    T.simd.vcmps(lane, 2, full, "ge"),
                )
                x[0] = T.simd.vadd(x[0], shifted_2, full)
                T.simd.vsts(dst_ub[0], x[0], full)
            T.copy(dst_ub, dst)

    return main


@tilelang.jit(
    target="ascend",
    pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
)
def bound_buffer_load_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), T.float32), dst: T.Tensor((64,), T.float32)) -> None:
        with T.Kernel(1):
            src_ub = T.alloc_shared((64,), T.float32)
            dst_ub = T.alloc_shared((64,), T.float32)
            T.copy(src, src_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                x = T.simd.alloc_local((1,), "float32")
                x[0] = T.simd.vld(src_ub[0])
                snapshot = x[0]
                x[0] = T.simd.vadds(x[0], 10.0, full)
                T.simd.vsts(dst_ub[0], snapshot, full)
            T.copy(dst_ub, dst)

    return main


def test_simdvf_vselr_reloads_mutated_local():
    src = torch.arange(1, 65, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    kernel = vselr_prefix_sum_kernel()

    kernel(src, dst)
    torch.npu.synchronize()
    expected = src.cpu().clone()
    for offset in (1, 2, 3):
        expected[offset:] += src.cpu()[:-offset]
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_simdvf_bind_materializes_buffer_load_snapshot():
    src = torch.arange(1, 65, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    kernel = bound_buffer_load_kernel()

    kernel(src, dst)
    torch.npu.synchronize()
    assert torch.equal(dst.cpu(), src.cpu())


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "dtype,dist,step",
    [("uint8", "NORM", 256), ("int16", "NORM", -128), ("float32", "NORM", 64), ("int16", "BRC_B16", 1), ("int16", "E2B_B16", 8)],
)
def test_simdvf_postupdate_load_threads_pointer_and_snapshots(dtype, dist, step):
    lanes = 256 // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    chunk = lanes if dist == "NORM" else (1 if dist == "BRC_B16" else 8)
    bits = 2048 // lanes

    @T.prim_func
    def main(A: T.Tensor((3, 3 * lanes), dtype), B: T.Tensor((3, 3 * lanes), dtype), increment: T.int32):
        with T.Kernel(1):
            src = T.alloc_shared((3 * lanes,), dtype)
            dst = T.alloc_shared((3 * lanes,), dtype)
            T.annotate_buffer_versions({src: 2, dst: 2})
            for tile in T.Pipelined(3, num_stages=2):
                with T.Stage(0):
                    T.copy(A[tile, :], src)
                with T.Stage(0), T.SimdVF():
                    full = S.pset(bits)
                    ptr = S.make_ubuf_ptr(T.access_ptr(src[0], "r", extent=3 * lanes), dtype)
                    first, ptr = S.vld(ptr, dist, post_inc=chunk)
                    second, ptr = S.vld(ptr, dist, post_inc=increment)
                    S.vsts(src[0], S.vadds(first, 7, full), full, dist=f"NORM_B{bits}")
                    S.mem_bar("VST_VLD")
                    last, ptr = S.vld(ptr, dist, post_inc=0)
                    S.vsts(dst[0], first, full, dist=f"NORM_B{bits}")
                    S.vsts(dst[lanes], second, full, dist=f"NORM_B{bits}")
                    S.vsts(dst[2 * lanes], last, full, dist=f"NORM_B{bits}")
                with T.Stage(0):
                    T.copy(dst, B[tile, :])

    loads = calls(main, "tl.simd.vld")
    bindings = [node for node in nodes(main, tirx.Bind) if any(node.value.same_as(load) for load in loads)]
    projections = calls(main, "tl.simd.pair_get")
    assert len(loads) == len(bindings) == 3
    for binding in bindings:
        assert sorted(int(call.args[1]) for call in projections if call.args[0].same_as(binding.var)) == [0, 1]

    kernel = tilelang.compile(main, target="ascend")
    source = kernel.get_kernel_source()
    # Both tuple elements must share each opaque load, including the zero step.
    assert source.count("_postupdate<") == 3
    values = (torch.arange(9 * lanes).reshape(3, 3 * lanes) % 113).to(getattr(torch, dtype))
    if dist == "NORM":
        indices = torch.arange(lanes)
    elif dist == "BRC_B16":
        indices = torch.zeros(lanes, dtype=torch.int64)
    else:
        indices = torch.arange(8).repeat_interleave(lanes // 8)
    first, second = values[:, indices], values[:, chunk + indices]
    # The negative NORM step returns to src[0], checking reload after the store.
    modified = values.clone()
    modified[:, :lanes] = first + 7
    expected = torch.cat((first, second, modified[:, chunk + step + indices]), dim=1)
    output = torch.empty_like(values, device="npu")
    kernel(values.npu(), output, step)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
