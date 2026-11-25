"""Real TileLang post-update stores preserve data, masks and caller pointers."""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "mask_kind,dtype",
    [
        ("empty", "uint8"),
        ("one", "uint8"),
        ("partial", "uint8"),
        ("full", "uint8"),
        ("partial", "int16"),
        ("partial", "bfloat16"),
        ("partial", "float32"),
    ],
)
def test_vsstb_advances_pointer_across_masked_stores(dtype, mask_kind):
    td = getattr(torch, dtype)
    lanes = 256 // torch.empty((), dtype=td).element_size()
    bits = 2048 // lanes
    count = {"empty": 0, "one": 1, "partial": lanes // 2 + 1, "full": lanes}[mask_kind]

    @T.prim_func
    def kernel(A: T.Tensor((3 * lanes,), dtype), Initial: T.Tensor((5 * lanes,), dtype), Out: T.Tensor((5 * lanes,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((3 * lanes,), dtype)
            dst = T.alloc_shared((5 * lanes,), dtype)
            T.copy(A, src)
            T.copy(Initial, dst)
            with T.SimdVF():
                full = T.simd.pset(bits)
                first_mask = T.simd.update_mask(T.uint32(count), bits)
                a = T.simd.vld(src[0])
                b = T.simd.vld(src[lanes])
                c = T.simd.vld(src[2 * lanes])
                ptr = T.simd.make_ubuf_ptr(T.access_ptr(dst[lanes], "w", extent=3 * lanes), dtype)
                ptr = T.simd.vsstb(a, ptr, T.int32((1 << 16) | 8), first_mask, update=True)
                ptr = T.simd.vsstb(b, ptr, T.int32((1 << 16) | 8), full, update=True)
                ptr = T.simd.vsstb(c, ptr, T.int32((1 << 16) | 8), full, update=True)
                T.simd.mem_bar("VST_VLD")
            T.copy(dst, Out)

    compiled = tilelang.compile(kernel, target="ascend")
    values = (torch.arange(3 * lanes) % 61 + 1).to(td)
    for sentinel in (91, 117):
        initial = torch.full((5 * lanes,), sentinel, dtype=td)
        expected = initial.clone()
        expected[lanes : lanes + count] = values[:count]
        expected[2 * lanes : 4 * lanes] = values[lanes:]
        out = torch.empty_like(initial, device="npu")
        compiled(values.to("npu"), initial.to("npu"), out)
        actual = out.cpu()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (dtype, mask_kind)
