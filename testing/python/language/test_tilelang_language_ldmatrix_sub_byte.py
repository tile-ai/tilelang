"""White-box tests for the sub-byte ldmatrix variants (su4 / su6).

``ldmatrix.m8n16.x4.shared.b8x16.b4x16_p64`` / ``b6x16_p32`` unpack 16
packed 4-/6-bit elements per 16-byte source row into 8-bit register
containers. Hardware semantics pinned here (verified on RTX PRO 6000,
matching a raw-CUDA probe):

- destination fragment layout is identical to the classic ``m8n8.b16``
  form: lane ``l`` holds bytes ``4*(l%4)..4*(l%4)+3`` of row ``l//4``,
  register ``r`` is matrix ``r`` - so the 8-bit lane-offset functions are
  shared;
- su4 payload: 8 bytes, element 0 in the LOW nibble; 8 padding bytes
  ignored;
- su6 payload: 12 bytes, LSB-first 6-bit stream; 4 padding bytes ignored;
- values are zero-extended (su4: bits[3:0], su6: bits[5:0]).
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import simplify_prim_func


@simplify_prim_func
def _make_ldmatrix_probe_kernel(variant: str):
    @T.prim_func
    def main(
        SRC: T.Tensor((512,), T.uint8),
        OUT: T.Tensor((32, 16), T.uint8),
    ):
        with T.Kernel(1, threads=32) as _:
            tx = T.get_thread_binding()
            smem = T.alloc_shared((512,), T.uint8, scope="shared.dyn")
            regs = T.alloc_local((16,), T.uint8)

            for i in T.serial(16):
                smem[tx * 16 + i] = SRC[tx * 16 + i]
            T.sync_threads()

            T.ptx_ldmatrix(
                T.bool(False),
                4,
                T.access_ptr(smem[(tx % 8) * 16 + (tx // 8) * 128], "r", extent=16),
                T.access_ptr(regs[0], "w", extent=16),
                variant=variant,
            )

            for i in T.serial(16):
                OUT[tx, i] = regs[i]

    return main


def _pack_su4(elements):
    """elements: (4 matrices, 8 rows, 16 elems) -> 512 source bytes."""
    import torch

    src = torch.full((512,), 0xEE, dtype=torch.uint8)  # padding sentinel
    for mat in range(4):
        for row in range(8):
            base = mat * 128 + row * 16
            for j in range(8):
                e0 = int(elements[mat, row, 2 * j])
                e1 = int(elements[mat, row, 2 * j + 1])
                src[base + j] = e0 | (e1 << 4)
    return src


def _pack_su6(elements):
    import torch

    src = torch.full((512,), 0xEE, dtype=torch.uint8)
    for mat in range(4):
        for row in range(8):
            base = mat * 128 + row * 16
            payload = bytearray(12)
            for i in range(16):
                v = int(elements[mat, row, i])
                bit = 6 * i
                for b in range(6):
                    if v & (1 << b):
                        payload[(bit + b) // 8] |= 1 << ((bit + b) % 8)
            for j in range(12):
                src[base + j] = payload[j]
    return src


def _expected_fragment(elements):
    """Classic m8n8.b16-shaped fragment: lane l -> row l//4, 4 bytes at 4*(l%4)."""
    import torch

    out = torch.zeros((32, 16), dtype=torch.uint8)
    for lane in range(32):
        row = lane // 4
        col0 = 4 * (lane % 4)
        for mat in range(4):
            for j in range(4):
                out[lane, mat * 4 + j] = elements[mat, row, col0 + j]
    return out


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("variant,max_value,pack", [("su4", 16, _pack_su4), ("su6", 64, _pack_su6)])
def test_ldmatrix_sub_byte_lane_mapping(variant, max_value, pack):
    import torch

    torch.manual_seed(0)
    kernel = tilelang.compile(_make_ldmatrix_probe_kernel(variant), target="cuda", out_idx=[1])
    src = kernel.get_kernel_source()
    assert f"tl::ptx_ldmatrix_{variant}_x4" in src

    elements = torch.randint(0, max_value, (4, 8, 16), dtype=torch.int64)
    out = kernel(pack(elements).cuda())
    expected = _expected_fragment(elements)
    assert torch.equal(out.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
