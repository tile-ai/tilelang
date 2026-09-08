"""pytest test for T.simd.vabsdif — fused abs-sub on Ascend SimdVF."""

import torch
import tilelang
import tilelang.testing
import tilelang.language as T

import pytest

NUM_BLOCKS = 64
NUM_THREADS = 2048
TILE = NUM_THREADS * 4
NUM_STAGES = 2


def vabsdif_kernel(N, dtype, backend="asc"):
    if N % (TILE * NUM_BLOCKS) != 0:
        raise ValueError(f"N must be a multiple of {TILE * NUM_BLOCKS}, got {N}")

    TOTAL_TILES = N // (TILE * NUM_BLOCKS)
    vec_len = 64 if dtype == "float32" else 128

    @T.prim_func
    def main(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype),
        C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            tA = T.alloc_shared((TILE,), dtype)
            tB = T.alloc_shared((TILE,), dtype)
            tC = T.alloc_shared((TILE,), dtype)
            T.annotate_buffer_versions({tA: NUM_STAGES, tB: NUM_STAGES, tC: NUM_STAGES})

            for iter in T.Pipelined(TOTAL_TILES, num_stages=NUM_STAGES):
                begin = (iter * NUM_BLOCKS + bx) * TILE
                end = (iter * NUM_BLOCKS + bx + 1) * TILE

                T.copy(A[begin:end], tA)
                T.copy(B[begin:end], tB)
                with T.SimdVF():
                    if backend == "pto":
                        mask = T.vmi.create_mask(vec_len, size=vec_len)
                        for i in range(TILE // vec_len):
                            r0 = T.vmi.vload(tA[i * vec_len], size=vec_len)
                            r1 = T.vmi.vload(tB[i * vec_len], size=vec_len)
                            T.vmi.vstore(
                                T.vmi.vabs(T.vmi.vsub(r0, r1, mask), mask),
                                tC[i * vec_len],
                                mask,
                            )
                    else:
                        elem_width = 32 if dtype == "float32" else 16
                        mask = T.simd.pset(elem_width)
                        for i in range(TILE // vec_len):
                            r0 = T.simd.vld(tA[i * vec_len])
                            r1 = T.simd.vld(tB[i * vec_len])
                            r2 = T.simd.vabsdif(r0, r1, mask)
                            T.simd.vsts(tC[i * vec_len], r2, mask)
                T.copy(tC, C[begin:end])

    return main


def _test_dtype(N, dtype, backend):
    kernel = tilelang.compile(vabsdif_kernel(N, dtype, backend), target=backend, out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=getattr(torch, dtype), device="cpu").to(device)
    b = torch.randn(N, dtype=getattr(torch, dtype), device="cpu").to(device)
    c = kernel(a, b)
    torch.npu.synchronize()
    c_ref = torch.abs(a.cpu() - b.cpu())
    assert torch.equal(c.cpu(), c_ref), f"{backend} vabsdif {dtype} mismatch"


@pytest.mark.parametrize(
    ("backend", "n"),
    [
        ("asc", 2**20),
        pytest.param("pto", 2**20, marks=pytest.mark.pto),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_simdvf_vabsdif(dtype, backend, n):
    _test_dtype(N=n, dtype=dtype, backend=backend)


if __name__ == "__main__":
    tilelang.testing.main()
