import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


@pytest.fixture(
    params=[
        pytest.param({"kind": "cuda", "arch": "sm_80"}, marks=tilelang.testing.requires_cuda.marks(), id="cuda"),
        pytest.param({"kind": "hip", "mcpu": "gfx942"}, marks=tilelang.testing.requires_rocm.marks(), id="rocm"),
    ]
)
def copy_target(request):
    # These diagnostics come from SIMT copy vectorization. Ascend DMA copies
    # do not use the coalesced-width planner.
    return tvm.target.Target(request.param)


def _lower_without_device_compile(func, target):
    with target:
        return tl.lower(func, target=target, enable_device_compile=False)


def test_ragged_copy_coalesced_width_clamps_with_warning(capfd, copy_target):
    m, n = 32, 33
    block_m, block_n = 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((m, n), T.float32),
        B: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            shared = T.alloc_shared((block_m, block_n), T.float32)
            T.copy(A[by * block_m, bx * block_n], shared, coalesced_width=2)
            T.copy(shared, B[by * block_m, bx * block_n], coalesced_width=2)

    artifact = _lower_without_device_compile(main, copy_target)

    warnings = capfd.readouterr().err
    assert "Requested coalesced_width=2" in warnings
    assert "using 1 instead" in warnings
    assert artifact.kernel_source


@pytest.mark.parametrize("coalesced_width", [8, 257])
def test_full_tile_oversized_coalesced_width_clamps_with_warning(capfd, copy_target, coalesced_width):
    # A 128x128 fp32 tile over 128 threads vectorizes to 4 elements per thread;
    # any wider request must fall back to that width, not below it.
    m, n = 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((m, n), T.float32),
        B: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((m, n), T.float32)
            T.copy(A, shared, coalesced_width=coalesced_width)
            T.copy(shared, B, coalesced_width=coalesced_width)

    artifact = _lower_without_device_compile(main, copy_target)

    warnings = capfd.readouterr().err
    assert f"Requested coalesced_width={coalesced_width}" in warnings
    assert "using 4 instead" in warnings
    assert "float4" in artifact.kernel_source
