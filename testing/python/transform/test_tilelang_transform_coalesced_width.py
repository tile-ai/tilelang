from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
import tilelang as tl
import tilelang.language as T


auto_target = tvm.target.Target(determine_target("auto"))


def _lower_without_device_compile(func):
    with tvm.target.Target(auto_target):
        return tl.lower(func, target=auto_target, enable_device_compile=False)


def test_ragged_copy_coalesced_width_clamps_with_warning(capfd):
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

    artifact = _lower_without_device_compile(main)

    warnings = capfd.readouterr().err
    assert "Requested coalesced_width=2" in warnings
    assert "using 1 instead" in warnings
    assert artifact.kernel_source


def test_full_tile_oversized_coalesced_width_clamps_with_warning(capfd):
    m, n = 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((m, n), T.float32),
        B: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((m, n), T.float32)
            T.copy(A, shared, coalesced_width=257)
            T.copy(shared, B, coalesced_width=257)

    artifact = _lower_without_device_compile(main)

    warnings = capfd.readouterr().err
    assert "Requested coalesced_width=257" in warnings
    assert "using " in warnings
    assert artifact.kernel_source
