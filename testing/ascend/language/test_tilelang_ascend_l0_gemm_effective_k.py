"""L0 GEMM consumes effective regions while copies and storage may stay padded."""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import pytest
import torch
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit

M, N, K_ALLOC = 64, 64, 32


def _transposed_b_kernel():
    @T.prim_func
    def main(
        A: T.Buffer((M, K_ALLOC), "float32"),
        B: T.Buffer((K_ALLOC, N), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((M, K_ALLOC), "float32")
            b_l1 = T.alloc_l1((K_ALLOC, N), "float32")
            a_l0 = T.alloc_l0a((M, K_ALLOC), "float32")
            b_l0 = T.alloc_l0b((N, K_ALLOC), "float32")
            acc = T.alloc_l0c((M, N), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.set_hf32_mode("nearest_even")
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0, transpose=True)
            T.gemm(a_l0[:, :24], b_l0[:, :24], acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return main


def test_effective_k_keeps_padded_storage_budget():
    allocations = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureStorage:
        def run_after_pass(self, mod, info):
            if info.name != "tl.NormalizeAscendFractalStorage":
                return

            def visit(node):
                if isinstance(node, tirx.SBlock):
                    for buffer in node.alloc_buffers:
                        if buffer.scope().startswith("shared.l0"):
                            allocations[buffer.name] = tuple(int(dim) for dim in buffer.shape)

            post_order_visit(mod["main"].body, visit)

    with tvm.transform.PassContext(instruments=[CaptureStorage()]):
        source = tilelang.lower(_transposed_b_kernel(), target="ascend").kernel_source
    # AutoSchedule must see the full B32 allocation even though MAD reads K24.
    assert allocations["a_l0"] == (M, K_ALLOC)
    assert allocations["b_l0"] == (N, K_ALLOC)
    assert "asc_copy_l12l0b_transpose(" in source
    assert ", 64, 24, 64," in source


def test_effective_k_with_transposed_b_ignores_nonzero_padding():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    kernel = tilelang.compile(_transposed_b_kernel(), out_idx=-1, target="ascend")
    generator = torch.Generator().manual_seed(42)
    a = torch.randint(-4, 5, (M, K_ALLOC), generator=generator).float() / 8
    b = torch.randint(-4, 5, (K_ALLOC, N), generator=generator).float() / 8
    a[:, 24:] = 2
    b[24:, :] = 3
    expected = a[:, :24] @ b[:24, :]
    actual = kernel(a.npu(), b.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
