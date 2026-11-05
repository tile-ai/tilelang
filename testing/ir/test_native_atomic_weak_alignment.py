"""The original indirect-row kernel remains correct for weakly aligned inputs."""

import pytest
import tilelang
import torch

from testing.python.language.test_tilelang_language_atomic import atomic_add_dynamic_rows_program


@pytest.mark.parametrize("indirect", [False, True])
def test_original_kernel_default_alignment_with_offset_inputs(indirect):
    # No original-vector-alignment fixture: these are the public Tensor
    # annotation's default alignment=1 and the existing, unchanged DSL body.
    # The width test and this offset test use different annotation alignment
    # guarantees. Give each run its own JIT factory so an earlier compiled
    # specialization cannot cross those harness configurations.
    program = tilelang.jit(atomic_add_dynamic_rows_program.func.orig_func)
    kernel = program(indirect)
    source = kernel.get_kernel_source()
    assert "AtomicAddx" not in source
    a_storage = torch.full((4 * 256 + 2,), -17, dtype=torch.bfloat16, device="cuda")
    a = a_storage[1:-1].view(4, 256)
    a.copy_((torch.arange(a.numel(), device="cuda").reshape_as(a) % 4 + 1).to(a.dtype))
    b_storage = torch.full((8 * 256 + 2,), -31, dtype=torch.float32, device="cuda")
    b = b_storage[1:-1].view(8, 256)
    b.zero_()
    index_storage = torch.tensor([-43, 0, 1, 0, 1, -43], dtype=torch.int64, device="cuda")
    indices = index_storage[1:-1]
    before_a = a_storage.clone()
    before_indices = index_storage.clone()
    assert a.data_ptr() % 16 != 0
    assert b.data_ptr() % 16 != 0
    assert indices.data_ptr() % 16 != 0
    kernel(a, b, indices)
    expected = torch.zeros_like(b)
    expected.index_add_(0, indices if indirect else torch.arange(4, device="cuda"), a.float())
    torch.testing.assert_close(b, expected, atol=0, rtol=0)
    torch.testing.assert_close(a_storage, before_a, atol=0, rtol=0)
    torch.testing.assert_close(index_storage, before_indices, atol=0, rtol=0)
    assert torch.equal(b_storage[[0, -1]], torch.full((2,), -31, dtype=torch.float32, device="cuda"))
