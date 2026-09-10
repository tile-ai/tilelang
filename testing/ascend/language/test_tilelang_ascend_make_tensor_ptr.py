"""Tests for T.make_tensor pointer-table reconstruction on Ascend."""

import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def make_tensor_copy(n):
    """Copy src addressed through a pointer table to out, staging through UB."""

    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1) as _:
            src = T.make_tensor(src_ptrs[0], (n,), "float32")
            buf = T.alloc_shared(n, "float32")
            T.copy(src[0], buf)
            T.copy(buf, out)

    return main


def make_tensor_copy_both(n):
    """Both src and out are addressed through pointer tables."""

    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out_ptrs: T.Tensor((1,), T.ptr),
    ):
        with T.Kernel(1) as _:
            src = T.make_tensor(src_ptrs[0], (n,), "float32")
            out = T.make_tensor(out_ptrs[0], (n,), "float32")
            buf = T.alloc_shared(n, "float32")
            T.copy(src[0], buf)
            T.copy(buf, out[0])

    return main


def make_tensor_copy_guarded(n):
    """T.make_tensor under a runtime if guard."""

    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out: T.Tensor((n,), "float32"),
        sel: T.int32,
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared(n, "float32")
            if sel > 0:
                src = T.make_tensor(src_ptrs[0], (n,), "float32")
                T.copy(src[0], buf)
                T.copy(buf, out)

    return main


def make_ptr_table(tensors):
    device = tensors[0].device
    return torch.tensor([t.data_ptr() for t in tensors], device=device, dtype=torch.int64)


def test_make_tensor_ptr():
    n = 64
    kernel = tilelang.compile(make_tensor_copy(n), target="ascend")
    device = torch.device("npu")
    src = torch.randn(n, dtype=torch.float32, device=device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    src_ptrs = make_ptr_table([src])
    kernel(src_ptrs, out)
    torch.npu.synchronize()
    assert torch.equal(out, src)


def test_make_tensor_ptr_both():
    n = 64
    kernel = tilelang.compile(make_tensor_copy_both(n), target="ascend")
    device = torch.device("npu")
    src = torch.randn(n, dtype=torch.float32, device=device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    kernel(make_ptr_table([src]), make_ptr_table([out]))
    torch.npu.synchronize()
    assert torch.equal(out, src)


def test_make_tensor_ptr_guarded():
    n = 64
    kernel = tilelang.compile(make_tensor_copy_guarded(n), target="ascend")
    device = torch.device("npu")
    src = torch.randn(n, dtype=torch.float32, device=device)
    src_ptrs = make_ptr_table([src])

    out = torch.zeros(n, dtype=torch.float32, device=device)
    kernel(src_ptrs, out, 1)
    torch.npu.synchronize()
    assert torch.equal(out, src)

    out = torch.zeros(n, dtype=torch.float32, device=device)
    kernel(src_ptrs, out, 0)
    torch.npu.synchronize()
    assert torch.equal(out, torch.zeros(n, dtype=torch.float32, device=device))


if __name__ == "__main__":
    tilelang.testing.main()
