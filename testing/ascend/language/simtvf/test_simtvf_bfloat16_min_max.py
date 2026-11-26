"""Regression tests for bfloat16 scalar min/max codegen in Ascend SIMT blocks.

CCE provides no min(bfloat16_t, bfloat16_t) overload, so a plain min()/max()
call on bfloat16 values is ambiguous across the float/half/integer overloads
and fails device compilation (issue #4143). The codegen routes bfloat16
min/max through the float overloads, which lower to __builtin_cce_fminf /
__builtin_cce_fmaxf (the same fmin/fmax semantics the float16 codegen gets),
and converts the result back to bfloat16.
"""

import torch

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _source(func, *, target):
    with tilelang.tvm.target.Target(target), tilelang.transform.PassContext(config={}):
        return tilelang.lower(func, target=target).kernel_source


def _scalar_kernel(num_elements):
    @T.prim_func
    def main(
        a_bfloat: T.Tensor((num_elements,), "bfloat16"),
        b_bfloat: T.Tensor((num_elements,), "bfloat16"),
        out_min: T.Tensor((num_elements,), "bfloat16"),
        out_max: T.Tensor((num_elements,), "bfloat16"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((num_elements,), "bfloat16")
            b_ub = T.alloc_shared((num_elements,), "bfloat16")
            out_min_ub = T.alloc_shared((num_elements,), "bfloat16")
            out_max_ub = T.alloc_shared((num_elements,), "bfloat16")
            T.copy(a_bfloat, a_ub)
            T.copy(b_bfloat, b_ub)
            with T.SimtVF(threads=num_elements):
                tx = T.get_thread_binding()
                out_min_ub[tx] = T.min(a_ub[tx], b_ub[tx])
                out_max_ub[tx] = T.max(a_ub[tx], b_ub[tx])
            T.copy(out_min_ub, out_min)
            T.copy(out_max_ub, out_max)

    return main


def _vectorized_kernel(num_elements, lanes):
    @T.prim_func
    def main(
        a_bfloat: T.Tensor((num_elements,), "bfloat16"),
        b_bfloat: T.Tensor((num_elements,), "bfloat16"),
        out_min: T.Tensor((num_elements,), "bfloat16"),
        out_max: T.Tensor((num_elements,), "bfloat16"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((num_elements,), "bfloat16")
            b_ub = T.alloc_shared((num_elements,), "bfloat16")
            out_min_ub = T.alloc_shared((num_elements,), "bfloat16")
            out_max_ub = T.alloc_shared((num_elements,), "bfloat16")
            T.copy(a_bfloat, a_ub)
            T.copy(b_bfloat, b_ub)
            with T.SimtVF(threads=num_elements // lanes):
                lane = T.get_thread_binding()
                for i in T.vectorized(lanes):
                    index = lane * lanes + i
                    out_min_ub[index] = T.min(a_ub[index], b_ub[index])
                    out_max_ub[index] = T.max(a_ub[index], b_ub[index])
            T.copy(out_min_ub, out_min)
            T.copy(out_max_ub, out_max)

    return main


def test_scalar_min_max_use_float_overload():
    source = _source(_scalar_kernel(64), target="ascend")
    assert "(bfloat16_t)min((float)(" in source
    assert "(bfloat16_t)max((float)(" in source


def test_vectorized_min_max_use_float_overload():
    source = _source(_vectorized_kernel(64, 2), target="ascend")
    assert "(bfloat16_t)min((float)(" in source
    assert "(bfloat16_t)max((float)(" in source


@tilelang.testing.requires_ascend
def test_scalar_min_max_match_torch():
    num_elements = 64
    func = _scalar_kernel(num_elements)
    a = torch.randn(num_elements, device="npu").to(torch.bfloat16)
    b = torch.randn(num_elements, device="npu").to(torch.bfloat16)
    out_min = torch.empty(num_elements, device="npu", dtype=torch.bfloat16)
    out_max = torch.empty(num_elements, device="npu", dtype=torch.bfloat16)
    tilelang.compile(func, target="ascend")(a, b, out_min, out_max)
    assert torch.equal(out_min.cpu(), torch.minimum(a.cpu(), b.cpu()))
    assert torch.equal(out_max.cpu(), torch.maximum(a.cpu(), b.cpu()))


@tilelang.testing.requires_ascend
def test_vectorized_min_max_match_torch():
    num_elements = 64
    func = _vectorized_kernel(num_elements, 2)
    a = torch.randn(num_elements, device="npu").to(torch.bfloat16)
    b = torch.randn(num_elements, device="npu").to(torch.bfloat16)
    out_min = torch.empty(num_elements, device="npu", dtype=torch.bfloat16)
    out_max = torch.empty(num_elements, device="npu", dtype=torch.bfloat16)
    tilelang.compile(func, target="ascend")(a, b, out_min, out_max)
    assert torch.equal(out_min.cpu(), torch.minimum(a.cpu(), b.cpu()))
    assert torch.equal(out_max.cpu(), torch.maximum(a.cpu(), b.cpu()))
