"""Register-count search must consider scalar ownership without forcing it."""

import importlib.util
from pathlib import Path

import pytest
import torch
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def make_reducer(**kwargs):
    path = Path(__file__).resolve().parents[3] / "maint/layout_inference/cases/reducer_scalar_candidates.py"
    spec = importlib.util.spec_from_file_location("reducer_scalar_candidates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_reducer(**kwargs)


def infer_reducer(model="register-count", factory=make_reducer, **kwargs):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        mod = tvm.IRModule({"main": factory(**kwargs)})
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        configs = {} if model is None else {"tl.layout_cost_model": model}
        with tl.transform.PassContext(config=configs):
            mod = tl.transform.LayoutInference()(mod)
    layouts = {}

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buffer, layout in node.annotations.get("layout_map", {}).items():
                layouts[buffer.name] = layout

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return layouts


@tilelang.testing.requires_cuda
def test_register_count_considers_scalar_column_ownership():
    native = infer_reducer(width=4)
    scalar = infer_reducer()
    default = infer_reducer(model=None)
    assert int(native["acc"].combine_size) == 4
    assert int(scalar["acc"].combine_size) == 1
    assert default["values"].is_equal(scalar["values"])


@tilelang.testing.requires_cuda
def test_full_reduction_keeps_native_packed_layout():
    native = infer_reducer(total=True, pinned=True)
    automatic = infer_reducer(total=True)
    assert int(automatic["acc"].combine_size) == 128
    assert automatic["values"].is_equal(native["values"])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kwargs", [{"width": 4}, {"pinned": True}, {"pin_loop": True}])
def test_explicit_layout_constraints_win(kwargs):
    automatic = infer_reducer(**kwargs)
    pinned = infer_reducer(width=4)
    assert int(automatic["acc"].combine_size) == 4
    assert automatic["values"].is_equal(pinned["values"])


@tilelang.testing.requires_cuda
def test_scalar_plan_cap_does_not_leak():
    before = infer_reducer(total=True)
    infer_reducer()
    after = infer_reducer(total=True)
    assert before["values"].is_equal(after["values"])
    assert before["acc"].is_equal(after["acc"])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("updates", [0, 2])
def test_zero_and_multiple_update_sites(updates):
    layouts = infer_reducer(updates=updates)
    assert int(layouts["acc"].combine_size) == (128 if updates == 0 else 1)


@tilelang.testing.requires_cuda
def test_io_aware_retains_its_candidate_search():
    native = infer_reducer(model="io-aware", width=4)
    automatic = infer_reducer(model="io-aware")
    assert automatic["values"].is_equal(native["values"])
    assert int(automatic["acc"].combine_size) == 4


def make_dynamic_full_reduction():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n, 512), T.float32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(n, threads=64) as row:
            values = T.alloc_fragment((512,), T.float32)
            acc = T.alloc_reducer((1,), T.float32)
            result = T.alloc_fragment((1,), T.float32)
            T.copy(A[row, :], values)
            T.reducer_init(acc)
            for i in T.Parallel(512):
                T.reducer_update(acc[0], values[i])
            T.finalize_reducer(acc, result)
            T.copy(result, B[row : row + 1])

    return main


@tilelang.testing.requires_cuda
def test_dynamic_shape_does_not_favor_replicated_fragments():
    layouts = infer_reducer(factory=make_dynamic_full_reduction)
    assert int(layouts["values"].replicate_size) == 1
    assert int(layouts["values"].get_output_shape()[0]) == 8
    assert int(layouts["acc"].combine_size) == 64


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("total", [False, True])
def test_scalar_candidate_kernel_correctness(total):
    kernel = tl.compile(make_reducer(total=total), target="cuda", pass_configs={"tl.layout_cost_model": "register-count"})
    if total:
        inputs = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
        expected = inputs.float().sum().reshape(1)
    else:
        inputs = torch.randn((8, 128), device="cuda", dtype=torch.float32)
        expected = inputs.sum(dim=0)
        assert "tl::AllReduce<" not in kernel.get_kernel_source()
    output = torch.empty_like(expected)
    kernel(inputs, output)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
