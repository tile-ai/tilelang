"""Nullable data and shape carriers use the actual native packed entry."""

import pytest
import torch
import tvm_tirx

import tilelang
import tilelang.cpu.language as T


def compile_cpu(function, outputs=None, *, callee=False, kind="c"):
    if kind == "llvm" and not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")
    if callee:
        function = function.with_attr("tilelang_callee_allocated_outputs", True)
    return tilelang.compile(
        function,
        out_idx=outputs,
        execution_backend="tvm_ffi",
        target=tvm_tirx.target.Target({"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})}),
        target_host=tvm_tirx.target.Target({"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})}),
    )


def shared_shape():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((1,), "int64")):
        with T.Kernel(1):
            C[0] = n

    return main


@pytest.mark.parametrize("kind", ["llvm", "c"])
def test_shape_carrier_is_selected_per_call(kind):
    kernel = compile_cpu(shared_shape(), kind=kind)
    output = torch.empty(1, dtype=torch.int64)
    for size, first, second in [(7, True, False), (19, False, True), (3, True, True)]:
        a = torch.empty(size) if first else None
        b = torch.empty(size) if second else None
        kernel(a, b, output)
        assert output.item() == size
    with pytest.raises(RuntimeError, match="at least one non-null buffer"):
        kernel(None, None, output)
    with pytest.raises(RuntimeError):
        kernel(torch.empty(7), torch.empty(9), output)
    kernel(None, torch.empty(23), output)
    assert output.item() == 23


def test_single_missing_shape_carrier_keeps_zero_extent():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((1,), "int64")):
        with T.Kernel(1):
            C[0] = n

    kernel = compile_cpu(main)
    output = torch.empty(1, dtype=torch.int64)
    kernel(None, output)
    assert output.item() == 0
    kernel(torch.empty(13), output)
    assert output.item() == 13


def test_all_unused_shape_carriers_may_be_absent():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            C[0] = 6.0

    output = torch.empty(1)
    compile_cpu(main)(None, None, output)
    assert output.item() == 6.0


def test_used_tensor_is_required_even_if_another_carrier_exists():
    @T.prim_func
    def main(A: T.Tensor((1,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            C[0] = A[0] + 1.0

    kernel = compile_cpu(main)
    with pytest.raises((TypeError, RuntimeError), match="Tensor|non-NULL"):
        kernel(None, torch.empty(1))
    # Object handles must be checked before interpreting them as a DLTensor.
    with pytest.raises(RuntimeError, match="Tensor"):
        kernel.adapter._get_executable()(tvm_tirx.ir.const(1), torch.empty(1))


def test_late_scalar_can_supply_shape_for_missing_tensor():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((1,), "int64")):
        with T.Kernel(1):
            C[0] = n

    function = tvm_tirx.tirx.Function([*main.params, n], main.body, main.ret_type, main.attrs)
    kernel = compile_cpu(function)
    output = torch.empty(1, dtype=torch.int64)
    kernel(None, output, 11)
    assert output.item() == 11


def test_callee_outputs_use_nullable_input_shape_before_allocation():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(C: T.Tensor((n,), "int32"), A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(n):
                C[i] = T.int32(i) + 2

    kernel = compile_cpu(main, [0], callee=True)
    for size in (0, 1, 17):
        output = kernel(None, torch.empty(size))
        torch.testing.assert_close(output, torch.arange(size, dtype=torch.int32) + 2, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="at least one non-null buffer"):
        kernel(None, None)


@pytest.mark.parametrize("kind", ["llvm", "c"])
@pytest.mark.parametrize("bare_first", [True, False])
def test_bare_and_linear_shape_carriers_share_availability(kind, bare_first):
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((2 * n + 1,), "float32"), C: T.Tensor((1,), "int64")):
        with T.Kernel(1):
            C[0] = n

    if not bare_first:
        main = tvm_tirx.tirx.Function([main.params[1], main.params[0], main.params[2]], main.body, main.ret_type, main.attrs)
    kernel = compile_cpu(main, kind=kind)
    output = torch.empty(1, dtype=torch.int64)

    def invoke(bare, linear):
        args = (bare, linear) if bare_first else (linear, bare)
        kernel(*args, output)

    for size in (0, 3, 17):
        bare, linear = torch.empty(size), torch.empty(2 * size + 1)
        for first, second in ((bare, None), (None, linear), (bare, linear)):
            invoke(first, second)
            assert output.item() == size
    with pytest.raises(RuntimeError, match="at least one non-null buffer"):
        invoke(None, None)
    with pytest.raises(RuntimeError):
        invoke(torch.empty(3), torch.empty(9))
    with pytest.raises(RuntimeError, match="divisible"):
        invoke(None, torch.empty(8))
    invoke(None, torch.empty(11))
    assert output.item() == 5
