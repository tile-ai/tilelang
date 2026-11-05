import tilelang
import pytest
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import _make_parameter_alias_contract_explicit
from tvm_tirx import ir, tirx
from tvm_tirx.ir import IRModule


def test_annotate_readonly_bare_data_var():
    """Regression test for #2598: bare .data var passed to call_extern should not be marked readonly.

    When a parameter is written only by passing its bare .data pointer to an extern
    (e.g., call_extern("store", Out.data, value)), the pass must recognize this as
    a potential write and not mark the parameter as read-only.
    """

    @T.prim_func
    def bare_data_var_write(A: T.Tensor((256,), "float32"), Out: T.Tensor((256,), "float32")):
        with T.Kernel(256, threads=1) as i:
            T.evaluate(T.call_extern("void", "store_val", Out.data, A[i] + 1.0))

    @T.prim_func
    def address_of_write(A: T.Tensor((256,), "float32"), Out: T.Tensor((256,), "float32")):
        with T.Kernel(256, threads=1) as i:
            T.evaluate(T.call_extern("void", "store_val", T.address_of(Out[i]), A[i] + 1.0))

    @T.prim_func
    def pure_intrinsic_with_bare_data(A: T.Tensor((16,), "float32"), Out: T.Tensor((16,), "float32")):
        # tirx.isnullptr is pure and does not write A
        T.evaluate(T.call_intrin("bool", "tirx.isnullptr", A.data))
        Out[0] = A[0]

    # Test bare data var write via call_extern
    # The real backend pipeline first establishes TileLang's noalias default.
    # A standalone native Function makes no such promise: writes through Out
    # could then also affect A. Exercise this pass at its actual pipeline stage.
    mod = _make_parameter_alias_contract_explicit(IRModule.from_expr(bare_data_var_write))
    mod_annotated = tilelang.transform.AnnotateReadOnlyParams()(mod)
    readonly_indices = mod_annotated["bare_data_var_write"].attrs.get("tl.readonly_param_indices")
    # Only A (param 0) should be readonly; Out (param 1) is written via bare data var
    assert list(readonly_indices) == [0], f"Expected [0], got {list(readonly_indices)}"

    # Test address_of write (control case, should work the same)
    mod = _make_parameter_alias_contract_explicit(IRModule.from_expr(address_of_write))
    mod_annotated = tilelang.transform.AnnotateReadOnlyParams()(mod)
    readonly_indices = mod_annotated["address_of_write"].attrs.get("tl.readonly_param_indices")
    assert list(readonly_indices) == [0], f"Expected [0], got {list(readonly_indices)}"

    # Test pure intrinsic with bare data var does NOT mark A as written
    mod = _make_parameter_alias_contract_explicit(IRModule.from_expr(pure_intrinsic_with_bare_data))
    mod_annotated = tilelang.transform.AnnotateReadOnlyParams()(mod)
    readonly_indices = mod_annotated["pure_intrinsic_with_bare_data"].attrs.get("tl.readonly_param_indices")
    # A should remain readonly; pure intrinsics don't write even if they receive bare data vars
    assert list(readonly_indices) == [0], f"Expected [0], got {list(readonly_indices)}"


@pytest.mark.parametrize("noalias", [None, False])
def test_standalone_readonly_does_not_invent_alias_contract(noalias):
    source = ir.Var("source", tirx.TensorType((1,), "float32", layout="default"))
    output = ir.Var("output", source.ty)
    func = tirx.Function([source, output], ir.TensorStore(output, [0], source[0]), ret_type=ir.PrimType("void"))
    if noalias is not None:
        func = func.with_attr("tirx.noalias", noalias)
    annotated = tilelang.transform.AnnotateReadOnlyParams()(IRModule.from_expr(func))["main"]
    assert not list(annotated.attrs.get("tl.readonly_param_indices", []))
    assert ("tirx.noalias" in annotated.attrs) == (noalias is not None)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("noalias", [False, True])
def test_pipeline_alias_writes_and_readonly_metadata(noalias):
    import torch

    @T.prim_func
    def main(A: T.Tensor((1,), T.int32), B: T.Tensor((1,), T.int32), C: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            B[0] = A[0] + 2
            C[0] = A[0]

    kernel = tilelang.compile(main.with_attr("tirx.noalias", noalias), target="cuda", execution_backend="tvm_ffi")
    functions = list(kernel.artifact.device_mod.functions.values())
    assert len(functions) == 1
    function = functions[0]
    assert bool(function.attrs["tirx.noalias"]) == noalias
    assert [int(index) for index in function.attrs.get("tl.readonly_param_indices", [])] == ([0] if noalias else [])
    a = torch.full((1,), 5, dtype=torch.int32, device="cuda")
    b = torch.full_like(a, -1) if noalias else a
    c = torch.full_like(a, -1)
    kernel(a, b, c)
    assert b.item() == 7
    assert c.item() == (5 if noalias else 7)
    assert a.item() == (5 if noalias else 7)


if __name__ == "__main__":
    tilelang.testing.main()
