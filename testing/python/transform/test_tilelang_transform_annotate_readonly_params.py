import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import IRModule


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
    mod = IRModule.from_expr(bare_data_var_write)
    mod_annotated = tilelang.transform.AnnotateReadOnlyParams()(mod)
    readonly_indices = mod_annotated["bare_data_var_write"].attrs.get("tl.readonly_param_indices")
    # Only A (param 0) should be readonly; Out (param 1) is written via bare data var
    assert list(readonly_indices) == [0], f"Expected [0], got {list(readonly_indices)}"

    # Test address_of write (control case, should work the same)
    mod = IRModule.from_expr(address_of_write)
    mod_annotated = tilelang.transform.AnnotateReadOnlyParams()(mod)
    readonly_indices = mod_annotated["address_of_write"].attrs.get("tl.readonly_param_indices")
    assert list(readonly_indices) == [0], f"Expected [0], got {list(readonly_indices)}"

    # Test pure intrinsic with bare data var does NOT mark A as written
    mod = IRModule.from_expr(pure_intrinsic_with_bare_data)
    mod_annotated = tilelang.transform.AnnotateReadOnlyParams()(mod)
    readonly_indices = mod_annotated["pure_intrinsic_with_bare_data"].attrs.get("tl.readonly_param_indices")
    # A should remain readonly; pure intrinsics don't write even if they receive bare data vars
    assert list(readonly_indices) == [0], f"Expected [0], got {list(readonly_indices)}"


if __name__ == "__main__":
    tilelang.testing.main()
