import pytest

from tilelang import tvm
import tilelang.language as T


@T.macro
def _copy(A, B, n):
    for i in T.Parallel(n):
        B[i] = A[i]


def test_build_prim_func_declares_dynamic_abi_and_composes_macros():
    parameters = (
        ("A", T.Tensor((128,), T.float32)),
        ("scratch", T.Tensor((128,), T.float32)),
        ("B", T.Tensor((128,), T.float32)),
    )

    def body(A, scratch, B):
        with T.Kernel(1, threads=128):
            _copy(A, scratch, 128)
        with T.Kernel(1, threads=128):
            _copy(scratch, B, 128)

    function = T.build_prim_func("copy", parameters, body)

    assert function.attrs["global_symbol"] == "copy"
    assert len(function.params) == 3
    assert [function.buffer_map[param].name for param in function.params] == [
        "A",
        "scratch",
        "B",
    ]


def test_build_prim_func_rejects_duplicate_parameter_names():
    with pytest.raises(ValueError, match="unique"):
        T.build_prim_func(
            "duplicate",
            (("A", T.Tensor((1,), T.float32)), ("A", T.Tensor((1,), T.float32))),
            lambda *_: None,
        )


def test_build_prim_module_reuses_one_private_schedule_at_multiple_call_sites():
    parameters = (
        ("A", T.Tensor((128,), T.float32)),
        ("scratch", T.Tensor((128,), T.float32)),
        ("B", T.Tensor((128,), T.float32)),
    )

    def schedule_body(source, target):
        with T.Kernel(1, threads=128):
            _copy(source, target, 128)

    schedule = T.PrimFuncDefinition(
        "copy_schedule",
        (("source", T.Tensor((128,), T.float32)), ("target", T.Tensor((128,), T.float32))),
        schedule_body,
    )

    def body(private, A, scratch, B):
        private["copy_schedule"](A, scratch)
        private["copy_schedule"](scratch, B)

    module = T.build_prim_module("main", parameters, body, (schedule,))

    assert len(module.functions) == 2
    assert module["main"].attrs["global_symbol"] == "main"
    assert module["copy_schedule"].attrs.get("global_symbol") is None
    assert module["copy_schedule"].attrs["tl.is_host_launcher"] == 1
    calls = []
    private_global = module.get_global_var("copy_schedule")
    tvm.tirx.stmt_functor.post_order_visit(
        module["main"].body,
        lambda node: calls.append(node) if isinstance(node, tvm.tirx.Call) and node.op.same_as(private_global) else None,
    )
    assert len(calls) == 2
