import pytest

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
