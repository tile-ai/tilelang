"""CuTe hints retain same-scope control flow and local Bind uses."""

import tilelang.language as T
import ast
import os
import site

import pytest
import tilelang
import torch
import tvm_tirx


def make_kernel(control, style, hint_key):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if style == "plain":
                if control == "return":
                    local_value = A[tx] + 1
                    B[tx] = local_value
                    if tx % 2 == 0:
                        T.thread_return()
                    B[tx] = local_value + 10
                else:
                    B[tx] = A[tx]
                    for k in T.serial(4):
                        B[tx] = B[tx] + 1
                        if k == 2:
                            T.loop_break()
                        B[tx] = B[tx] + 10
            elif style == "attr":
                with T.attr(0, hint_key, 1):
                    if control == "return":
                        local_value = A[tx] + 1
                        B[tx] = local_value
                        if tx % 2 == 0:
                            T.thread_return()
                        B[tx] = local_value + 10
                    else:
                        B[tx] = A[tx]
                        for k in T.serial(4):
                            B[tx] = B[tx] + 1
                            if k == 2:
                                T.loop_break()
                            B[tx] = B[tx] + 10
            else:
                with T.sblock("hint"):
                    T.sblock_attr({hint_key: 1})
                    if control == "return":
                        local_value = A[tx] + 1
                        B[tx] = local_value
                        if tx % 2 == 0:
                            T.thread_return()
                        B[tx] = local_value + 10
                    else:
                        B[tx] = A[tx]
                        for k in T.serial(4):
                            B[tx] = B[tx] + 1
                            if k == 2:
                                T.loop_break()
                            B[tx] = B[tx] + 10

    return kernel


@pytest.mark.parametrize("control", ["return", "break"])
@pytest.mark.parametrize(
    "style,hint_key",
    [
        ("plain", "none"),
        ("attr", "user_hint_152"),
        ("attr", "pragma_unknown_152"),
        ("attr", "code_block_unknown_152"),
        ("sblock", "user_hint_152"),
    ],
)
def test_cutedsl_hint_scope_keeps_control_flow(control, style, hint_key):
    extra = os.environ.get("TILELANG_TEST_CUTEDSL_SITE")
    if extra:
        site.addsitedir(extra)
    pytest.importorskip("cutlass.cute")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for CuTe control-flow execution")
    major, minor = torch.cuda.get_device_capability()
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": f"sm_{major}{minor}", "keys": ["cuda", "cutedsl"]})
    function = make_kernel(control, style, hint_key)
    assert tvm_tirx.tirx.analysis.check_well_formed(function)
    executable = tilelang.compile(function, target=target, execution_backend="cutedsl")
    source = executable.get_kernel_source()
    if control == "break":
        assert not [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Break)]
        assert "_loop_break_" in source
    data = torch.arange(32, dtype=torch.int32, device="cuda") - 11
    output = torch.full((32,), -123, dtype=torch.int32, device="cuda")
    executable(data, output)
    torch.cuda.synchronize()
    expected = [value + (1 if index % 2 == 0 else 11) if control == "return" else value + 23 for index, value in enumerate(range(-11, 21))]
    assert output.cpu().tolist() == expected


def make_nested_break(hint_key):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[tx] = A[tx]
            for outer in T.serial(2):  # noqa: B007 - repeat the inner loop with a fresh break guard
                for k in T.serial(4):
                    with T.attr(0, hint_key, 1):
                        local_value = B[tx] + 1
                        B[tx] = local_value
                        if k == 2:
                            T.loop_break()
                    B[tx] = B[tx] + 10

    return kernel


@pytest.mark.parametrize("hint_key", ["user_hint_152", "pragma_unknown_152", "code_block_unknown_152"])
def test_nested_break_guards_the_enclosing_sequence_suffix(hint_key):
    extra = os.environ.get("TILELANG_TEST_CUTEDSL_SITE")
    if extra:
        site.addsitedir(extra)
    pytest.importorskip("cutlass.cute")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for CuTe control-flow execution")
    major, minor = torch.cuda.get_device_capability()
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": f"sm_{major}{minor}", "keys": ["cuda", "cutedsl"]})
    function = make_nested_break(hint_key)
    assert tvm_tirx.tirx.analysis.check_well_formed(function)
    executable = tilelang.compile(function, target=target, execution_backend="cutedsl")
    source = executable.get_kernel_source()
    assert not [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Break)]
    data = torch.arange(32, dtype=torch.int32, device="cuda") - 11
    output = torch.full((32,), -123, dtype=torch.int32, device="cuda")
    executable(data, output)
    torch.cuda.synchronize()
    assert output.cpu().tolist() == list(range(35, 67))
