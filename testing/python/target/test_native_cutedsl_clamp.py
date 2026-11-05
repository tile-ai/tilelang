"""CuTe clamp preserves NaN priority and evaluates each argument once."""

import ast
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx, target

import tilelang
import tilelang.language as T


def _emitted_scalar(dtype):
    output = ir.Var("out", ir.PtrType(dtype))
    view = ir.Var("output", tirx.TensorType((1,), dtype, layout="default"))
    arguments = [tirx.call_extern("clamp_value_" + name, ty=dtype) for name in ("x", "lo", "hi")]
    function = tirx.Function(
        [output],
        [
            ir.Bind(view, ir.Call("tirx.decl_tensor", [output], ty_args=[view.ty])),
            ir.TensorStore(view, [0], ir.Call("tl.clamp", arguments, ty=dtype)),
        ],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "clamp_probe", "calling_conv": 2}),
    )
    runtime = tvm_ffi.get_global_func("target.build.tilelang_cutedsl_without_compile")(
        ir.IRModule({"clamp_probe": function}), target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    return runtime.inspect_source("cutedsl")


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
def test_scalar_clamp_has_single_evaluation_and_nan_priority(dtype, tmp_path):
    source = _emitted_scalar(dtype)
    (tmp_path / "clamp.py").write_text(source)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    for name in ("x", "lo", "hi"):
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "clamp_value_" + name
        ]
        assert len(calls) == 1
    # Execute only the emitted scalar assignments and the exact stored RHS.
    # Tensor addressing, GPU launch and CuTe execution are tested separately.
    bindings = [
        node
        for node in function.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("clamp_")
    ]
    store = next(node for node in function.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript))
    scalar = ast.Module(body=bindings + [ast.Assign(targets=[ast.Name(id="actual", ctx=ast.Store())], value=store.value)], type_ignores=[])
    program = compile(ast.fix_missing_locations(scalar), "emitted_clamp_scalar.py", "exec")
    values = [(-2, -1, 1), (0, -1, 1), (2, -1, 1), (3, 5, 1)]
    scalar_type = np.dtype(dtype)
    if dtype.startswith("float"):
        bits = "uint32" if dtype == "float32" else "uint64"
        nan_base = 0x7FC00000 if dtype == "float32" else 0x7FF8000000000000
        nan_x, nan_lo, nan_hi = np.array([nan_base + 1, nan_base + 2, nan_base + 3], dtype=bits).view(dtype)
        values += [(nan_x, nan_lo, nan_hi), (2, nan_lo, nan_hi), (2, -1, nan_hi)]
    for row in values:
        x, lo, hi = [scalar_type.type(value) for value in row]
        counts = {name: 0 for name in ("x", "lo", "hi")}

        def value(name, number, counters):
            def invoke():
                counters[name] += 1
                return number

            return invoke

        scope = {"tl": SimpleNamespace(min=min, max=max)}
        scope.update({"clamp_value_" + name: value(name, number, counts) for name, number in zip(("x", "lo", "hi"), (x, lo, hi))})
        exec(program, scope)
        assert counts == {"x": 1, "lo": 1, "hi": 1}
        expected = x if np.isnan(x) else lo if np.isnan(lo) else hi if np.isnan(hi) else min(max(x, lo), hi)
        assert np.asarray(scope["actual"], dtype=dtype).tobytes() == np.asarray(expected, dtype=dtype).tobytes()


@pytest.mark.parametrize("launcher", ["cpp", "cutlass"])
def test_cutedsl_clamp_cuda_numeric(launcher, monkeypatch):
    pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CuTe numeric kernel")
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", launcher)
    monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"), Lo: T.Tensor((128,), "float32"), Hi: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                C[i] = T.clamp(A[i], Lo[i], Hi[i])

    kernel = tilelang.compile(
        main,
        target={"kind": "cutedsl", "arch": "sm_100a"},
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="cutedsl",
        out_idx=[3],
        pass_configs={"tl.disable_warp_specialized": True},
    )
    data = torch.tensor([float("nan"), -2, 0, 2, 3, 1, -1, 4], dtype=torch.float32, device="cuda").repeat(16)
    lo, hi = torch.full_like(data, -1), torch.full_like(data, 1)
    lo[5::8] = float("nan")
    hi[6::8] = float("nan")
    lo[7::8] = 5
    torch.testing.assert_close(kernel(data, lo, hi), torch.clamp(data, lo, hi), equal_nan=True, rtol=0, atol=0)
