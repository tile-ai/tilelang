import importlib.util
import json
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, PreLowerSemanticCheck
from tilelang.utils.target import determine_target
from tvm import tir


tilelang.env.disable_cache()


def _load_elementwise_add_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_add.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_elementwise_add_example", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sunmmio_target():
    return tvm.target.Target(determine_target("sunmmio", return_object=True), tvm.target.Target.canon_target("llvm"))


def _is_sunmmio_device_func(func: tir.PrimFunc) -> bool:
    attrs = func.attrs
    return bool(attrs and attrs.get("tir.is_global_func", False))


def _single_prim_func(mod: tvm.IRModule):
    funcs = [(gvar, func) for gvar, func in mod.functions.items() if isinstance(func, tir.PrimFunc)]
    assert len(funcs) == 1
    return funcs[0]


def _var_name(var: tir.Var) -> str:
    return getattr(var, "name", getattr(var, "name_hint", str(var)))


def _dynamic_elementwise_add_prim_func():
    example = _load_elementwise_add_example()
    return example._elementwise_add_prim_func(
        T.dynamic("m"),
        T.dynamic("n"),
        32,
        32,
        T.bfloat16,
        T.float32,
    )


def _optimize_for_sunmmio_direct_runtime(mod: tvm.IRModule) -> tvm.IRModule:
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
    mod = tilelang.transform.InjectSunmmioPipeline()(mod)

    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.InjectSunmmioSync()(mod)

    # Direct Sunmmio runtimes consume the device PrimFunc ABI directly. MakePackedAPI
    # is a host-wrapper transform and is intentionally absent from this boundary.
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


def _lower_dynamic_elementwise_add_for_direct_runtime() -> tvm.IRModule:
    target = _sunmmio_target()
    kernel = _dynamic_elementwise_add_prim_func()
    mod = tvm.IRModule({kernel.attrs["global_symbol"]: kernel})

    with tvm.transform.PassContext(opt_level=3), target:
        PreLowerSemanticCheck(mod)
        mod = LowerAndLegalize(mod, target)
        mod = _optimize_for_sunmmio_direct_runtime(mod)

    return mod


def _device_mod_for_direct_runtime() -> tvm.IRModule:
    mod = _lower_dynamic_elementwise_add_for_direct_runtime()
    return tir.transform.Filter(_is_sunmmio_device_func)(mod)


def test_direct_runtime_boundary_keeps_dynamic_device_abi_without_make_packed_api():
    device_mod = _device_mod_for_direct_runtime()
    gvar, func = _single_prim_func(device_mod)

    assert gvar.name_hint == "elem_add_kernel"
    assert str(func.attrs["global_symbol"]) == "elem_add_kernel"
    assert bool(func.attrs["tir.is_global_func"])
    assert "layout_map" in func.attrs
    assert "global_layout_map" in func.attrs

    assert [_var_name(param) for param in func.params] == ["A", "B", "C", "m", "n"]
    assert [str(param.dtype) for param in func.params[-2:]] == ["int32", "int32"]

    script = func.script()
    assert "m: T.int32" in script
    assert "n: T.int32" in script
    assert "__tvm_ffi" not in script
    assert "self_handle" not in script


def test_direct_runtime_codegen_accepts_dynamic_mesh_symbols_without_make_packed_api(tmp_path, monkeypatch):
    target = _sunmmio_target()
    device_mod = _device_mod_for_direct_runtime()
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile", allow_missing=True)
    if builder is None:
        pytest.skip("Sunmmio SUVM codegen is not available in this build.")

    coverage_path = tmp_path / "dynamic_elementwise_coverage.json"
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", str(coverage_path))
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    runtime_mod = builder(device_mod, target, "suvm")
    src = runtime_mod.inspect_source()

    missing = [
        token
        for token in (
            "func.func @elem_add_kernel",
            "%arg3: i32",
            "%arg4: i32",
            "suvm.bind_layout",
            "dynamic_shapes =",
            "dynamic_strides =",
        )
        if token not in src
    ]
    assert not missing, f"missing expected SUVM MLIR tokens: {missing}\n{src}"

    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    tiles = coverage["tiles"]
    assert tiles["missing_node_types"] == []
    assert tiles["missing_call_ops"] == []
    for node_type in ("tir.BufferLoad", "tir.Add"):
        assert node_type in tiles["expected_node_types"]
        assert node_type in tiles["visited_node_types"]
