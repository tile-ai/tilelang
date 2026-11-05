"""Direct Pass use initializes the same callbacks as the full backend pipeline."""

import subprocess
import sys
import textwrap

import pytest


def _run(source):
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("key", ["tl.gemm.lower", "tl.gemm.infer_layout", "tl.gemm_sp.lower", "tl.gemm_sp.infer_layout"])
@pytest.mark.parametrize(
    "module,name",
    [
        ("tilelang.transform", "LayoutInference"),
        ("tilelang.transform", "LowerTileOp"),
        ("tilelang.ascend.transform", "AscendLayoutInference"),
        ("tilelang.ascend.transform", "AscendLowerTileOp"),
    ],
)
def test_callback_override_before_first_pass_is_preserved(key, module, name):
    _run(f"""
        import importlib
        import tilelang
        import tvm_ffi
        assert tvm_ffi.get_global_func({key!r}, allow_missing=True) is None
        tvm_ffi.register_global_func({key!r}, lambda *args: "early user override", override=True)
        factory = getattr(importlib.import_module({module!r}), {name!r})
        for _ in range(2):
            factory()
            assert tvm_ffi.get_global_func({key!r})() == "early user override"
        for default in ["tl.gemm.lower", "tl.gemm.infer_layout", "tl.gemm_sp.lower", "tl.gemm_sp.infer_layout"]:
            assert tvm_ffi.get_global_func(default, allow_missing=True) is not None
        # The global override must not replace the builtin Python methods that
        # a user callback may explicitly invoke as its fallback.
        from tilelang.tileop.gemm import gemm_lower
        from tilelang.tileop.gemm_sp import GemmSP
        assert gemm_lower.__name__ == "gemm_lower"
        assert GemmSP.gemm_sp_lower.__name__ == "gemm_sp_lower"
        tvm_ffi.register_global_func({key!r}, lambda *args: "later user override", override=True)
        factory()
        assert tvm_ffi.get_global_func({key!r})() == "later user override"
    """)


@pytest.mark.parametrize("first", ["tilelang", "tvm_tirx"])
@pytest.mark.parametrize(
    "module,name",
    [
        ("tilelang.transform", "LayoutInference"),
        ("tilelang.transform", "LowerTileOp"),
        ("tilelang.ascend.transform", "AscendLayoutInference"),
        ("tilelang.ascend.transform", "AscendLowerTileOp"),
    ],
)
def test_direct_pass_registration_is_lazy_idempotent_and_preserves_overrides(first, module, name):
    _run(f"""
        import importlib
        import sys
        importlib.import_module({first!r})
        import tvm_ffi
        module = importlib.import_module({module!r})
        factory = getattr(module, {name!r})
        assert tvm_ffi.get_global_func("tl.gemm.lower", allow_missing=True) is None
        raw = getattr(module._ffi_api, {name!r})()
        result = factory()
        assert type(result) is type(raw)
        assert result.info.name == raw.info.name
        assert result.info.opt_level == raw.info.opt_level
        for key in ["tl.gemm.lower", "tl.gemm.infer_layout", "tl.gemm_sp.lower", "tl.gemm_sp.infer_layout"]:
            assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, key
        from tilelang.tileop.gemm.registry import _GEMM_IMPLS
        before = [entry.name for entry in _GEMM_IMPLS]
        assert len(before) == len(set(before))
        assert "ascend.mad" in before
        if {module!r} == "tilelang.transform":
            assert {{"cpu.scalar", "cuda.mma", "rocm.mfma"}}.issubset(before)
        else:
            assert "tilelang.cuda.op" not in sys.modules
        assert "tilelang.cuda.op.gemm.gemm_mma" not in sys.modules
        tvm_ffi.register_global_func("tl.gemm.lower", lambda: "user override", override=True)
        factory()
        factory()
        assert tvm_ffi.get_global_func("tl.gemm.lower")() == "user override"
        assert [entry.name for entry in _GEMM_IMPLS] == before
        assert "tvm" not in sys.modules
    """)


@pytest.mark.parametrize("first", ["tilelang", "tvm_tirx"])
@pytest.mark.parametrize("backend", ["llvm", "cuda"])
def test_generic_pass_lowers_gemm_without_backend_import(first, backend):
    _run(f"""
        import importlib
        importlib.import_module({first!r})
        import numpy as np
        import tvm_ffi
        import tvm_tirx
        from tvm_tirx import ir, tirx
        import tilelang.language as T
        from tilelang import transform
        backend = {backend!r}
        target = tvm_tirx.target.Target(
            {{"kind": "llvm", "mcpu": "generic"}} if backend == "llvm"
            else {{"kind": "cuda", "arch": "sm_70"}}
        )
        a, b, c = [ir.Var(name, tirx.TensorType(shape, "float32"))
                   for name, shape in [("a", (3, 4)), ("b", (4, 5)), ("c", (3, 5))]]
        call = T.gemm(a, b, c, clear_accum=True)
        body = ir.SeqStmt([ir.Evaluate(call)])
        if backend == "cuda":
            thread = ir.Var("thread", "int32")
            body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)],
                                 [thread], None, body)
        function = tirx.Function([a, b, c], body, ir.TupleType([])).with_attr(
            {{"global_symbol": "main", "target": target}}
        )
        assert tvm_ffi.get_global_func("tl.gemm.lower", allow_missing=True) is None
        with target:
            inferred = transform.LayoutInference()(ir.IRModule({{"main": function}}))
            lowered = transform.LowerTileOp()(inferred)
        assert tirx.analysis.check_well_formed(lowered["main"])
        calls = []
        tvm_ffi.structural_walk(lowered, (ir.Call, lambda node: calls.append(node.op.name)
                                 if isinstance(node.op, ir.Op) else None))
        assert not any(name.startswith("tl.tileop.") for name in calls)
        if backend == "llvm":
            lowered = transform.FinalizeAnnotations("llvm")(lowered)
            executable = tirx.compile(lowered, target=target)
            av = np.arange(12, dtype="float32").reshape(3, 4) / 8
            bv = np.arange(20, dtype="float32").reshape(4, 5) / 4
            cv = np.full((3, 5), -99, dtype="float32")
            tensors = [tvm_tirx.runtime.tensor(value) for value in (av, bv, cv)]
            executable(*tensors)
            np.testing.assert_array_equal(tensors[-1].numpy(), av @ bv)
    """)
