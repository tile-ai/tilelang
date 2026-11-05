"""Lexical allocation scopes survive storage planning and real CPU codegen."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.cpu.language as T
from tilelang.language import _native
from tilelang.testing.packed import compile_packed_function


def scope_function():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with _native.attr(None, "lexical_alloc_scope", 1):
            scratch = T.alloc_local((8,), "float32")
            for i in T.serial(8):
                scratch[i] = A[i] + 1.0
            for i in T.serial(8):
                B[i] = scratch[i]
        with _native.attr(None, "lexical_alloc_scope", 1):
            scratch = T.alloc_local((8,), "float32")
            for i in T.serial(8):
                scratch[i] = B[i] * 2.0
            for i in T.serial(8):
                B[i] = scratch[i]

    module = tvm_tirx.IRModule({"main": main})
    return tvm_ffi.get_global_func("tl.transform.LowerFrontendScopes")()(module)


def test_native_scope_preserves_storage_and_executes_once():
    module = scope_function()
    scope = ir.Op.get("tl.lexical_scope")
    regions = []
    tvm_ffi.structural_walk(module, (ir.RegionStmt, regions.append), order="pre")
    assert sum(region.op.same_as(scope) for region in regions) == 2
    planned = tilelang.transform.StorageRewrite()(module)
    assert tirx.analysis.check_well_formed(planned)
    rewritten_regions = []
    tvm_ffi.structural_walk(planned, (ir.RegionStmt, rewritten_regions.append), order="pre")
    assert sum(region.op.same_as(scope) for region in rewritten_regions) == 2
    # TileLang owns this region and its allocation policy. Native tirx.compile
    # intentionally does not consume the private TileLang dialect.
    compiled = compile_packed_function(planned)
    source = tvm_tirx.runtime.tensor(np.arange(8, dtype="float32"))
    result = tvm_tirx.runtime.tensor(np.empty(8, dtype="float32"))
    compiled(source, result)
    np.testing.assert_array_equal(result.numpy(), (source.numpy() + 1) * 2)


def test_scope_rejects_escaping_definition():
    module = scope_function()
    function = module["main"]
    region = function.body.seq[0]
    scratch = region.body.seq[0].var
    invalid = function.with_body([region, ir.Evaluate(scratch)])
    assert not tirx.analysis.check_well_formed(invalid, assert_mode=False)
    with pytest.raises(RuntimeError, match="Invalid use of undefined variable scratch"):
        tirx.analysis.check_well_formed(invalid)
