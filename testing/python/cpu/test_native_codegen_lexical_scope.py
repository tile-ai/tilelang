"""Frontend allocation regions retain their lifetime in every C-like emitter."""

import re

import pytest
import tvm_ffi
import tvm_tirx

I = tvm_tirx.ir


def _scope(body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": "lexical_alloc_scope", "value": 1})
    return I.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


@pytest.mark.parametrize("backend", ["c", "c_host", "cuda", "hip", "metal"])
@pytest.mark.parametrize("in_loop", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_lexical_alloc_scope_is_a_real_block(backend, in_loop, nested, tmp_path):
    # CHost sees global allocations after host workspace lowering; the device
    # and CPU emitters also accept local stack allocations directly.
    scope = "global" if backend == "c_host" else "local"
    scratch = I.Var("lexical_scratch", tvm_tirx.tirx.TensorType((4,), "float32", scope=scope, layout="default"))
    allocate = I.Bind(scratch, I.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty]))
    body = _scope([allocate, I.TensorStore(scratch, [0], I.const(3, "float32"))])
    if nested:
        body = _scope([body])
    if in_loop:
        body = I.For(I.Var("i", "int32"), 0, 3, 1, I.ForKind.DEFAULT, [body])
    after = I.Bind(I.Var("after_scope", "int32"), I.const(9, "int32"))
    func = tvm_tirx.tirx.Function([], [body, after], ret_type=I.PrimType("void")).with_attr("global_symbol", "kernel")
    if backend in ("cuda", "hip", "metal"):
        func = func.with_attr("calling_conv", 2)
    suffix = "_without_compile" if backend in ("cuda", "hip", "metal") else ""
    target = "rocm" if backend == "hip" else "c" if backend == "c_host" else backend
    module = tvm_ffi.get_global_func("target.build.tilelang_" + backend + suffix)(
        tvm_tirx.IRModule({"kernel": func}), tvm_tirx.target.Target(target)
    )
    code = module.inspect_source()
    (tmp_path / "kernel.txt").write_text(code)
    name = "lexical_scratch_ptr" if backend in ("c", "c_host") else "lexical_scratch"
    allocation = re.search(rf"\bfloat {name}\[4\]", code)
    continuation = re.search(r"\bafter_scope\s*=", code)
    assert allocation and continuation, code

    def depth(position):
        prefix = code[:position]
        return prefix.count("{") - prefix.count("}")

    # CUDA merges lexical scopes at the tail of a loop: their allocation
    # lifetime ends at the same closing brace. Other emitters keep the braces.
    expected_depth = 1 if backend == "cuda" and in_loop else 1 + int(in_loop) + int(nested)
    assert depth(allocation.start()) - depth(continuation.start()) == expected_depth
    if in_loop:
        assert re.search(r"\bfor\s*\(", code), code
    assert code.index(name + "[0]") < continuation.start()


@pytest.mark.parametrize("nested", [False, True])
def test_cuda_lexical_scope_with_following_loop_sibling_is_not_merged(nested, tmp_path):
    scratch = I.Var("lexical_scratch", tvm_tirx.tirx.TensorType((4,), "float32", scope="local", layout="default"))
    allocate = I.Bind(scratch, I.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty]))
    body = _scope([allocate, I.TensorStore(scratch, [0], I.const(3, "float32"))])
    if nested:
        body = _scope([body])
    following = I.Bind(I.Var("after_scope_in_loop", "int32"), I.const(7, "int32"))
    loop = I.For(I.Var("i", "int32"), 0, 3, 1, I.ForKind.DEFAULT, [body, following])
    after = I.Bind(I.Var("after_loop", "int32"), I.const(9, "int32"))
    func = (
        tvm_tirx.tirx.Function([], [loop, after], ret_type=I.PrimType("void"))
        .with_attr("global_symbol", "kernel")
        .with_attr("calling_conv", 2)
    )
    saved = I.save_json(func)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        tvm_tirx.IRModule({"kernel": func}), tvm_tirx.target.Target("cuda")
    )
    assert I.save_json(func) == saved
    code = module.inspect_source()
    (tmp_path / "kernel.txt").write_text(code)
    allocation = re.search(r"\bfloat lexical_scratch\[4\]", code)
    continuation = re.search(r"\bafter_scope_in_loop\s*=", code)
    after_loop = re.search(r"\bafter_loop\s*=", code)
    assert allocation and continuation and after_loop, code

    def depth(position):
        prefix = code[:position]
        return prefix.count("{") - prefix.count("}")

    assert depth(allocation.start()) - depth(continuation.start()) == 1 + int(nested)
    assert depth(continuation.start()) - depth(after_loop.start()) == 1
    assert code.index("lexical_scratch[0]") < continuation.start() < after_loop.start()
