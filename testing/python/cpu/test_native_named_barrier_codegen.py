"""Named barriers reach the backend emitter through native Evaluate dispatch."""

import pytest
import tvm_ffi
import tvm_tirx as tvm

I = tvm.ir


def _module(scope, count):
    args = [I.StringImm(scope), I.const(3, "int32")]
    if count is not None:
        args.append(I.const(count, "int32"))
    call = I.Call("tl.storage_sync", args, ty=I.PrimType("void"))
    function = tvm.tirx.Function([], [I.Evaluate(call)], ret_type=I.PrimType("void"))
    function = function.with_attr("global_symbol", "barrier_kernel").with_attr("calling_conv", 2)
    return tvm.IRModule({"barrier_kernel": function})


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("count", [None, 64])
def test_cuda_named_barriers_are_emitted(scope, count, tmp_path):
    build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    source = build(_module(scope, count), target).inspect_source("cuda")
    (tmp_path / "kernel.cu").write_text(source)
    expected = "tl::__sync_thread_partial(3" + ("" if count is None else ", 64") + ");"
    assert source.count(expected) == 1
    assert "__syncthreads();" not in source


@pytest.mark.parametrize("backend,target", [("hip", "rocm"), ("metal", "metal")])
def test_named_barriers_are_not_silently_truncated(backend, target):
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile")
    with pytest.raises(ValueError, match="does not support named/partial barriers"):
        build(_module("shared", 64), tvm.target.Target(target))


def test_cpu_named_barriers_are_not_silently_omitted():
    build = tvm_ffi.get_global_func("target.build.tilelang_c")
    with pytest.raises(ValueError, match="does not support named/partial barriers"):
        build(_module("shared", 64), tvm.target.Target("c"))
