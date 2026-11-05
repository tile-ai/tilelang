"""The public Kernel facade must construct this native region contract."""

import pytest

import tilelang
import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.script.ir_builder import stmt as B
from tvm_tirx.script.ir_builder import IRBuilder


def launch_input(*, explicit_threads=False):
    n = ir.Var("n", "int64")
    args = [
        ir.Tuple([ir.IntImm("int64", 0)]),
        ir.Tuple([n]),
        ir.Tuple([ir.IntImm("int64", 1)]),
        ir.Tuple([ir.IntImm("int32", v) for v in (32, 1, 1)] if explicit_threads else []),
        ir.Tuple([]),
    ]
    params = [ir.Var("bx", "int64"), *(ir.Var(name, "int32") for name in ("tx", "ty", "tz"))]
    attrs = {
        "schema_version": 1,
        "threads_was_specified": explicit_threads,
        "grid_tags": ["blockIdx.x"],
        "thread_tags": ["threadIdx.x", "threadIdx.y", "threadIdx.z"],
    }
    return args, params, attrs


@pytest.mark.parametrize("explicit_threads", [False, True])
def test_native_region_keeps_parameters_and_dynamic_grid(explicit_threads):
    args, params, attrs = launch_input(explicit_threads=explicit_threads)
    body = ir.Evaluate(params[0])
    region = ir.RegionStmt("tl.launch_kernel", args, params, attrs, [body])
    assert isinstance(region.body, ir.SeqStmt)
    assert region.args[1].fields[0].same_as(args[1].fields[0])
    assert all(actual.same_as(expected) for actual, expected in zip(region.body_params, params))
    assert region.body[0].value.same_as(params[0])
    assert not region.result_vars


def test_builder_creates_fresh_native_parameters():
    args, _, attrs = launch_input()
    with IRBuilder():
        first = B.region("tl.launch_kernel", args, attrs=attrs)
        second = B.region("tl.launch_kernel", args, attrs=attrs)
        assert len(first.body_params) == 4
        assert [str(v.ty) for v in first.body_params] == ["int64", "int32", "int32", "int32"]
        assert not any(a.same_as(b) for a, b in zip(first.body_params, second.body_params))


@pytest.mark.parametrize("case", ["schema", "arity", "zero_step", "threads", "type", "duplicate", "results"])
def test_invalid_launch_is_rejected_before_lowering(case):
    args, params, attrs = launch_input()
    results = []
    if case == "schema":
        attrs["schema_version"] = 2
    elif case == "arity":
        args.pop()
    elif case == "zero_step":
        args[2] = ir.Tuple([ir.IntImm("int64", 0)])
    elif case == "threads":
        attrs["threads_was_specified"] = True
    elif case == "type":
        params[0] = ir.Var("bx", "int32")
    elif case == "duplicate":
        params[2] = params[1]
    elif case == "results":
        results = [ir.Var("escape", "int32")]
    with pytest.raises((ValueError, TypeError)):
        ir.RegionStmt("tl.launch_kernel", args, params, attrs, [], result_vars=results)


def test_provider_uses_native_namespace_and_one_registry():
    from importlib.metadata import version
    import sys
    import tvm_tirx
    from tilelang._provider import FFI_REF, FFI_VERSION

    assert tilelang._LIB is tvm_tirx.base._LOADED_LIBS["tvm_tirx_compiler"]
    assert not hasattr(tilelang, "tvm")
    assert not any(name == "tilelang.tvm" or name.startswith("tilelang.tvm.") for name in sys.modules)
    assert "tvm" not in sys.modules
    assert dict(tvm_ffi.get_global_func("tl.CompilerBuildInfo")())["ffi_ref"] == FFI_REF
    assert version("apache-tvm-ffi") == FFI_VERSION
    assert tvm_ffi.get_global_func("tl.SideEffect")(ir.IntImm("int32", 3)) == 1
