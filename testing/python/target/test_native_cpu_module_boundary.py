"""CPU host/compute artifacts preserve their explicit native call boundary."""

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.cpu.language as C


def test_reused_target_compiles_cpu_kernel_three_times():
    target = Target({"kind": "llvm", "mcpu": "generic"})
    initial = ir.save_json(target)
    for repeat in range(3):

        @C.prim_func
        def kernel(output: C.Tensor((1,), "int32"), n: C.int32):
            with C.Kernel(1):
                for i in C.serial(4, 5):
                    C.assume(n >= i + 3)
                    output[0] = n + i

        before = ir.save_json(kernel)
        executable = tilelang.compile(kernel, target=target, target_host=target, execution_backend="tvm_ffi")
        assert before == ir.save_json(kernel)
        assert initial == ir.save_json(target)
        result = torch.full((1,), -99, dtype=torch.int32)
        executable(result, 7)
        assert result.item() == 11, repeat


def _cpu_kernel():
    @C.prim_func
    def kernel(output: C.Tensor((1,), "int32"), n: C.int32):
        with C.Kernel(1):
            for i in C.serial(4, 5):
                C.assume(n >= i + 3)
                output[0] = n + i

    return kernel


def _calls(function):
    calls = []
    tvm_ffi.structural_walk(function.body, (ir.Call, calls.append))
    return calls


def _native_module(recursive=False, wrong_identity=False, kind="llvm"):
    target = Target({"kind": "llvm", "mcpu": "generic"} if kind == "llvm" else {"kind": kind})
    host_target = target.with_host(target)
    entry_gv, compute_gv, helper_gv = map(ir.GlobalVar, ["entry", "compute", "helper"])
    # Equal spellings in different function scopes must remain distinct Vars.
    arg, compute_arg, helper_arg = [ir.Var("value", "int32") for _ in range(3)]
    result = ir.Var("result", "int32")
    helper_call = ir.Call(helper_gv, [arg], ty="int32")
    call_target = ir.GlobalVar("compute") if wrong_identity else compute_gv
    compute_call = ir.Call(
        call_target,
        [helper_call],
        ty="int32",
        attrs=ir.DictAttrs({"test_tag": 28}),
        loc=ir.SourceLoc(ir.SourceName("cpu_boundary.py"), 7, 3, 7, 19),
    )
    entry = tirx.Function(
        [arg],
        [ir.Bind(result, compute_call), ir.Return(result)],
        ir.PrimType("int32"),
        ir.DictAttrs({"target": host_target, "global_symbol": "entry"}),
    )
    compute_body = ir.Call(compute_gv, [compute_arg], ty="int32") if recursive else compute_arg + 1
    compute = tirx.Function(
        [compute_arg],
        [ir.Return(compute_body)],
        ir.PrimType("int32"),
        ir.DictAttrs({"target": target, "tirx.is_global_func": True, "global_symbol": "custom_compute_export", "keep_metadata": 27}),
    )
    helper = tirx.Function(
        [helper_arg],
        [ir.Return(helper_arg * 2)],
        ir.PrimType("int32"),
        ir.DictAttrs({"target": host_target, "global_symbol": "helper"}),
    )
    return ir.IRModule({entry_gv: entry, compute_gv: compute, helper_gv: helper}), entry_gv, compute_gv, helper_gv


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("kind", ["llvm", "c"])
def test_only_cross_artifact_cpu_compute_call_becomes_external(recursive, kind):
    module, entry_gv, compute_gv, helper_gv = _native_module(recursive, kind=kind)
    before = ir.save_json(module)
    transformed = tilelang.transform.SplitHostDevice()(module)
    entry_calls = _calls(transformed[entry_gv])
    external = [call for call in entry_calls if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_extern"]
    assert len(external) == 1
    assert external[0].args[0].value == "custom_compute_export"
    assert str(external[0].ty) == "int32"
    assert len(external[0].args) == 2
    assert external[0].attrs["test_tag"] == 28
    assert external[0].loc.start_line == 7
    assert external[0].loc.start_column == 3
    assert external[0].loc.end_line == 7
    assert external[0].loc.end_column == 19
    helper_calls = [call for call in entry_calls if isinstance(call.op, ir.GlobalVar)]
    assert len(helper_calls) == 1
    assert helper_calls[0].op.same_as(helper_gv)
    assert helper_calls[0].args[0].same_as(transformed[entry_gv].params[0])
    assert transformed[compute_gv].attrs["global_symbol"] == "custom_compute_export"
    assert transformed[compute_gv].attrs["keep_metadata"] == 27
    compute_calls = [call for call in _calls(transformed[compute_gv]) if isinstance(call.op, ir.GlobalVar)]
    assert len(compute_calls) == int(recursive)
    if recursive:
        assert compute_calls[0].op.same_as(compute_gv)
        assert compute_calls[0].args[0].same_as(transformed[compute_gv].params[0])
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(transformed)


def test_same_named_unbound_global_var_remains_an_error():
    module, _, _, _ = _native_module(wrong_identity=True)
    with pytest.raises(RuntimeError, match="did not appear within the IRModule"):
        tilelang.transform.SplitHostDevice()(module)


def test_repeated_cpu_packed_wrapper_preserves_failed_assumption_and_shape():
    target = Target({"kind": "llvm", "mcpu": "generic"})
    kernel = tilelang.compile(_cpu_kernel(), target=target, target_host=target, execution_backend="tvm_ffi")
    result = torch.full((1,), -99, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        kernel(result, 6)
    assert result.item() == -99
    wrong_shape = torch.full((2,), -99, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        kernel(wrong_shape, 7)
    assert wrong_shape.tolist() == [-99, -99]
    kernel(result, 8)
    assert result.item() == 12


def test_shared_target_compiles_independent_cpu_entries_in_threads():
    target = Target({"kind": "llvm", "mcpu": "generic"})
    original = ir.save_json(target)
    source = _cpu_kernel()
    functions = [source.with_attr("global_symbol", f"threaded_cpu_{index}") for index in range(3)]

    def run(index):
        compiled = tilelang.compile(functions[index], target=target, target_host=target, execution_backend="tvm_ffi")
        result = torch.full((1,), -99, dtype=torch.int32)
        compiled(result, 7 + index)
        assert result.item() == 11 + index
        return result.item()

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(run, range(3))) == [11, 12, 13]
    assert ir.save_json(target) == original


def test_cpu_export_reload_in_fresh_process(tmp_path):
    target = Target({"kind": "llvm", "mcpu": "generic"})
    compiled = tilelang.compile(_cpu_kernel(), target=target, target_host=target, execution_backend="tvm_ffi")
    library = tmp_path / "cpu_kernel.so"
    compiled.export_library(str(library))
    script = tmp_path / "reload.py"
    script.write_text(
        "import sys\nimport torch\nimport tvm_tirx\n"
        "from tvm_tirx import runtime\n"
        "module = runtime.load_module(sys.argv[1])\n"
        "output = torch.full((1,), -99, dtype=torch.int32)\n"
        "module['kernel'](output, 7)\n"
        "assert output.tolist() == [11]\n"
        "assert 'tvm' not in sys.modules\n"
        "print('fresh_process_output=11')\n"
    )
    process = subprocess.run([sys.executable, "-B", str(script), str(library)], env=dict(os.environ), text=True, capture_output=True)
    (tmp_path / "reload.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "fresh_process_output=11" in process.stdout


@pytest.mark.parametrize("device_kind", ["c", "llvm"])
@pytest.mark.parametrize("host_kind", ["c", "llvm"])
def test_cpu_host_and_compute_target_combinations_execute(device_kind, host_kind):
    target = Target({"kind": "llvm", "mcpu": "generic"} if device_kind == "llvm" else {"kind": device_kind})
    host = (
        target if device_kind == host_kind else Target({"kind": "llvm", "mcpu": "generic"} if host_kind == "llvm" else {"kind": host_kind})
    )
    compiled = tilelang.compile(_cpu_kernel(), target=target, target_host=host, execution_backend="tvm_ffi")
    result = torch.full((1,), -99, dtype=torch.int32)
    compiled(result, 7)
    assert result.item() == 11
