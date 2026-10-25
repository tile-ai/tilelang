from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def kernel(producers=128, total=384, hints=(), custom=False, simt=False, nested=False):
    output = TensorView.create("output", [total], "int32")
    local = TensorView.create("local", [1], "int32", scope="local")
    thread = ir.Var("thread", "int32")
    producer = [output.store(thread, 1)]
    if simt:
        producer.insert(0, local.store(0, 1))
    branch = ir.If(thread < producers, producer, output.store(thread, 2))
    if nested:
        branch = ir.RegionStmt("tirx.scope", [], [], None, [ir.Evaluate(0), branch])
    marker = "kCustomWarpSpecialization" if custom else "kWarpSpecializationScope"
    body = ir.RegionStmt("tirx.scope", [], [], {marker: True}, branch)
    body = [*[ir.Evaluate(ir.Call(f"tl.{name}", list(arguments))) for name, arguments in hints], body]
    if simt:
        body.insert(0, local.allocate())
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(total)], [thread], None, body)
    function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([]))
    return ir.IRModule({"main": function.with_attr("global_symbol", "register_test")})


def lower(module):
    result = tvm_ffi.get_global_func("tl.cuda.transform.AnnotateWarpGroupRegAlloc")()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def calls(module, name="tl.set_max_nreg"):
    result = []
    stmt_functor.post_order_visit(
        module,
        lambda node: result.append(node) if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == name else None,
    )
    return result


@pytest.mark.parametrize(
    "producers,total,expected",
    [
        (128, 256, [[24, 0], [240, 1]]),
        (128, 384, [[24, 0], [240, 1]]),
        (256, 512, [[24, 0], [224, 1]]),
        (128, 512, [[24, 0], [160, 1]]),
        (64, 256, []),
        (384, 512, []),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_register_budget_and_nested_scopes(producers, total, expected, nested):
    result = lower(kernel(producers, total, nested=nested))
    assert [[int(argument) for argument in call.args] for call in calls(result)] == expected


def test_simt_producer_suppresses_only_automatic_reconfiguration():
    assert not calls(lower(kernel(simt=True)))
    hints = [("annotate_producer_reg_dealloc", [ir.const(32)]), ("annotate_consumer_reg_alloc", [ir.const(128)])]
    result = lower(kernel(simt=True, hints=hints))
    assert [[int(argument) for argument in call.args] for call in calls(result)] == [[32, 0], [128, 1]]
    assert not calls(result, "tl.annotate_consumer_reg_alloc")


def test_disabled_and_custom_register_annotations_preserve_explicit_calls():
    explicit = ("set_max_nreg", [ir.const(40), ir.const(0)])
    for module in (kernel(hints=[explicit], custom=True), kernel(hints=[explicit, ("no_set_max_nreg", [])])):
        result = lower(module)
        assert [[int(argument) for argument in call.args] for call in calls(result)] == [[40, 0]]


def test_invalid_and_oversubscribed_register_hints_are_rejected():
    with pytest.raises(ValueError, match="multiple of 8"):
        lower(kernel(hints=[("annotate_producer_reg_dealloc", [ir.const(27)])]))
    with pytest.raises(ValueError, match="register budget"):
        lower(kernel(128, 512, hints=[("annotate_consumer_reg_alloc", [ir.const(240)])]))


def test_register_reconfiguration_compiles_and_executes(tilelang_builders):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    architecture = device.compute_version.replace(".", "")
    if int(architecture) < 90:
        pytest.skip("Warpgroup register reconfiguration requires Hopper or later")
    root = Path(__file__).resolve().parents[3]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{architecture}a",
            "backend_config": {
                "cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"]}
            },
        },
        host="c",
    )
    module = lower(kernel(128, 256))
    executable = tirx.compile(module, target=target)
    actual = runtime.empty((256,), "int32", device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.repeat(np.array([1, 2], dtype="int32"), 128))
